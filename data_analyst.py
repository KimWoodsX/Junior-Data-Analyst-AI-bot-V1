"""
data_agent.py
Local CSV profiler & data analysis backend engine.
Zero data retention; produces plain-English summaries and statistical metadata.
"""

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

# ----------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# ----------------------------------------------------------------------------
MAX_CAT_UNIQUE = 20        # Above this, text is treated as identifier/free text
MAX_GROUP_UNIQUE = 12      # Max unique categories used for group comparisons
MAX_TEXT_LEN = 60          # Average string length above this = free text
CORR_THRESHOLD = 0.3       # |r| threshold for a meaningful relationship
REDUNDANT_R = 0.95         # |r| threshold for near-duplicate/derived metrics
MIN_GROUP = 10             # Minimum group size for reliable comparison
MAX_OUTLIER_SHARE = 0.10   # Columns with >10% extreme rows are considered skewed
MIN_ENTITIES = 10          # Distinct values needed to identify panel/entity data
BLOCK_CAP = 1800           # Character ceiling per prompt section

PLACEHOLDERS = ["-", "--", "\\N", "N/A", "NA", "n/a", "null", "NULL", "None", "", "nan", "NaN"]
GEO_NAMES = {"lat", "lng", "lon", "long", "latitude", "longitude"}
SOLAR_NAMES = {"solarradiation", "solarenergy", "uvindex"}
DERIVED_NAMES = {"dew", "dewpoint", "heatindex", "windchill"}


# ----------------------------------------------------------------------------
# FORMATTING & UTILITY HELPERS
# ----------------------------------------------------------------------------
def _cap(text: str) -> str:
    """Enforces block token limit to avoid prompt bloat."""
    if len(text) <= BLOCK_CAP:
        return text
    return text[:BLOCK_CAP] + "\n[...truncated...]"


def _fmt(x: float, year: bool = False) -> str:
    """Formats numbers into human-readable strings."""
    if pd.isna(x):
        return "N/A"
    if year:
        return f"{x:.0f}"
    if abs(x) >= 1e6:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:,.1f}"
    return f"{x:.2f}"


def _join(items: List[Any]) -> str:
    """Grammatically joins a list with commas and 'and'."""
    items = [str(i) for i in items if str(i).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _strength(r: float) -> str:
    """Maps correlation coefficient to descriptive strength."""
    a = abs(r)
    for limit, word in ((0.1, "negligible"), (0.3, "weak"), (0.5, "moderate"), (0.7, "strong")):
        if a < limit:
            return word
    return "very strong"


# ----------------------------------------------------------------------------
# DATA CLEANING & TYPE INFERENCE
# ----------------------------------------------------------------------------
def _load(path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Reads CSV, treating placeholders as missing values and converting 
    text-formatted numbers (e.g. '1,200', '3.5%', '$45.00') into numerics.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    ph = {c: int(raw[c].str.strip().isin(PLACEHOLDERS).sum()) for c in raw.columns}
    ph = {c: n for c, n in ph.items() if n > 0}

    df = pd.read_csv(path, na_values=PLACEHOLDERS, encoding="utf-8-sig")

    # Clean text columns that are actually numbers
    for c in df.select_dtypes(exclude=[np.number]).columns:
        if re.search(r"date|time", str(c), re.I):
            continue
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        cleaned = df[c].astype(str).str.replace(r"[,$%£€\s]", "", regex=True)
        num = pd.to_numeric(cleaned, errors="coerce")
        if num[df[c].notna()].notna().mean() >= 0.8:
            df[c] = num

    return df, ph


def _is_id_like(name: str, s: pd.Series) -> bool:
    """Detects primary keys, sequential indexes, and identifier counters."""
    n = str(name).strip()
    if re.search(r"(^|[\s_\-])(id|index|idx)$", n, re.I):
        return True
    if re.search(r"[a-z0-9](Id|ID)$", n):  # camelCase: orderId, userId
        return True
    if pd.api.types.is_integer_dtype(s) and len(s) > 1 and s.notna().all():
        diffs = s.diff().dropna()
        if (diffs == 1).all():
            return True
    return False


def _detect_dates(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Identifies date and timestamp columns."""
    found = {}
    for c in df.select_dtypes(exclude=[np.number]).columns:
        if not re.search(r"date|time|timestamp", str(c), re.I):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.7:
            found[c] = parsed
    return found


def _family(name: str) -> str:
    """Normalizes column names to group related metrics together."""
    n = str(name).lower()
    if n in SOLAR_NAMES:
        return "solar"
    n = n.replace("feelslike", "temp")
    return re.sub(r"[\s_\-]*(max|min|avg|mean|low|high|price|open|close|adjusted)$", "", n).strip("_ -")


def _families(cols: List[str]) -> List[List[str]]:
    fams: Dict[str, List[str]] = {}
    for c in cols:
        fams.setdefault(_family(c), []).append(c)
    return list(fams.values())


# ----------------------------------------------------------------------------
# STATISTICAL METRICS & CORRELATIONS
# ----------------------------------------------------------------------------
def _corr_pairs(df: pd.DataFrame, cols: List[str]) -> List[Tuple[str, str, float]]:
    if len(cols) < 2:
        return []
    corr = df[cols].corr()
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r):
                out.append((a, b, float(r)))
    out.sort(key=lambda t: abs(t[2]), reverse=True)
    return out


def _split_pairs(pairs: List[Tuple[str, str, float]]):
    real, redundant = [], []
    for a, b, r in pairs:
        if (abs(r) >= REDUNDANT_R or _family(a) == _family(b)
                or str(a).lower() in DERIVED_NAMES or str(b).lower() in DERIVED_NAMES):
            redundant.append((a, b, r))
        else:
            real.append((a, b, r))
    return real, redundant


def _outlier_counts(df: pd.DataFrame, cols: List[str]) -> Dict[str, Tuple[int, float, float]]:
    res = {}
    for c in cols:
        if re.search(r"dir|phase|moon|year|code|zip", str(c), re.I):
            continue
        s = df[c].dropna()
        if len(s) < 8 or (s == 0).mean() > 0.5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        k = int(((s < lo) | (s > hi)).sum())
        if 0 < k <= MAX_OUTLIER_SHARE * len(s):
            res[c] = (k, float(lo), float(hi))
    return dict(sorted(res.items(), key=lambda kv: kv[1][0], reverse=True))


def _find_entity(df: pd.DataFrame, text_cols: List[str], date_cols: Dict[str, pd.Series]) -> Optional[str]:
    """Detects if dataset is panel/timeseries grouped by entity (e.g. stock ticker, country)."""
    if not date_cols:
        return None
    n = len(df)
    for c in text_cols:
        s = df[c].dropna().astype(str)
        k = s.nunique()
        if MIN_ENTITIES <= k <= max(MIN_ENTITIES + 1, n / 5) and s.str.len().mean() <= MAX_TEXT_LEN:
            return c
    return None


def _pick(cols: List[str], pattern: str) -> Optional[str]:
    for c in cols:
        if re.search(pattern, str(c), re.I):
            return c
    return None


def _enti
