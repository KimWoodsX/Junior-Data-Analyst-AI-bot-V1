import os
import re
import warnings

import numpy as np
import pandas as pd
from langchain_core.tools import tool

print("⚡ Initializing Data Analyst Backend...")

MAX_CAT_UNIQUE = 20     # above this, a text column is treated as identifier/free text
MAX_GROUP_UNIQUE = 12   # categories used for group-by comparisons
MAX_TEXT_LEN = 60       # avg string length above this = free text, skip
CORR_THRESHOLD = 0.3    # |r| needed to be reported as a relationship
BLOCK_CAP = 1800        # max chars per report section (token control)


def _cap(text: str) -> str:
    return text if len(text) <= BLOCK_CAP else text[:BLOCK_CAP] + "\n[...truncated...]"


def _is_id_like(name, s: pd.Series) -> bool:
    """Row IDs and indexes carry no analytical signal; keep them out of stats."""
    if re.search(r"(^|[\s_\-])(id|index|idx)$", str(name).strip(), re.I):
        return True
    # A true row counter is a consecutive sequence (1,2,3...). Merely unique values
    # (e.g. employee counts) are real metrics and must stay in the analysis.
    if pd.api.types.is_integer_dtype(s) and len(s) > 1 and s.notna().all():
        return bool((s.diff().dropna() == 1).all())
    return False


def _detect_dates(df: pd.DataFrame) -> dict:
    """Parse columns whose names hint at dates. Returns {col: datetime Series}."""
    found = {}
    for c in df.select_dtypes(include="object").columns:
        if not re.search(r"date|time", str(c), re.I):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.8:
            found[c] = parsed
    return found


def _quality_block(df: pd.DataFrame) -> str:
    lines = [f"Rows: {len(df)} | Columns: {df.shape[1]} | Duplicate rows: {int(df.duplicated().sum())}"]
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        lines.append("Missing values: none")
    else:
        lines.append("Missing values (count, %):")
        for col, n in miss.head(10).items():
            lines.append(f"  {col}: {n} ({n / len(df) * 100:.1f}%)")
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const:
        lines.append(f"Constant columns (no information): {const}")
    return _cap("\n".join(lines))


def _numeric_block(df: pd.DataFrame, cols: list) -> str:
    if not cols:
        return "No analyzable numeric columns."
    desc = df[cols].describe().T
    desc["skew"] = df[cols].skew()
    return _cap(desc.round(2).to_string())


def _outlier_block(df: pd.DataFrame, cols: list) -> str:
    lines = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = int(((s < lo) | (s > hi)).sum())
        if n:
            lines.append(f"{c}: {n} outliers outside [{lo:.1f}, {hi:.1f}]")
    return _cap("\n".join(lines)) if lines else "No IQR outliers detected in any numeric column."


def _categorical_block(df: pd.DataFrame, cols: list):
    """Returns (text for low-cardinality columns, list of skipped columns)."""
    lines, skipped, low_card = [], [], []
    for c in cols:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        if s.str.len().mean() > MAX_TEXT_LEN or s.nunique() > MAX_CAT_UNIQUE:
            skipped.append(c)
            continue
        vc = s.value_counts()
        top = ", ".join(f"{k} ({v}, {v / len(s) * 100:.0f}%)" for k, v in vc.head(5).items())
        lines.append(f"{c} [{s.nunique()} unique]: {top}")
        if 2 <= s.nunique() <= MAX_GROUP_UNIQUE:
            low_card.append(c)
    text = "\n".join(lines) if lines else "No low-cardinality categorical columns."
    return _cap(text), skipped, low_card


def _group_block(df: pd.DataFrame, cat_cols: list, num_cols: list) -> str:
    if not cat_cols or not num_cols:
        return "Not enough categorical/numeric columns for group comparison."
    blocks = []
    for c in cat_cols[:2]:
        g = df.groupby(c)[num_cols[:3]].mean().round(1)
        blocks.append(f"Mean by {c}:\n{g.to_string()}")
    return _cap("\n\n".join(blocks))


def _correlation_block(df: pd.DataFrame, cols: list) -> str:
    if len(cols) < 2:
        return "Not enough numeric columns for correlation analysis."
    corr = df[cols].corr()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r):
                pairs.append((a, b, float(r)))
    if not pairs:
        return "Correlations could not be computed."
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    strong = [p for p in pairs if abs(p[2]) >= CORR_THRESHOLD]
    if strong:
        body = "\n".join(f"{a} <-> {b}: r={r:+.2f}" for a, b, r in strong[:10])
        return _cap(f"Pairs with |r| >= {CORR_THRESHOLD}:\n{body}")
    a, b, r = pairs[0]
    return (f"No pair reaches |r| >= {CORR_THRESHOLD}. Numeric columns are essentially "
            f"uncorrelated. Strongest: {a} <-> {b} (r={r:+.2f}).")


def _date_block(date_cols: dict) -> str:
    if not date_cols:
        return "No date columns detected."
    lines = []
    for c, s in date_cols.items():
        s = s.dropna()
        lines.append(f"{c}: {s.min().date()} to {s.max().date()} (span {(s.max() - s.min()).days} days)")
    return _cap("\n".join(lines))


@tool
def expert_data_analyst_engine(csv_filename: str) -> dict:
    """
    Profiles a local CSV dataset: data quality, numeric statistics, categorical
    breakdowns, group comparisons, outliers, date ranges and meaningful correlations.
    Returns compact text sections sized for an LLM plus structural metadata for charts.
    """
    print(f"📦 Data pipeline running on: '{csv_filename}'")

    if not os.path.exists(csv_filename):
        return {"error": f"Targeted data file '{csv_filename}' cannot be located."}

    try:
        df = pd.read_csv(csv_filename)
        if df.empty:
            return {"error": "The dataset contains no rows."}

        date_cols = _detect_dates(df)
        all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in all_numeric if not _is_id_like(c, df[c])]
        id_cols = [c for c in all_numeric if c not in numeric_cols]
        text_cols = [c for c in df.columns if c not in all_numeric and c not in date_cols]

        cat_text, skipped, group_cats = _categorical_block(df, text_cols)

        return {
            # --- contract used by the Streamlit app ---
            "file_path": csv_filename,
            "numeric_columns": numeric_cols,
            "time_axis": str(next(iter(date_cols), df.columns[0])),
            "summary_text": _numeric_block(df, numeric_cols),
            "correlation_text": _correlation_block(df, numeric_cols),
            "total_records": len(df),
            "column_layout": list(df.columns),
            # --- richer sections ---
            "quality_text": _quality_block(df),
            "categorical_text": cat_text,
            "group_text": _group_block(df, group_cats, numeric_cols),
            "outlier_text": _outlier_block(df, numeric_cols),
            "date_text": _date_block(date_cols),
            "skipped_note": (
                f"Excluded ID-like numeric columns: {id_cols}. "
                f"Skipped identifier/free-text/high-cardinality columns: {skipped}."
            ),
        }

    except Exception as data_fault:
        return {"error": f"Pipeline execution failed: {str(data_fault)}"}
