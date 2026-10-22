import os
import re
import warnings

import numpy as np
import pandas as pd
from langchain_core.tools import tool

print("⚡ Initializing Data Analyst Backend...")

MAX_CAT_UNIQUE = 20        # above this, a text column is identifier/free text
MAX_GROUP_UNIQUE = 12      # categories used for group-by comparisons
MAX_TEXT_LEN = 60          # avg string length above this = free text
CORR_THRESHOLD = 0.3       # |r| needed to count as a relationship
REDUNDANT_R = 0.95         # |r| at/above this = near-duplicate measures, not a finding
MIN_GROUP = 10             # groups/months smaller than this are too thin to compare
MAX_OUTLIER_SHARE = 0.10   # if >10% of rows are "outliers", the column is just skewed
BLOCK_CAP = 1800           # max chars per report section (token control)


# ---------------------------------------------------------------- helpers
def _cap(text: str) -> str:
    return text if len(text) <= BLOCK_CAP else text[:BLOCK_CAP] + "\n[...truncated...]"


def _fmt(x: float, year: bool = False) -> str:
    if year:
        return f"{x:.0f}"
    return f"{x:,.1f}" if abs(x) < 1e6 else f"{x:,.0f}"


def _strength(r: float) -> str:
    a = abs(r)
    for limit, word in ((0.1, "negligible"), (0.3, "weak"), (0.5, "moderate"), (0.7, "strong")):
        if a < limit:
            return word
    return "very strong"


def _is_id_like(name, s: pd.Series) -> bool:
    """Row IDs and counters carry no analytical signal."""
    if re.search(r"(^|[\s_\-])(id|index|idx)$", str(name).strip(), re.I):
        return True
    # A true row counter is a consecutive sequence (1,2,3...). Merely unique
    # values (e.g. employee counts) are real metrics and stay in.
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


def _corr_pairs(df: pd.DataFrame, cols: list) -> list:
    corr = df[cols].corr()
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r):
                out.append((a, b, float(r)))
    out.sort(key=lambda t: abs(t[2]), reverse=True)
    return out


def _family(name) -> str:
    """Columns measuring the same thing (temp / tempmax / feelslikemin ...) share a family."""
    n = str(name).lower().replace("feelslike", "temp")
    return re.sub(r"(max|min|avg|mean)$", "", n).strip("_ -")


def _split_pairs(pairs: list):
    """(genuine pairs, redundant pairs). Redundant = near-identical OR same measure family."""
    real, redundant = [], []
    for a, b, r in pairs:
        if abs(r) >= REDUNDANT_R or _family(a) == _family(b):
            redundant.append((a, b, r))
        else:
            real.append((a, b, r))
    return real, redundant


def _outlier_counts(df: pd.DataFrame, cols: list) -> dict:
    """IQR outliers, skipping zero-inflated or heavily skewed columns (rain, etc.)
    where 'outliers' are just normal behaviour."""
    res = {}
    for c in cols:
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
            res[c] = (k, lo, hi)
    return dict(sorted(res.items(), key=lambda kv: kv[1][0], reverse=True))


def _monthly(df: pd.DataFrame, dates: pd.Series, cols: list) -> pd.DataFrame:
    """Monthly means, dropping thin (partial) months."""
    d = df[cols].copy()
    d["_m"] = dates.dt.to_period("M")
    d = d.dropna(subset=["_m"])
    counts = d.groupby("_m").size()
    m = d.groupby("_m")[cols].mean()
    return m[counts.reindex(m.index) >= MIN_GROUP]


# ---------------------------------------------------------------- report blocks
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
    res = _outlier_counts(df, cols)
    if not res:
        return ("No meaningful IQR outliers (skewed or zero-inflated columns such as "
                "rainfall are excluded).")
    lines = [f"{c}: {k} outliers outside [{lo:.1f}, {hi:.1f}]"
             for c, (k, lo, hi) in list(res.items())[:8]]
    return _cap("\n".join(lines))


def _categorical_block(df: pd.DataFrame, cols: list):
    """Returns (text for low-cardinality columns, skipped columns, group-able columns)."""
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
        sizes = df[c].value_counts()
        keep = sizes[sizes >= MIN_GROUP].index
        if len(keep) < 2:
            continue
        g = df[df[c].isin(keep)].groupby(c)[num_cols[:3]].mean().round(1)
        g["rows"] = sizes.reindex(g.index)
        blocks.append(f"Mean by {c} (groups under {MIN_GROUP} rows omitted):\n{g.to_string()}")
    return _cap("\n\n".join(blocks)) if blocks else "No groups large enough to compare."


def _correlation_block(df: pd.DataFrame, cols: list) -> str:
    if len(cols) < 2:
        return "Not enough numeric columns for correlation analysis."
    pairs = _corr_pairs(df, cols)
    if not pairs:
        return "Correlations could not be computed."
    real, redundant = _split_pairs(pairs)
    strong = [p for p in real if abs(p[2]) >= CORR_THRESHOLD]
    lines = []
    if strong:
        lines.append(f"Genuine relationships between different measures (|r| >= {CORR_THRESHOLD}):")
        lines += [f"  {a} <-> {b}: r={r:+.2f}" for a, b, r in strong[:12]]
    elif real:
        a, b, r = real[0]
        lines.append(f"No pair reaches |r| >= {CORR_THRESHOLD}. Strongest: {a} <-> {b} (r={r:+.2f}).")
    if redundant:
        lines.append("Same-measure or near-duplicate pairs (derived from each other, NOT insights):")
        lines += [f"  {a} <-> {b}: r={r:+.2f}" for a, b, r in redundant[:8]]
    return _cap("\n".join(lines))


def _date_block(date_cols: dict) -> str:
    if not date_cols:
        return "No date columns detected."
    lines = []
    for c, s in date_cols.items():
        s = s.dropna()
        lines.append(f"{c}: {s.min().date()} to {s.max().date()} (span {(s.max() - s.min()).days} days)")
    return _cap("\n".join(lines))


def _trend_block(df: pd.DataFrame, date_cols: dict, numeric_cols: list) -> str:
    if not date_cols or not numeric_cols:
        return ""
    dcol = next(iter(date_cols))
    pref = [c for c in ("temp", "tempmax", "tempmin") if c in numeric_cols]
    cols = (pref or numeric_cols)[:3]
    m = _monthly(df, date_cols[dcol], cols)
    if len(m) < 3:
        return ""
    m = m.round(1)
    m.index = m.index.astype(str)
    return _cap(f"Monthly mean by {dcol} (months under {MIN_GROUP} rows omitted):\n{m.to_string()}")


# ---------------------------------------------------------------- plain-English narrative
def _narrative(df, numeric_cols, text_cols, date_cols, id_cols, skipped, group_cats) -> str:
    """Prose summary generated locally. No tables, no LLM."""
    n, c = len(df), df.shape[1]
    P = []

    dup = int(df.duplicated().sum())
    P.append(f"**Overview.** The dataset contains {n:,} records across {c} columns"
             + (f", including {dup} duplicate rows." if dup else " with no duplicate rows."))

    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        P.append("**Data quality.** Nothing is missing, so every column is complete.")
    else:
        parts = ", ".join(f"{k} ({v / n * 100:.1f}%)" for k, v in miss.head(5).items())
        P.append(f"**Data quality.** Some values are missing, mainly in {parts}.")

    sents = []
    shown = numeric_cols[:8]
    for col in shown:
        s = df[col].dropna()
        if s.empty:
            continue
        year = bool(s.min() >= 1800 and s.max() <= 2100 and (s % 1 == 0).all())
        t = (f"{col} averages {_fmt(s.mean(), year)} (typical value {_fmt(s.median(), year)}), "
             f"ranging from {_fmt(s.min(), year)} to {_fmt(s.max(), year)}")
        sk = s.skew()
        if abs(sk) > 1:
            t += f", and leans toward {'high' if sk > 0 else 'low'} values"
        sents.append(t + ".")
    if sents:
        extra = len(numeric_cols) - len(shown)
        tail = (f" {extra} more numeric column{'s are' if extra != 1 else ' is'} profiled "
                f"in the raw payload.") if extra > 0 else ""
        P.append("**Numbers.** " + " ".join(sents) + tail)

    cat_sents = []
    for col in text_cols:
        s = df[col].dropna().astype(str)
        if s.empty or s.str.len().mean() > MAX_TEXT_LEN or s.nunique() > MAX_CAT_UNIQUE:
            continue
        vc = s.value_counts()
        if s.nunique() == 1:
            cat_sents.append(f"{col} has a single value ({vc.index[0]}).")
        else:
            cat_sents.append(f"{col} has {s.nunique()} categories, led by {vc.index[0]} "
                             f"({vc.iloc[0] / len(s) * 100:.0f}%).")
    if cat_sents:
        P.append("**Categories.** " + " ".join(cat_sents[:8]))

    g_sents = []
    for cat in group_cats[:2]:
        sizes = df[cat].value_counts()
        keep = sizes[sizes >= MIN_GROUP].index
        if len(keep) < 2:
            continue
        sub = df[df[cat].isin(keep)]
        for num in numeric_cols[:2]:
            m = sub.groupby(cat)[num].mean().dropna()
            if len(m) < 2:
                continue
            hi, lo = m.idxmax(), m.idxmin()
            g_sents.append(f"By {cat}, average {num} is highest for {hi} ({_fmt(m[hi])}, "
                           f"{sizes[hi]} rows) and lowest for {lo} ({_fmt(m[lo])}, {sizes[lo]} rows).")
    if g_sents:
        P.append("**Comparisons.** " + " ".join(g_sents)
                 + f" Groups under {MIN_GROUP} rows are left out as too thin to compare.")

    out = _outlier_counts(df, numeric_cols)
    P.append("**Unusual values.** "
             + (("Genuine outliers were found in "
                 + ", ".join(f"{c} ({k})" for c, (k, _, _) in list(out.items())[:6]) + ".")
                if out else "No meaningful outliers were found. Skewed columns such as rainfall "
                            "are excluded because their extremes are normal."))

    if date_cols:
        d = []
        for col, s in date_cols.items():
            s = s.dropna()
            if not s.empty:
                d.append(f"{col} runs from {s.min().date()} to {s.max().date()}")
        if d:
            P.append("**Dates.** " + "; ".join(d) + ".")

        dcol = next(iter(date_cols))
        col = "temp" if "temp" in numeric_cols else (numeric_cols[0] if numeric_cols else None)
        if col:
            mm = _monthly(df, date_cols[dcol], [col])[col]
            if len(mm) >= 3:
                hi, lo = mm.idxmax(), mm.idxmin()
                P.append(f"**Over time.** Monthly average {col} peaks in {hi} ({_fmt(mm[hi])}) and is "
                         f"lowest in {lo} ({_fmt(mm[lo])}), a swing of {_fmt(mm[hi] - mm[lo])}. "
                         f"Months with fewer than {MIN_GROUP} rows are excluded.")

    if len(numeric_cols) >= 2:
        pairs = _corr_pairs(df, numeric_cols)
        real, redundant = _split_pairs(pairs)
        strong = [p for p in real if abs(p[2]) >= CORR_THRESHOLD]
        if strong:
            txt = "; ".join(
                f"{a} and {b} {'rise together' if r > 0 else 'move in opposite directions'} "
                f"({_strength(r)}, r = {r:+.2f})" for a, b, r in strong[:4])
            P.append(f"**Relationships.** The clearest relationships are: {txt}.")
        elif real:
            a, b, r = real[0]
            P.append(f"**Relationships.** None of the numeric columns are meaningfully related. "
                     f"The closest pair is {a} and {b} ({_strength(r)}, r = {r:+.2f}).")
        if redundant:
            names = "; ".join(f"{a} and {b}" for a, b, _ in redundant[:5])
            P.append(f"**Redundant measures.** {names} describe the same underlying measure or "
                     f"are nearly identical, so their link is built in, not a finding.")

    left_out = list(id_cols) + list(skipped)
    if left_out:
        P.append(f"**Not analyzed.** {', '.join(map(str, left_out))} were left out of the statistics "
                 f"because they are identifiers, timestamps, constants, or free text.")

    return "\n\n".join(P)


# ---------------------------------------------------------------- engine
@tool
def expert_data_analyst_engine(csv_filename: str) -> dict:
    """
    Profiles a local CSV dataset: data quality, numeric statistics, categorical
    breakdowns, group comparisons, outliers, date ranges, monthly trends and
    meaningful correlations. Returns compact text sections sized for an LLM plus
    structural metadata for charts.
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
        numeric_cols = [c for c in all_numeric
                        if not _is_id_like(c, df[c]) and df[c].nunique(dropna=True) > 1]
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
            "trend_text": _trend_block(df, date_cols, numeric_cols),
            "narrative_text": _narrative(
                df, numeric_cols, text_cols, date_cols, id_cols, skipped, group_cats
            ),
            "skipped_note": (
                f"Excluded ID-like/constant numeric columns: {id_cols}. "
                f"Skipped identifier/timestamp/free-text/high-cardinality columns: {skipped}."
            ),
        }

    except Exception as data_fault:
        return {"error": f"Pipeline execution failed: {str(data_fault)}"}
