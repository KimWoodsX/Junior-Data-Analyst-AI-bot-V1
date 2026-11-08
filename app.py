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
MIN_ENTITIES = 10          # a repeating text column with this many values = "one series per entity"
BLOCK_CAP = 1800           # max chars per report section (token control)

PLACEHOLDERS = ["-", "--", "\\N", "N/A", "NA", "n/a", "null", "NULL", "None", ""]
GEO_NAMES = {"lat", "lng", "lon", "long", "latitude", "longitude"}


# ---------------------------------------------------------------- helpers
def _cap(text: str) -> str:
    return text if len(text) <= BLOCK_CAP else text[:BLOCK_CAP] + "\n[...truncated...]"


def _fmt(x: float, year: bool = False) -> str:
    if year:
        return f"{x:.0f}"
    if abs(x) >= 1e6:
        return f"{x:,.0f}"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:,.1f}"
    return f"{x:.2f}"


def _join(items) -> str:
    items = [str(i) for i in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _strength(r: float) -> str:
    a = abs(r)
    for limit, word in ((0.1, "negligible"), (0.3, "weak"), (0.5, "moderate"), (0.7, "strong")):
        if a < limit:
            return word
    return "very strong"


def _load(path: str):
    """Read a CSV treating placeholders ('-', '\\N', 'N/A'...) as missing and turning
    numeric-looking text (1,200 / 3.5%) into numbers. Returns (df, placeholder_counts)."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    ph = {c: int(raw[c].str.strip().isin(PLACEHOLDERS).sum()) for c in raw.columns}
    ph = {c: n for c, n in ph.items() if n > 0}

    df = pd.read_csv(path, na_values=PLACEHOLDERS, encoding="utf-8-sig")
    for c in df.select_dtypes(exclude=[np.number]).columns:
        if re.search(r"date|time", str(c), re.I):
            continue
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        num = pd.to_numeric(df[c].astype(str).str.replace(r"[,%]", "", regex=True), errors="coerce")
        if num[df[c].notna()].notna().mean() >= 0.8:
            df[c] = num
    return df, ph


def _is_id_like(name, s: pd.Series) -> bool:
    """Row IDs and counters carry no analytical signal."""
    n = str(name).strip()
    if re.search(r"(^|[\s_\-])(id|index|idx)$", n, re.I):
        return True
    if re.search(r"[a-z0-9](Id|ID)$", n):          # camelCase: circuitId, userID
        return True
    if pd.api.types.is_integer_dtype(s) and len(s) > 1 and s.notna().all():
        return bool((s.diff().dropna() == 1).all())
    return False


def _detect_dates(df: pd.DataFrame) -> dict:
    found = {}
    for c in df.select_dtypes(exclude=[np.number]).columns:
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
    """Columns measuring the same thing (temp / tempmax, Day Low / Day High / Day Price...)."""
    n = str(name).lower()
    return re.sub(r"[\s_\-]*(max|min|avg|mean|low|high|price|open|close)$", "", n).strip("_ -")


def _split_pairs(pairs: list):
    real, redundant = [], []
    for a, b, r in pairs:
        if abs(r) >= REDUNDANT_R or _family(a) == _family(b):
            redundant.append((a, b, r))
        else:
            real.append((a, b, r))
    return real, redundant


def _families(cols: list) -> list:
    """Group columns into families, preserving order."""
    fams = {}
    for c in cols:
        fams.setdefault(_family(c), []).append(c)
    return list(fams.values())


def _outlier_counts(df: pd.DataFrame, cols: list) -> dict:
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
    d = df[cols].copy()
    d["_m"] = dates.dt.to_period("M")
    d = d.dropna(subset=["_m"])
    counts = d.groupby("_m").size()
    m = d.groupby("_m")[cols].mean()
    return m[counts.reindex(m.index) >= MIN_GROUP]


def _find_entity(df: pd.DataFrame, text_cols: list, date_cols: dict):
    """A text column with many repeating values + a date column = one time series per entity
    (stocks, stations, customers...). Pooled averages across entities are misleading."""
    if not date_cols:
        return None
    n = len(df)
    for c in text_cols:
        s = df[c].dropna().astype(str)
        k = s.nunique()
        if MIN_ENTITIES <= k <= n / 5 and s.str.len().mean() <= MAX_TEXT_LEN:
            return c
    return None


def _pick(cols: list, pattern: str):
    for c in cols:
        if re.search(pattern, str(c), re.I):
            return c
    return None


def _entity_table(df, entity, date_series, price_col, volume_col):
    d = df[[entity] + [c for c in (price_col, volume_col) if c]].copy()
    d["_d"] = date_series
    d = d.dropna(subset=["_d"]).sort_values("_d")
    out = pd.DataFrame(index=d[entity].unique())
    if price_col:
        p = d.dropna(subset=[price_col])
        p = p[p[price_col] > 0]
        g = p.groupby(entity)[price_col]
        out["first"], out["last"] = g.first(), g.last()
        out["ret"] = (out["last"] / out["first"] - 1) * 100
        out["vol"] = g.apply(lambda s: s.pct_change().std() * 100)
    if volume_col:
        g = d.groupby(entity)[volume_col]
        out["total_vol"] = g.sum()
        out["trade_share"] = g.count() / g.size()
    return out.dropna(how="all")


# ---------------------------------------------------------------- report blocks
def _quality_block(df: pd.DataFrame, ph: dict) -> str:
    lines = [f"Rows: {len(df)} | Columns: {df.shape[1]} | Duplicate rows: {int(df.duplicated().sum())}"]
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        lines.append("Missing values: none")
    else:
        lines.append("Missing values (count, %), placeholders such as '-' counted as missing:")
        for col, n in miss.head(10).items():
            lines.append(f"  {col}: {n} ({n / len(df) * 100:.1f}%)")
    const = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    if const:
        lines.append(f"Constant or empty columns (no information): {const}")
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
        return ("No columns with a small set of extreme values. Columns where more than "
                f"{MAX_OUTLIER_SHARE:.0%} of rows are extreme are skipped as simply skewed.")
    lines = [f"{c}: {k} outliers outside [{lo:.1f}, {hi:.1f}]"
             for c, (k, lo, hi) in list(res.items())[:8]]
    return _cap("\n".join(lines))


def _categorical_block(df: pd.DataFrame, cols: list):
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


def _trend_block(df: pd.DataFrame, date_cols: dict, numeric_cols: list, panel: bool) -> str:
    if panel or not date_cols or not numeric_cols:
        return ""   # monthly means across different entities are not a real trend
    dcol = next(iter(date_cols))
    cols = numeric_cols[:3]
    m = _monthly(df, date_cols[dcol], cols)
    if len(m) < 3:
        return ""
    m = m.round(1)
    m.index = m.index.astype(str)
    return _cap(f"Monthly mean by {dcol} (months under {MIN_GROUP} rows omitted):\n{m.to_string()}")


def _entity_block(tbl: pd.DataFrame, entity: str) -> str:
    if tbl is None or tbl.empty:
        return ""
    parts = []
    if "ret" in tbl:
        r = tbl["ret"].dropna().sort_values(ascending=False).round(1)
        parts.append(f"Return first->last by {entity} (top 5):\n{r.head(5).to_string()}")
        parts.append(f"Bottom 5:\n{r.tail(5).to_string()}")
    if "total_vol" in tbl:
        parts.append(f"Highest total volume:\n{tbl['total_vol'].dropna().sort_values(ascending=False).head(5).round(0).to_string()}")
    return _cap("\n\n".join(parts))


# ---------------------------------------------------------------- plain-English narrative
def _describe_family(df: pd.DataFrame, cols: list, i: int = 0) -> str:
    col = cols[0]
    s = df[col].dropna()
    year = bool(s.min() >= 1800 and s.max() <= 2100 and (s % 1 == 0).all())
    mean, med, lo, hi, sk = s.mean(), s.median(), s.min(), s.max(), s.skew()
    if sk > 1 and mean > med * 1.5:
        if i % 2 == 0:
            t = (f"A typical {col} is around {_fmt(med, year)}, though values run all the way from "
                 f"{_fmt(lo, year)} to {_fmt(hi, year)} and a small number of very high ones drag the "
                 f"average up to {_fmt(mean, year)}.")
        else:
            t = (f"{col} has the same lopsided shape: the middle value is {_fmt(med, year)}, "
                 f"but a few big numbers (up to {_fmt(hi, year)}) lift the mean to {_fmt(mean, year)}.")
    elif sk < -1:
        t = (f"{col} averages {_fmt(mean, year)} and ranges from {_fmt(lo, year)} to {_fmt(hi, year)}, "
             f"with a handful of unusually low values pulling the average down.")
    else:
        t = f"{col} averages {_fmt(mean, year)}, ranging from {_fmt(lo, year)} to {_fmt(hi, year)}."
    if len(cols) > 1:
        t += f" {_join(cols[1:])} follow{'s' if len(cols) == 2 else ''} much the same pattern."
    return t


def _narrative(df, numeric_cols, text_cols, date_cols, id_cols, skipped, group_cats,
               ph, empty_cols, entity, tbl, price_col, volume_col) -> str:
    n, c = len(df), df.shape[1]
    P = []

    # --- opening ---
    s = f"This is a table of {n:,} rows and {c} columns"
    if entity:
        s += f", tracking {df[entity].nunique()} different {entity} values over time"
    s += "."
    if date_cols:
        col = next(iter(date_cols))
        d = date_cols[col].dropna()
        if not d.empty:
            s += f" The records run from {d.min():%d %b %Y} to {d.max():%d %b %Y}."
    dup = int(df.duplicated().sum())
    s += f" {dup} rows are exact duplicates." if dup else " There are no duplicate rows."
    P.append(s)

    # --- data quality ---
    miss = (df.isna().sum() / n).sort_values(ascending=False)
    miss = miss[(miss > 0) & ~miss.index.isin(empty_cols)]
    q = []
    if empty_cols:
        q.append(f"{_join(empty_cols)} {'has' if len(empty_cols) == 1 else 'have'} no real data at all")
    if not miss.empty:
        big = [f"{k} (about {v:.0%})" for k, v in miss.head(4).items() if v >= 0.01]
        if big:
            q.append(f"{_join(big)} {'is' if len(big) == 1 else 'are'} blank for a good share of rows")
    if q:
        msg = "On data quality, " + "; ".join(q) + "."
        if ph:
            msg += (" In the file those gaps are written as a placeholder like a dash rather than "
                    "left empty, so a quick check would call the data complete when it isn't. "
                    "Here they're treated as missing.")
        P.append(msg)
    else:
        P.append("On data quality, nothing is missing, so every column is complete.")

    # --- numbers ---
    fams = _families(numeric_cols)[:4]
    if fams:
        scale = ""
        if entity:
            scale = ("Because the values come from many different " + entity + " entries at very "
                     "different scales, these overall figures are best read as a rough guide. ")
        body = " ".join(_describe_family(df, f, i) for i, f in enumerate(fams))
        rest = len(_families(numeric_cols)) - len(fams)
        tail = f" {rest} more measure{'s' if rest != 1 else ''} {'are' if rest != 1 else 'is'} covered in the detailed tables." if rest > 0 else ""
        P.append(scale + body + tail)

    # --- per-entity view ---
    if entity and tbl is not None and not tbl.empty:
        e = []
        if "ret" in tbl and tbl["ret"].notna().any():
            r = tbl["ret"].dropna().sort_values(ascending=False)
            up = int((r > 0).sum())
            e.append(f"Looking at each {entity} separately is more telling. Between its first and last "
                     f"record, {up} of {len(r)} rose and {len(r) - up} fell, with a median change of "
                     f"{r.median():+.1f}%. The biggest gainer was {r.index[0]} ({r.iloc[0]:+.0f}%) and the "
                     f"biggest loser was {r.index[-1]} ({r.iloc[-1]:+.0f}%). If the prices aren't adjusted "
                     f"for splits or dividends, a few of the extreme moves may be exaggerated.")
        if "vol" in tbl and tbl["vol"].notna().any():
            v = tbl["vol"].dropna().sort_values(ascending=False)
            e.append(f"{v.index[0]} was the most volatile, swinging about {v.iloc[0]:.1f}% on a typical day, "
                     f"against a median of {v.median():.1f}% across all.")
        if "total_vol" in tbl and tbl["total_vol"].notna().any():
            t = tbl["total_vol"].dropna().sort_values(ascending=False)
            thin = int((tbl["trade_share"] < 0.25).sum())
            line = f"In terms of activity, {t.index[0]} saw the most volume ({_fmt(t.iloc[0])})"
            if thin:
                line += f", while {thin} {entity} entries traded on fewer than a quarter of days"
            e.append(line + ".")
        if e:
            P.append(" ".join(e))

    # --- categories & comparisons ---
    cat_sents = []
    for col in text_cols:
        if col == entity or col in skipped:
            continue
        sv = df[col].dropna().astype(str)
        if sv.empty or sv.str.len().mean() > 40 or sv.nunique() > MAX_GROUP_UNIQUE:
            continue
        vc = sv.value_counts()
        if sv.nunique() == 1:
            cat_sents.append(f"{col} is the same everywhere ({vc.index[0]})")
        else:
            cat_sents.append(f"{col} has {sv.nunique()} categories, led by {vc.index[0]} "
                             f"({vc.iloc[0] / len(sv):.0%})")
    if cat_sents:
        P.append("Among the category columns, " + "; ".join(cat_sents[:6]) + ".")

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
            g_sents.append(f"average {num} is highest for {hi} ({_fmt(m[hi])}) and lowest for "
                           f"{lo} ({_fmt(m[lo])})")
    if g_sents:
        P.append("Comparing groups, " + "; ".join(g_sents) + f". Groups under {MIN_GROUP} rows are ignored.")

    # --- unusual values ---
    out = _outlier_counts(df, numeric_cols)
    if out:
        P.append("A few columns have a small number of genuinely unusual values: "
                 + _join([f"{c} ({k} rows)" for c, (k, _, _) in list(out.items())[:5]]) + ".")
    elif entity:
        P.append("Extreme values are everywhere in this data because the entries trade at very "
                 "different price levels, so I didn't flag them one by one.")

    # --- over time (only for a single series) ---
    if date_cols and not entity and numeric_cols:
        dcol = next(iter(date_cols))
        col = numeric_cols[0]
        mm = _monthly(df, date_cols[dcol], [col])[col]
        if len(mm) >= 3:
            hi, lo = mm.idxmax(), mm.idxmin()
            P.append(f"Over time, the monthly average of {col} is highest in {hi} ({_fmt(mm[hi])}) and "
                     f"lowest in {lo} ({_fmt(mm[lo])}).")

    # --- relationships ---
    if len(numeric_cols) >= 2:
        real, redundant = _split_pairs(_corr_pairs(df, numeric_cols))
        strong = [p for p in real if abs(p[2]) >= CORR_THRESHOLD]
        if strong:
            txt = "; ".join(f"{a} and {b} {'tend to rise together' if r > 0 else 'move in opposite directions'} "
                            f"({_strength(r)}, r = {r:+.2f})" for a, b, r in strong[:3])
            msg = f"Between different measures, the clearest links are: {txt}."
            if entity:
                msg += f" These pool every {entity} together, so they partly reflect which entries are large."
            P.append(msg)
        elif real:
            a, b, r = real[0]
            P.append(f"None of the measures are meaningfully linked to each other; the closest pair, "
                     f"{a} and {b}, is only {_strength(r)} (r = {r:+.2f}).")
        if redundant:
            names = []
            for a, b, _ in redundant[:6]:
                for x in (a, b):
                    if x not in names:
                        names.append(x)
            P.append(f"{_join(names[:6])} are essentially the same thing in different forms, so "
                     f"their strong correlations are built in and aren't a finding.")

    # --- left out ---
    left = [x for x in list(id_cols) + list(skipped) if x != entity]
    if left:
        one = len(left) == 1
        P.append(f"I left out {_join(left)} because {'it looks' if one else 'they look'} like "
                 f"{'an identifier' if one else 'identifiers'} or free text rather than a measurement.")

    return "\n\n".join(P)


# ---------------------------------------------------------------- engine
@tool
def expert_data_analyst_engine(csv_filename: str) -> dict:
    """
    Profiles a local CSV dataset: data quality, numeric statistics, categorical
    breakdowns, group comparisons, outliers, date ranges, monthly trends,
    per-entity performance for panel data, and meaningful correlations. Returns
    compact text sections sized for an LLM plus structural metadata for charts.
    """
    print(f"📦 Data pipeline running on: '{csv_filename}'")

    if not os.path.exists(csv_filename):
        return {"error": f"Targeted data file '{csv_filename}' cannot be located."}

    try:
        df, ph = _load(csv_filename)
        if df.empty:
            return {"error": "The dataset contains no rows."}

        date_cols = _detect_dates(df)
        all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        empty_cols = [c for c in df.columns if df[c].notna().sum() == 0]
        numeric_cols = [c for c in all_numeric
                        if c not in empty_cols
                        and str(c).lower() not in GEO_NAMES
                        and not re.fullmatch(r"prev(ious)?([\s_\-]\w+)?", str(c).strip(), re.I)
                        and not _is_id_like(c, df[c])
                        and df[c].nunique(dropna=True) > 1]
        id_cols = [c for c in all_numeric if c not in numeric_cols and c not in empty_cols
                   and not re.fullmatch(r"prev(ious)?([\s_\-]\w+)?", str(c).strip(), re.I)]
        text_cols = [c for c in df.columns
                     if c not in all_numeric and c not in date_cols and c not in empty_cols]

        cat_text, skipped, group_cats = _categorical_block(df, text_cols)

        entity = _find_entity(df, text_cols, date_cols)
        tbl = None
        if entity:
            price_col = _pick(numeric_cols, r"close|price")
            volume_col = _pick(numeric_cols, r"volume|qty|quantity")
            tbl = _entity_table(df, entity, next(iter(date_cols.values())), price_col, volume_col)
        else:
            price_col = volume_col = None

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
            "quality_text": _quality_block(df, ph),
            "categorical_text": cat_text,
            "group_text": _group_block(df, group_cats, numeric_cols),
            "outlier_text": _outlier_block(df, numeric_cols),
            "date_text": _date_block(date_cols),
            "trend_text": _trend_block(df, date_cols, numeric_cols, bool(entity)),
            "entity_column": entity,
            "entity_text": _entity_block(tbl, entity) if entity else "",
            "narrative_text": _narrative(
                df, numeric_cols, text_cols, date_cols, id_cols, skipped, group_cats,
                ph, empty_cols, entity, tbl, price_col, volume_col
            ),
            "skipped_note": (
                f"Excluded ID-like/coordinate numeric columns: {id_cols}. "
                f"Empty columns: {empty_cols}. "
                f"Skipped identifier/timestamp/free-text/high-cardinality columns: {skipped}."
            ),
        }

    except Exception as data_fault:
        return {"error": f"Pipeline execution failed: {str(data_fault)}"}


# ---------------------------------------------------------------- stateless tools
# Every tool loads the CSV fresh from the path it is given and returns text.
# Nothing is cached, stored in module variables, or written to disk.
def _read(csv_filename: str):
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"File not found: {csv_filename}")
    df, _ = _load(csv_filename)
    return df


def _match_col(df: pd.DataFrame, name: str) -> str:
    for c in df.columns:
        if str(c).strip().lower() == str(name).strip().lower():
            return c
    raise KeyError(f"No column named '{name}'. Columns are: {', '.join(map(str, df.columns))}")


@tool
def list_columns(csv_filename: str) -> str:
    """List every column in a CSV with its type and how many values are missing."""
    try:
        df = _read(csv_filename)
        lines = [f"{len(df)} rows, {df.shape[1]} columns"]
        for c in df.columns:
            lines.append(f"- {c}: {df[c].dtype}, {int(df[c].isna().sum())} missing, "
                         f"{df[c].nunique(dropna=True)} unique")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool
def column_stats(csv_filename: str, column: str) -> str:
    """Describe one column of a CSV: statistics for numbers, most common values for text."""
    try:
        df = _read(csv_filename)
        c = _match_col(df, column)
        s = df[c].dropna()
        if pd.api.types.is_numeric_dtype(s):
            d = s.describe().round(2)
            return f"{c}: " + ", ".join(f"{k}={v}" for k, v in d.items()) + f", skew={s.skew():.2f}"
        vc = s.astype(str).value_counts().head(8)
        return f"{c}: {s.nunique()} unique values. Most common: " + ", ".join(f"{k} ({v})" for k, v in vc.items())
    except Exception as e:
        return f"Error: {e}"


@tool
def group_summary(csv_filename: str, group_column: str, value_column: str, top: int = 10) -> str:
    """Compare a numeric column across the groups of another column (mean, median, row count)."""
    try:
        df = _read(csv_filename)
        g, v = _match_col(df, group_column), _match_col(df, value_column)
        out = (df.groupby(g)[v].agg(["mean", "median", "count"]).round(2)
               .sort_values("mean", ascending=False).head(int(top)))
        return f"{v} by {g} (top {len(out)} by mean):\n{out.to_string()}"
    except Exception as e:
        return f"Error: {e}"


@tool
def profile_dataset(csv_filename: str) -> str:
    """Give a full plain-language profile of any CSV: quality, key numbers, relationships, per-entity results."""
    res = expert_data_analyst_engine.invoke({"csv_filename": csv_filename})
    if "error" in res:
        return f"Error: {res['error']}"
    return res["narrative_text"] + ("\n\n" + res["entity_text"] if res.get("entity_text") else "")


ANALYSIS_TOOLS = [profile_dataset, list_columns, column_stats, group_summary]


# ---------------------------------------------------------------- LLM-written summary
# The engine above computes exact facts. The local model below turns them into prose,
# so the wording is natural and changes with the dataset instead of following a template.
SUMMARY_SYSTEM = (
    "You are a data analyst explaining a dataset to a smart reader who is not a statistician. "
    "Use ONLY the facts you are given and never invent or round-trip numbers that are not there. "
    "Write plain, conversational English in short paragraphs. No section labels, no bold headings, "
    "no stock phrases like 'the dataset contains'. Start with what the data is about, then what "
    "stands out, then what to be careful about (missing values, placeholders, misleading averages, "
    "outliers, redundant columns). Be honest about weak or meaningless findings instead of "
    "dressing them up. Finish with 'In short:' followed by 3 one-sentence takeaways."
)

LENGTHS = {"short": "about 120 words", "medium": "about 250 words", "detailed": "about 400 words"}


def _facts_for_llm(res: dict, max_chars: int = 6000) -> str:
    """Compact, labelled facts block sized for a small local model."""
    parts = [
        ("File", f"{res['file_path']} | {res['total_records']} rows | columns: {', '.join(res['column_layout'])}"),
        ("Data quality", res["quality_text"]),
        ("Dates", res["date_text"]),
        ("Numeric columns (count, mean, std, min, quartiles, max, skew)", res["summary_text"]),
        ("Per-entity results" + (f" (one series per {res['entity_column']})" if res.get("entity_column") else ""),
         res.get("entity_text", "")),
        ("Correlations", res["correlation_text"]),
        ("Outliers", res["outlier_text"]),
        ("Group comparisons", res["group_text"]),
        ("Monthly trend", res.get("trend_text", "")),
        ("Skipped columns", res["skipped_note"]),
        ("Plain-language draft from the profiler (check it, do not copy it)", res["narrative_text"]),
    ]
    text = "\n\n".join(f"## {k}\n{v}" for k, v in parts if v)
    return text[:max_chars]


def write_summary(csv_filename: str, length: str = "medium", model_name: str = "llama3.1:8b") -> str:
    """Profile the CSV, then have the local Ollama model write a readable summary.
    Falls back to the rule-based narrative if the model is unavailable."""
    res = expert_data_analyst_engine.invoke({"csv_filename": csv_filename})
    if "error" in res:
        return res["error"]
    facts = _facts_for_llm(res)
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model_name, temperature=0.4, num_ctx=8192)
        msg = llm.invoke([
            ("system", SUMMARY_SYSTEM),
            ("user", f"Write the summary in {LENGTHS.get(length, LENGTHS['medium'])}.\n\nFACTS:\n{facts}"),
        ])
        return msg.content.strip()
    except Exception as e:
        print(f"⚠️ Local model unavailable ({e}). Using the rule-based summary instead.")
        return res["narrative_text"]


def rewrite_summary(summary: str, instruction: str, model_name: str = "llama3.1:8b") -> str:
    """Summarize or reshape an existing summary, e.g. 'in 3 sentences', 'for a manager',
    'explain it like I'm new to stocks'."""
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=model_name, temperature=0.3, num_ctx=8192)
    msg = llm.invoke([
        ("system", "Rewrite the text as instructed. Keep every number exactly as written and add "
                   "no new facts. Plain, natural English."),
        ("user", f"Instruction: {instruction}\n\nText:\n{summary}"),
    ])
    return msg.content.strip()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "NSE_data_all_stocks_2025.csv"
    s = write_summary(path, length="medium")
    print("\n" + s)
    while True:
        cmd = input("\nRewrite it (e.g. 'in 3 sentences', 'for a manager') or press Enter to quit: ").strip()
        if not cmd:
            break
        s = rewrite_summary(s, cmd)
        print("\n" + s)
