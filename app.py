import hashlib
import os
import re
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

try:
    from data_analyst import expert_data_analyst_engine
except ImportError:  # patched profiler saved as data_agent.py
    from data_agent import expert_data_analyst_engine

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DEFAULT_MODEL = "llama-3.3-70b-versatile"   # override with GROQ_MODEL in secrets/env
MAX_PAYLOAD_CHARS = 12000
MAX_EXTRA_CHARS = 2500
MAX_HISTORY_MSGS = 8        # recent chat turns sent to the model
MAX_PLOT_ROWS = 5000        # plot a sample: keeps charts fast
PLACEHOLDERS = ["-", "--", "\\N", "N/A", "NA", "n/a", "null", "NULL", "None", ""]

SUGGESTIONS = [
    "Summarize this dataset in plain English",
    "What looks odd or unreliable in this data?",
    "What are the most important patterns?",
    "What should I analyze next?",
]

CHART_TYPES = [
    "Line Trend", "Area Trend", "Bar (Feature Means)", "Grouped Bar (by Category)",
    "Box (Outliers)", "Violin (Distribution)", "Scatter (Relationship)", "Histogram",
    "Donut", "Sunburst", "Correlation Heatmap", "Missing Values",
]
THEMES = ["plotly_white", "simple_white", "seaborn", "ggplot2", "plotly_dark"]

warnings.filterwarnings("ignore", message="Could not infer format")

st.set_page_config(page_title="K1 Data Analyst", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------------
# SESSION STATE  (the dataset lives only in memory; nothing is written to disk)
# ----------------------------------------------------------------------------
for k, v in {
    "file_sig": None, "file_name": None, "df": None, "backend": None, "payload": None,
    "approved": False, "messages": [], "pending": None,
}.items():
    st.session_state.setdefault(k, v)


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY")


def get_model_name():
    try:
        return st.secrets.get("GROQ_MODEL", DEFAULT_MODEL)
    except Exception:
        return os.environ.get("GROQ_MODEL", DEFAULT_MODEL)


def split_columns(df):
    nums = df.select_dtypes(include="number").columns.tolist()
    return nums, [c for c in df.columns if c not in nums]


def load_clean(uploaded):
    """Read a CSV, treat '-' / '\\N' / 'N/A' as missing, turn numeric-looking text into numbers."""
    df = pd.read_csv(uploaded, na_values=PLACEHOLDERS, encoding="utf-8-sig")
    for c in df.select_dtypes(exclude=[np.number]).columns:
        if re.search(r"date|time", str(c), re.I):
            continue
        present = df[c].notna()
        if not present.any():
            continue
        probe = df[c].dropna().head(2000).astype(str).str.replace(r"[,%]", "", regex=True)
        if pd.to_numeric(probe, errors="coerce").notna().mean() < 0.8:
            continue   # clearly text: skip the expensive full conversion
        num = pd.to_numeric(df[c].astype(str).str.replace(r"[,%]", "", regex=True), errors="coerce")
        if num[present].notna().mean() >= 0.8:
            df[c] = num
    return df


def run_engine(df):
    """The profiler wants a file path, so write a temp copy, analyze, and delete it immediately."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        df.to_csv(path, index=False)
        return expert_data_analyst_engine.invoke({"csv_filename": path})
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def build_payload(b):
    sections = [
        ("DATASET METADATA",
         f"Total Records: {b['total_records']}\nColumns: {b['column_layout']}\n{b.get('skipped_note', '')}"),
        ("DATA QUALITY", b.get("quality_text")),
        ("NUMERIC SUMMARY (count/mean/std/min/quartiles/max/skew)", b["summary_text"]),
        ("PER-ENTITY RESULTS", b.get("entity_text")),
        ("CATEGORICAL BREAKDOWN", b.get("categorical_text")),
        ("GROUP COMPARISONS", b.get("group_text")),
        ("OUTLIERS", b.get("outlier_text")),
        ("DATE RANGES", b.get("date_text")),
        ("TIME TREND", b.get("trend_text")),
        ("CORRELATIONS", b["correlation_text"]),
    ]
    text = "\n\n".join(f"======= {t} =======\n{body}" for t, body in sections if body)
    return text[:MAX_PAYLOAD_CHARS] + ("\n\n[...Truncated...]" if len(text) > MAX_PAYLOAD_CHARS else "")


def clear_dataset():
    st.session_state.update(file_sig=None, file_name=None, df=None, backend=None, payload=None,
                            approved=False, messages=[], pending=None)


def ingest(uploaded):
    """Load once per distinct upload (by content). New file = fresh chat, nothing carried over."""
    raw = uploaded.getvalue()
    sig = hashlib.md5(raw).hexdigest()
    if st.session_state.file_sig == sig:
        return
    uploaded.seek(0)
    t0 = time.time()
    df = load_clean(uploaded)
    backend = run_engine(df)
    st.session_state.profile_secs = time.time() - t0
    messages, payload = [], None
    if "error" in backend:
        messages.append({"role": "assistant", "local": True,
                         "content": f"⚠️ I couldn't profile this file: {backend['error']}"})
        backend = None
    else:
        payload = build_payload(backend)
        messages.append({"role": "assistant", "local": True, "content": (
            f"Here's a first look at **{uploaded.name}**. I wrote this on your machine, "
            f"so nothing has been sent anywhere yet.\n\n{backend['narrative_text']}\n\n"
            "Ask me anything about it below.")})
    st.session_state.update(file_sig=sig, file_name=uploaded.name, df=df, backend=backend,
                            payload=payload, approved=False, messages=messages, pending=None)


def question_facts(df, q):
    """Cheap local lookups for whatever the question mentions (columns, stock/city/etc. values).
    Only small aggregates are produced, never raw rows."""
    ql = q.lower()
    nums, cats = split_columns(df)
    out = []

    mentioned_num = [c for c in nums if len(str(c)) >= 3 and re.search(rf"(?<!\w){re.escape(str(c).lower())}(?!\w)", ql)]
    mentioned_cat = [c for c in cats if len(str(c)) >= 3 and re.search(rf"(?<!\w){re.escape(str(c).lower())}(?!\w)", ql)]

    for c in mentioned_num:
        s = df[c].dropna()
        out.append(f"[{c}] mean={s.mean():,.2f} median={s.median():,.2f} min={s.min():,.2f} "
                   f"max={s.max():,.2f} missing={int(df[c].isna().sum())}")
    for c in mentioned_cat:
        vc = df[c].dropna().astype(str).value_counts().head(8)
        out.append(f"[{c}] {df[c].nunique()} unique; most common: " + ", ".join(f"{k} ({v})" for k, v in vc.items()))
        if mentioned_num and df[c].nunique() <= 500:
            g = df.groupby(c)[mentioned_num[0]].agg(["mean", "sum", "count"]).round(2)
            out.append(f"[{mentioned_num[0]} by {c}] top 10 by mean:\n" + g.sort_values("mean", ascending=False).head(10).to_string())

    date_col = next((c for c in df.columns if re.search(r"date|time", str(c), re.I)), None)
    order = None
    if date_col is not None:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        if parsed.notna().mean() > 0.8:
            order = parsed
    for c in cats:
        s = df[c].dropna().astype(str)
        if s.empty or s.nunique() > 1000 or s.str.len().mean() > 40:
            continue
        hits = [v for v in s.unique() if len(v) >= 2 and re.search(rf"(?<!\w){re.escape(v.lower())}(?!\w)", ql)]
        for v in hits[:3]:
            sub = df[df[c].astype(str) == v]
            lines = [f"[{c} = {v}] {len(sub)} rows"]
            if order is not None:
                sub = sub.assign(_o=order.loc[sub.index]).sort_values("_o")
                lines.append(f"  from {sub['_o'].min():%Y-%m-%d} to {sub['_o'].max():%Y-%m-%d}")
            for n in nums[:6]:
                col = sub[n].dropna()
                if col.empty:
                    continue
                piece = f"  {n}: min={col.min():,.2f} max={col.max():,.2f} mean={col.mean():,.2f}"
                if order is not None:
                    piece = f"  {n}: first={col.iloc[0]:,.2f} last={col.iloc[-1]:,.2f} min={col.min():,.2f} max={col.max():,.2f} mean={col.mean():,.2f}"
                lines.append(piece)
            out.append("\n".join(lines))
    text = "\n\n".join(out)
    return text[:MAX_EXTRA_CHARS]


SYSTEM = """You are a sharp, friendly data analyst chatting with someone about their dataset.
- Answer the question directly first, then add only the context that helps.
- Use the numbers in DATASET FACTS and EXTRA FACTS. Never invent numbers. If the facts cannot answer the question, say what is missing and which column or breakdown would settle it.
- Write plain, conversational English in short paragraphs. Use a small table or bullets only when comparing several items. No stock headings, no filler.
- Be honest about weak, redundant, or misleading results (pooled averages across very different items, near-duplicate columns, many missing values).
- Placeholders like '-' mean missing. Do not assume units: use USER CONTEXT, otherwise say units are unconfirmed.
- Do not forecast or predict beyond the data.
- You can see the earlier turns of this chat; use them for follow-ups like "what about the second one?"

USER CONTEXT: {context}

DATASET FACTS:
{payload}

EXTRA FACTS FOR THIS QUESTION:
{extra}"""


def stream_answer(question, history, payload, extra, context):
    key = get_api_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY missing from st.secrets or environment.")
    llm = ChatGroq(model=get_model_name(), groq_api_key=key, temperature=0.2,
                   timeout=30, max_retries=1)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM), MessagesPlaceholder("history"), ("human", "{question}"),
    ])
    return (prompt | llm | StrOutputParser()).stream({
        "question": question, "history": history, "payload": payload,
        "extra": extra or "none", "context": context.strip() or "none provided",
    })


def history_messages():
    """Recent real conversation turns (skips locally written messages)."""
    msgs = [m for m in st.session_state.messages if not m.get("local")][-MAX_HISTORY_MSGS:]
    return [HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"]) for m in msgs]


def conversation_markdown():
    out = [f"# Chat about {st.session_state.file_name}\n"]
    for m in st.session_state.messages:
        out.append(f"**{'You' if m['role'] == 'user' else 'Analyst'}:**\n\n{m['content']}\n")
    return "\n".join(out)


def prep_x(df, x):
    out = df.copy()
    if not pd.api.types.is_numeric_dtype(out[x]) and not pd.api.types.is_datetime64_any_dtype(out[x]):
        parsed = pd.to_datetime(out[x], errors="coerce")
        if parsed.notna().mean() > 0.8:
            out[x] = parsed
    return out.sort_values(x)


def guess_time_col(df):
    for i, c in enumerate(df.columns):
        if re.search(r"date|time", str(c), re.I):
            return i
    return 0


# ----------------------------------------------------------------------------
# CHART FACTORY
# ----------------------------------------------------------------------------
def build_chart(kind, df, x, ys, cat, color_by, theme):
    nums, cats = split_columns(df)
    plot_df = df.sample(MAX_PLOT_ROWS, random_state=1) if len(df) > MAX_PLOT_ROWS else df
    ys = [y for y in ys if y in nums] or nums[:2]
    color = None if color_by == "None" else color_by
    fig = None
    try:
        if kind in ("Line Trend", "Area Trend"):
            if not ys:
                return None, "No numeric columns available."
            d = prep_x(plot_df, x)
            fn = px.line if kind == "Line Trend" else px.area
            kwargs = {"markers": True} if kind == "Line Trend" else {}
            fig = fn(d, x=x, y=ys, title=f"{kind}: {', '.join(ys)} over {x}", **kwargs)
        elif kind == "Bar (Feature Means)":
            m = plot_df[ys or nums].mean().reset_index()
            m.columns = ["Feature", "Mean"]
            fig = px.bar(m, x="Feature", y="Mean", color="Feature", title="Feature Means")
        elif kind == "Grouped Bar (by Category)":
            if not cats or not ys:
                return None, "Needs at least one categorical and one numeric column."
            top = plot_df[cat].value_counts().head(15).index
            g = plot_df[plot_df[cat].isin(top)].groupby(cat)[ys].mean().reset_index()
            g = g.melt(id_vars=cat, var_name="Feature", value_name="Mean")
            fig = px.bar(g, x=cat, y="Mean", color="Feature", barmode="group",
                         title=f"Mean of features by {cat} (top 15)")
        elif kind in ("Box (Outliers)", "Violin (Distribution)"):
            if not ys:
                return None, "No numeric columns available."
            long = plot_df[ys].melt(var_name="Feature", value_name="Value")
            fn = px.box if kind.startswith("Box") else px.violin
            kwargs = {} if kind.startswith("Box") else {"box": True}
            fig = fn(long, x="Feature", y="Value", color="Feature", title=kind, **kwargs)
        elif kind == "Scatter (Relationship)":
            if len(nums) < 2:
                return None, "Needs at least two numeric columns."
            sx = x if x in nums else nums[0]
            sy = next((y for y in ys if y != sx), next(n for n in nums if n != sx))
            fig = px.scatter(plot_df, x=sx, y=sy, color=color, title=f"{sy} vs {sx}", opacity=0.7)
        elif kind == "Histogram":
            if not ys:
                return None, "No numeric columns available."
            fig = px.histogram(plot_df, x=ys[0], color=color, marginal="box", nbins=40,
                               title=f"Distribution of {ys[0]}")
        elif kind == "Donut":
            if not cats:
                return None, "No categorical column available."
            if ys:
                agg = plot_df.groupby(cat)[ys[0]].sum().nlargest(10).reset_index()
                fig = px.pie(agg, names=cat, values=ys[0], hole=0.6, title=f"{ys[0]} by {cat} (top 10)")
            else:
                agg = plot_df[cat].value_counts().head(10).reset_index()
                agg.columns = [cat, "Count"]
                fig = px.pie(agg, names=cat, values="Count", hole=0.6, title=f"{cat} composition (top 10)")
            fig.update_traces(textposition="inside", textinfo="percent+label")
        elif kind == "Sunburst":
            usable = [c for c in cats if 1 < plot_df[c].nunique() <= 30][:2]
            if not usable or not ys:
                return None, "Needs low-cardinality categorical columns and a numeric column."
            s = plot_df[usable + [ys[0]]].copy()
            s[usable] = s[usable].fillna("N/A").astype(str)
            s = s[s[ys[0]].fillna(0) > 0]
            fig = px.sunburst(s, path=usable, values=ys[0],
                              title=f"Hierarchy: {' > '.join(usable)} ({ys[0]})")
        elif kind == "Correlation Heatmap":
            if len(nums) < 2:
                return None, "Needs at least two numeric columns."
            fig = px.imshow(plot_df[nums].corr(), text_auto=".2f", zmin=-1, zmax=1,
                            color_continuous_scale="RdBu_r", aspect="auto", title="Correlation Matrix")
        elif kind == "Missing Values":
            miss = df.isna().sum().reset_index()
            miss.columns = ["Column", "Missing"]
            miss["Percent"] = (miss["Missing"] / len(df) * 100).round(2)
            fig = px.bar(miss.sort_values("Missing", ascending=False), x="Column", y="Percent",
                         hover_data=["Missing"], title="Missing Values (% of rows)")
    except Exception as e:
        return None, f"Chart failed: {e}"
    if fig is not None:
        fig.update_layout(template=theme, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=60, l=10, r=10, b=10))
    return fig, None


def render_chart(kind, df, x, ys, cat, color_by, theme, key):
    fig, msg = build_chart(kind, df, x, ys, cat, color_by, theme)
    if msg:
        st.warning(msg)
    elif fig is not None:
        st.plotly_chart(fig, key=key)


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("📁 Workspace")
uploaded_file = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
human_approval = st.sidebar.checkbox(
    "Ask before sending data to Groq", value=True,
    help="Only summary statistics are ever sent, never your raw rows. Untick to skip the approval step.",
)
st.sidebar.text_input(
    "Context for the analyst (optional)", key="context",
    placeholder="e.g. NSE Kenya daily prices in KES",
    help="Units and background the data itself can't tell the AI.",
)

if uploaded_file is not None:
    try:
        with st.spinner("Profiling your data locally..."):
            ingest(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"🚨 Could not load file: {e}")
elif st.session_state.df is not None:
    clear_dataset()   # file removed -> data, chat and charts all go

df = st.session_state.df
has_key = bool(get_api_key())
if not has_key:
    st.sidebar.warning("No GROQ_API_KEY found. The local summary works, but chat needs the key.")

if df is not None:
    st.sidebar.caption(f"⏱ Profiled locally in {st.session_state.get('profile_secs', 0):.1f}s")
    st.sidebar.markdown("---")
    if st.sidebar.button("🧹 Clear chat"):
        st.session_state.messages = st.session_state.messages[:1]
        st.session_state.pending = None
        st.rerun()
    if len(st.session_state.messages) > 1:
        st.sidebar.download_button("⬇️ Download conversation", conversation_markdown(),
                                   file_name="analyst_chat.md", mime="text/markdown")

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
st.title("K1 Data Analyst")

if df is None:
    st.info("💡 Upload a CSV in the sidebar to start. Your data stays in memory only, and when you remove the file, everything is cleared.")
    st.stop()

st.caption(f"📦 {st.session_state.file_name} · {len(df):,} rows × {df.shape[1]} columns")
view = st.radio("View", ["💬 Chat", "📈 Charts", "📊 Data"], horizontal=True, label_visibility="collapsed")

# ------------------------------------------------------------------ CHAT
if view == "💬 Chat":
    gate_open = (not human_approval) or st.session_state.approved
    can_chat = has_key and gate_open and st.session_state.payload is not None

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("timing"):
                st.caption(m["timing"])
            if m.get("extra"):
                with st.expander("Extra facts sent with this question"):
                    st.code(m["extra"])

    if human_approval and not st.session_state.approved and st.session_state.payload:
        with st.expander("🔒 Review what will be sent to Groq (summary statistics only)", expanded=True):
            st.code(st.session_state.payload)
            if st.button("✅ Approve and start chatting"):
                st.session_state.approved = True
                st.rerun()

    if len(st.session_state.messages) <= 1 and can_chat:
        st.caption("Try one of these:")
        cols = st.columns(len(SUGGESTIONS))
        for col, s in zip(cols, SUGGESTIONS):
            if col.button(s, key=f"sug_{s}"):
                st.session_state.pending = s
                st.rerun()

    typed = st.chat_input("Ask anything about your data…", disabled=not can_chat)
    question = typed or st.session_state.pop("pending", None)

    if question and can_chat:
        extra = question_facts(df, question)
        history = history_messages()          # taken before adding this turn
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                t0, first = time.time(), []

                def timed(gen):
                    for chunk in gen:
                        if not first:
                            first.append(time.time() - t0)
                        yield chunk

                with st.spinner("Thinking…"):
                    streamed = st.write_stream(timed(stream_answer(
                        question, history, st.session_state.payload, extra, st.session_state.get("context", ""))))
                answer = re.sub(r"<think>.*?</think>", "", str(streamed), flags=re.DOTALL).strip()
                timing = f"⏱ first word {first[0] if first else 0:.1f}s · total {time.time() - t0:.1f}s"
                st.caption(timing)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "extra": extra, "timing": timing})
                if extra:
                    with st.expander("Extra facts sent with this question"):
                        st.code(extra)
            except Exception as e:
                msg = f"⚠️ I couldn't get an answer: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg, "local": True})

# ------------------------------------------------------------------ CHARTS
elif view == "📈 Charts":
    nums, cats = split_columns(df)
    fk = (st.session_state.file_sig or "none")[:8]
    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("X-Axis / Timeline", df.columns.tolist(), index=guess_time_col(df), key=f"x_{fk}")
        theme = st.selectbox("Theme", THEMES, key=f"theme_{fk}")
    with c2:
        ys = st.multiselect("Numeric features (Y)", nums, default=nums[:2], key=f"ys_{fk}")
        cat = st.selectbox("Category dimension", cats or df.columns.tolist(), key=f"cat_{fk}")
    with c3:
        color_by = st.selectbox("Color by (scatter/histogram)", ["None"] + cats, key=f"color_{fk}")

    section = st.radio("Section", ["🎛️ Dual-Panel Studio", "🖼️ Auto Gallery"], horizontal=True,
                       key=f"sec_{fk}", label_visibility="collapsed")
    if section.startswith("🎛️"):
        left, right = st.columns(2)
        with left:
            kind_l = st.selectbox("Left panel", CHART_TYPES, index=0, key=f"kl_{fk}")
            render_chart(kind_l, df, x_axis, ys, cat, color_by, theme, f"chart_left_{fk}")
        with right:
            kind_r = st.selectbox("Right panel", CHART_TYPES, index=4, key=f"kr_{fk}")
            render_chart(kind_r, df, x_axis, ys, cat, color_by, theme, f"chart_right_{fk}")
    else:
        gallery = ["Correlation Heatmap", "Histogram", "Box (Outliers)",
                   "Scatter (Relationship)", "Donut", "Missing Values"]
        for i in range(0, len(gallery), 2):
            ga, gb = st.columns(2)
            for col, kind in zip((ga, gb), gallery[i:i + 2]):
                with col:
                    render_chart(kind, df, x_axis, ys, cat, color_by, theme, f"gal_{kind}_{fk}")

# ------------------------------------------------------------------ DATA
else:
    st.dataframe(df.head(1000), height=520)
    st.caption("Showing the first 1,000 rows.")
    if st.session_state.backend:
        with st.expander("Local profile (nothing here is sent anywhere)"):
            st.markdown(st.session_state.backend["narrative_text"])
        with st.expander("Raw payload used as chat context"):
            st.code(st.session_state.payload)
