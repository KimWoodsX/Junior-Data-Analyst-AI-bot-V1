import hashlib
import os
import re
import time
import tempfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from data_analyst import expert_data_analyst_engine

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# Production model on Groq. Override with GROQ_MODEL in secrets/env.
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_PAYLOAD_CHARS = 12000
MAX_PLOT_ROWS = 20000  # sample above this for speed

CHART_TYPES = [
    "Line Trend",
    "Area Trend",
    "Bar (Feature Means)",
    "Grouped Bar (by Category)",
    "Box (Outliers)",
    "Violin (Distribution)",
    "Scatter (Relationship)",
    "Histogram",
    "Donut",
    "Sunburst",
    "Correlation Heatmap",
    "Missing Values",
]
THEMES = ["plotly_white", "simple_white", "seaborn", "ggplot2", "plotly_dark"]

st.set_page_config(
    page_title="Enterprise Cloud Data Analyst Hub",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
defaults = {
    "session_token": str(int(time.time())),
    "file_sig": None,
    "df": None,
    "csv_path": None,
    "backend": None,
    "payload": None,
    "report": None,
}
for k, v in defaults.items():
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
    cats = [c for c in df.columns if c not in nums]
    return nums, cats


def cleanup_old_file():
    path = st.session_state.csv_path
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def clear_dataset():
    """Wipe the loaded dataset and everything derived from it."""
    cleanup_old_file()
    st.session_state.update(
        file_sig=None, df=None, csv_path=None, backend=None, payload=None, report=None
    )


def ingest(uploaded):
    """Load the CSV once per distinct upload (by content); reset all derived state."""
    raw = uploaded.getvalue()
    sig = hashlib.md5(raw).hexdigest()
    if st.session_state.file_sig == sig:
        return
    uploaded.seek(0)
    df = pd.read_csv(uploaded)
    cleanup_old_file()
    path = os.path.join(
        tempfile.gettempdir(), f"secure_target_{st.session_state.session_token}.csv"
    )
    df.to_csv(path, index=False)
    st.session_state.update(
        file_sig=sig, df=df, csv_path=path, backend=None, payload=None, report=None
    )


def prep_x(df, x):
    """Parse date-like object columns so trend charts sort properly."""
    out = df.copy()
    if out[x].dtype == object:
        parsed = pd.to_datetime(out[x], errors="coerce")
        if parsed.notna().mean() > 0.8:
            out[x] = parsed
    return out.sort_values(x)


def build_payload(b):
    sections = [
        ("DATASET METADATA",
         f"Total Records: {b['total_records']}\nColumns: {b['column_layout']}\n{b.get('skipped_note', '')}"),
        ("DATA QUALITY", b.get("quality_text")),
        ("NUMERIC SUMMARY (count/mean/std/min/quartiles/max/skew)", b["summary_text"]),
        ("CATEGORICAL BREAKDOWN", b.get("categorical_text")),
        ("GROUP COMPARISONS", b.get("group_text")),
        ("OUTLIERS", b.get("outlier_text")),
        ("DATE RANGES", b.get("date_text")),
        ("TIME TREND", b.get("trend_text")),
        ("CORRELATIONS", b["correlation_text"]),
    ]
    text = "\n\n".join(f"======= {t} =======\n{body}" for t, body in sections if body)
    if len(text) > MAX_PAYLOAD_CHARS:
        text = text[:MAX_PAYLOAD_CHARS] + "\n\n[...Truncated...]"
    return text


def run_llm(directive, payload, context=""):
    key = get_api_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY missing from st.secrets or environment.")
    llm = ChatGroq(model=get_model_name(), groq_api_key=key, temperature=0.1)
    prompt = ChatPromptTemplate.from_template(
        """SYSTEM: You are a Lead Data Scientist. Compile a concise engineering report using these headers and bullets:
1. EXECUTIVE DATA SUMMARY
2. CORE DIFFERENTIAL ANALYSIS (quantified values)
3. STATISTICAL OBSERVATIONS (non-basic variance/skewness)
4. CORRELATIONS & RELATIONSHIPS
5. STRATEGIC INSIGHTS
Only state what the data supports. Do not invent numbers. Do not assume units: use the units in USER CONTEXT, and if none are given say the units are unconfirmed. Treat near-duplicate measures as redundant, not as findings. Do not forecast or predict beyond the data.

USER DIRECTIVE: {directive}

USER CONTEXT: {context}

DATASET PAYLOAD:
{payload}"""
    )
    out = (prompt | llm | StrOutputParser()).invoke(
        {"directive": directive, "payload": payload,
         "context": context.strip() or "none provided"}
    )
    # Qwen3 emits <think>...</think>; strip it.
    return re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()


# ----------------------------------------------------------------------------
# CHART FACTORY
# ----------------------------------------------------------------------------
def build_chart(kind, df, x, ys, cat, color_by, theme):
    """Return (figure, message). Never raises into the UI."""
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
            feats = ys or nums
            m = plot_df[feats].mean().reset_index()
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
            try:
                fig = px.scatter(plot_df, x=sx, y=sy, color=color, trendline="ols",
                                 title=f"{sy} vs {sx}", opacity=0.7)
            except Exception:  # statsmodels missing
                fig = px.scatter(plot_df, x=sx, y=sy, color=color,
                                 title=f"{sy} vs {sx}", opacity=0.7)

        elif kind == "Histogram":
            if not ys:
                return None, "No numeric columns available."
            fig = px.histogram(plot_df, x=ys[0], color=color, marginal="box",
                               nbins=40, title=f"Distribution of {ys[0]}")

        elif kind == "Donut":
            if not cats:
                return None, "No categorical column available."
            if ys:
                agg = plot_df.groupby(cat)[ys[0]].sum().nlargest(10).reset_index()
                fig = px.pie(agg, names=cat, values=ys[0], hole=0.6,
                             title=f"{ys[0]} by {cat} (top 10)")
            else:
                agg = plot_df[cat].value_counts().head(10).reset_index()
                agg.columns = [cat, "Count"]
                fig = px.pie(agg, names=cat, values="Count", hole=0.6,
                             title=f"{cat} composition (top 10)")
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
            corr = plot_df[nums].corr()
            fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                            color_continuous_scale="RdBu_r", aspect="auto",
                            title="Correlation Matrix")

        elif kind == "Missing Values":
            miss = df.isna().sum().reset_index()
            miss.columns = ["Column", "Missing"]
            miss["Percent"] = (miss["Missing"] / len(df) * 100).round(2)
            fig = px.bar(miss.sort_values("Missing", ascending=False), x="Column",
                         y="Percent", hover_data=["Missing"],
                         title="Missing Values (% of rows)")

    except Exception as e:
        return None, f"Chart failed: {e}"

    if fig is not None:
        fig.update_layout(template=theme, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=60, l=10, r=10, b=10))
    return fig, None


def render(kind, df, x, ys, cat, color_by, theme, key):
    fig, msg = build_chart(kind, df, x, ys, cat, color_by, theme)
    if msg:
        st.warning(msg)
    elif fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=key)


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("📁 Control Panel Workspace")
st.sidebar.write("Ingest structured data (CSV only).")
uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"], label_visibility="collapsed")
human_approval = st.sidebar.checkbox(
    "Require Manual Human Approval", value=True,
    help="Review the exact payload before it is sent to the cloud LLM.",
)
st.sidebar.markdown("---")
st.sidebar.write(f"Inference: **Groq** · model `{get_model_name()}`")

if uploaded_file is not None:
    try:
        ingest(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"🚨 Ingestion fault: {e}")
elif st.session_state.df is not None:
    clear_dataset()  # file removed -> old data and graphs must go too

df = st.session_state.df

# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
st.title("✨ Enterprise Data Analyst & File Co-Pilot")
st.caption("Analytical pipelines orchestrated via LangChain Expression Language (LCEL)")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("📊 Active Dataset Preview")
    if df is not None:
        st.success(f"📦 '{uploaded_file.name if uploaded_file else 'dataset'}' — {len(df):,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(12), use_container_width=True)
    else:
        st.info("💡 Upload a .csv to begin.")

with col2:
    st.header("⚙️ AI Agent Control Center")
    directive = st.text_area(
        "Directives",
        placeholder="e.g., Run a deep statistical variance audit on the active rows...",
        height=150,
        label_visibility="collapsed",
    )
    context_note = st.text_input(
        "Context for the analyst (optional)",
        placeholder="e.g., daily Nairobi weather, temperatures in °F, source: Visual Crossing",
        help="Units and background the data itself can't tell the AI. Sent with the payload.",
    )

    if st.button("Execute Pipeline Analysis 🚀", use_container_width=True):
        if df is None:
            st.error("❌ No dataset loaded.")
        elif not directive.strip():
            st.warning("⚠️ Provide an instruction first.")
        else:
            with st.spinner("Running local analytics engine..."):
                try:
                    backend = expert_data_analyst_engine.invoke(
                        {"csv_filename": st.session_state.csv_path}
                    )
                    if "error" in backend:
                        st.error(backend["error"])
                    else:
                        st.session_state.backend = backend
                        st.session_state.payload = build_payload(backend)
                        st.session_state.report = None
                        st.session_state.directive = directive
                        st.session_state.context = context_note
                        if not human_approval:
                            st.session_state.report = run_llm(
                                directive, st.session_state.payload, context_note
                            )
                except Exception as e:
                    st.error(f"🚨 Pipeline exception: {e}")

    # Approval gate: payload persists in session state, so this survives reruns.
    if human_approval and st.session_state.payload and not st.session_state.report:
        with st.expander("🔒 Review payload before it leaves your machine", expanded=True):
            tab_plain, tab_raw = st.tabs(["📖 Plain-English Summary", "🧾 Raw Payload"])
            with tab_plain:
                st.caption("Written locally from your data. This view is not sent to Groq.")
                st.markdown(
                    (st.session_state.backend or {}).get("narrative_text")
                    or "No summary available for this dataset."
                )
            with tab_raw:
                st.caption("Exactly what will be sent to Groq if you approve.")
                st.code(st.session_state.payload)
        if st.button("✅ Approve & send to Groq", use_container_width=True):
            with st.spinner("Synthesizing report..."):
                try:
                    st.session_state.report = run_llm(
                        st.session_state.get("directive", directive),
                        st.session_state.payload,
                        st.session_state.get("context", ""),
                    )
                except Exception as e:
                    st.error(f"🚨 Cloud exception: {e}")

    if st.session_state.report:
        st.success("🎯 Analysis cycle complete")
        with st.container(height=320):
            st.markdown(st.session_state.report)
        st.download_button("⬇️ Download report", st.session_state.report,
                           file_name="analyst_report.md", mime="text/markdown")

# ----------------------------------------------------------------------------
# VISUAL STUDIO
# ----------------------------------------------------------------------------
if df is not None:
    st.markdown("---")
    st.header("📈 Interactive Visual Studio")
    nums, cats = split_columns(df)

    fk = (st.session_state.file_sig or "none")[:8]  # widget keys change per dataset
    c1, c2, c3 = st.columns(3)
    with c1:
        x_axis = st.selectbox("X-Axis / Timeline", df.columns.tolist(), index=0, key=f"x_{fk}")
        theme = st.selectbox("Theme", THEMES, key=f"theme_{fk}")
    with c2:
        ys = st.multiselect("Numeric features (Y)", nums, default=nums[:2], key=f"ys_{fk}")
        cat = st.selectbox("Category dimension", cats or df.columns.tolist(), key=f"cat_{fk}")
    with c3:
        color_by = st.selectbox("Color by (scatter/histogram)", ["None"] + cats, key=f"color_{fk}")

    tab_studio, tab_gallery = st.tabs(["🎛️ Dual-Panel Studio", "🖼️ Auto Gallery"])

    with tab_studio:
        left, right = st.columns(2)
        with left:
            kind_l = st.selectbox("Left panel", CHART_TYPES, index=0, key=f"kl_{fk}")
            render(kind_l, df, x_axis, ys, cat, color_by, theme, f"chart_left_{fk}")
        with right:
            kind_r = st.selectbox("Right panel", CHART_TYPES, index=4, key=f"kr_{fk}")
            render(kind_r, df, x_axis, ys, cat, color_by, theme, f"chart_right_{fk}")

    with tab_gallery:
        st.caption("Six core views of your data, generated with the current selections.")
        gallery = ["Correlation Heatmap", "Histogram", "Box (Outliers)",
                   "Scatter (Relationship)", "Donut", "Missing Values"]
        for i in range(0, len(gallery), 2):
            ga, gb = st.columns(2)
            for col, kind in zip((ga, gb), gallery[i:i + 2]):
                with col:
                    render(kind, df, x_axis, ys, cat, color_by, theme, f"gal_{kind}_{fk}")
