import streamlit as st
import os
import time
import pandas as pd
import plotly.express as px
from data_analyst import expert_data_analyst_engine
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. ENTERPRISE GLOBAL PAGE MATRIX ---
st.set_page_config(
    page_title="Enterprise Cloud Data Analyst Hub", 
    layout="wide",
    initial_sidebar_state="expanded"
)

if "session_token" not in st.session_state:
    st.session_state.session_token = str(int(time.time()))
if "active_matrix_data" not in st.session_state:
    st.session_state.active_matrix_data = None

# --- 2. SIDEBAR CONTROL PANELS ---
st.sidebar.title("📁 Control Panel Workspace")
st.sidebar.write("Ingest structured data arrays securely (CSV formatting only).")

uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"], label_visibility="collapsed")
st.sidebar.text_input("🔍 Search active folder workspace", placeholder="Search filenames...")
human_approval = st.sidebar.checkbox("Require Manual Human Approval", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Infrastructure Status: **Cloud API Inversion Layer Active (Groq Node)**")

# --- 3. MAIN BLOCK AREA ---
st.title("✨ Enterprise Local Data Analyst & File Co-Pilot")
st.write("Orchestrating state-of-the-art analytical data pipelines via LangChain Expression Language (LCEL)")
st.markdown("---")

col1, col2 = st.columns([0.5, 0.5])
runtime_csv_target = None

# --- COLUMN 1: ACTIVE DATASET ISOLATION & PREVIEW ---
with col1:
    st.header("📊 Active Dataset Preview")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"📦 Source Data Ingested: '{uploaded_file.name}' — Discovered {len(df)} records.")
            st.dataframe(df.head(12), use_container_width=True)
            
            runtime_csv_target = f"secure_target_{st.session_state.session_token}.csv"
            with open(runtime_csv_target, "wb") as secure_stream:
                secure_stream.write(uploaded_file.getbuffer())
                
        except Exception as ingestion_error:
            st.error(f"🚨 Structural Ingestion Fault: {ingestion_error}")
    else:
        st.info("💡 Storage Core Standby: Please upload a .csv data array to initialize validation.")

# --- COLUMN 2: ANALYTICAL AGENT CONTROL SYSTEM ---
with col2:
    st.header("⚙️ AI Agent Control Center")
    st.write("Transmit high-priority optimization directives to your analytics execution layer:")
    
    user_directive = st.text_area(
        "Directives",
        placeholder="e.g., Run a deep statistical variance audit on the active rows...",
        height=150,
        label_visibility="collapsed"
    )
    
    execute_pipeline = st.button("Execute Pipeline Analysis 🚀", use_container_width=True)
    
    if execute_pipeline:
        if runtime_csv_target is None or not os.path.exists(runtime_csv_target):
            st.error("❌ Execution Refused: The processing matrix pipeline is empty.")
        elif not user_directive.strip():
            st.warning("⚠️ Direction Missing: Please provide text instructions.")
        else:
            with st.spinner("Processing deep analytical arrays via high-speed Groq cloud nodes..."):
                try:
                    # 1. Trigger backend engine logic
                    backend_output = expert_data_analyst_engine.invoke({"csv_filename": runtime_csv_target})
                    
                    if "error" in backend_output:
                        st.error(backend_output["error"])
                    else:
                        # Save structural metadata layout to session state
                        st.session_state.active_matrix_data = backend_output
                        
                        # Isolate texts securely for prompt ingestion
                        summary_payload = f"""
======= DATASET METADATA =======
Total Records Isolation: {backend_output['total_records']}
Discovered Column Layout: {backend_output['column_layout']}

======= SUMMARY STATISTICS =======
{backend_output['summary_text']}

======= CORRELATION MATRIX COEFFS =======
{backend_output['correlation_text']}
"""
                        # Slice securely under token caps
                        if len(summary_payload) > 8000:
                            summary_payload = summary_payload[:8000] + "\n\n[...Truncated...]"

                        st.info("🧠 Syncing metrics to cloud infrastructure for executive synthesis...")
                        
                        # 2. Complete LCEL pipeline execution
                        groq_vault_key = st.secrets["GROQ_API_KEY"]
                        model_layer = ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=groq_vault_key, temperature=0.1)
                        
                        report_prompt = ChatPromptTemplate.from_template("""SYSTEM: You are a Lead Data Scientist. Compile a concise engineering report using the following headers and bullets:
1. EXECUTIVE DATA SUMMARY
2. CORE DIFFERENTIAL ANALYSIS (Quantified values)
3. STATISTICAL OBSERVATIONS (Non-basic variance/skewness)
4. CORRELATIONS & RELATIONSHIPS
5. STRATEGIC INSIGHTS

USER DIRECTIVE: {directive}

DATASET PAYLOAD:
{payload}""")
                        
                        orchestration_chain = report_prompt | model_layer | StrOutputParser()
                        compiled_insights = orchestration_chain.invoke({
                            "directive": user_directive,
                            "payload": summary_payload
                        })
                        
                        st.success("🎯 Cloud Analytics Cycle Complete!")
                        st.text_area("📋 Compiled Lead Data Scientist Report:", compiled_insights, height=320)
                        
                except Exception as runtime_fault:
                    st.error(f"🚨 Cloud Pipeline Exception: {runtime_fault}")

# --- 5. HIGH-RESOLUTION FULL-WIDTH VISUAL DISPLAY BLOCK (FULLY DYNAMIC LAYER) ---
if st.session_state.active_matrix_data and uploaded_file is not None:
    st.markdown("---")
    st.header("📈 Interactive Analytics Matrix Profile (Fully Customizable Grid Output)")
    
    meta = st.session_state.active_matrix_data
    plot_df = pd.read_csv(meta["file_path"])
    all_cols = meta["column_layout"]
    num_cols = meta["numeric_columns"]
    
    # 🎛️ NEW: DYNAMIC USER CONTROL PANELS
    st.markdown("### 🎛️ Chart Layout Controls")
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
    
    with ctrl_col1:
        selected_x_axis = st.selectbox("Select X-Axis (Timeline / Base Dimension)", options=all_cols, index=0)
    with ctrl_col2:
        selected_y_features = st.multiselect("Select Trend Line Features (Y-Axis)", options=num_cols, default=num_cols[:2])
    with ctrl_col3:
        scatter_target_y = st.selectbox("Select Scatter Target Variable (Y-Axis)", options=num_cols, index=min(1, len(num_cols)-1))

    st.markdown("---")
    
    # Render dynamic grid partitions natively based on widget selections
    v_col1, v_col2 = st.columns(2)
    
    with v_col1:
        # 1. Continuous Trend Chart (Fully Dynamic)
        if selected_y_features:
            fig_trend = px.line(plot_df, x=selected_x_axis, y=selected_y_features, title=f"1. Dynamic Trend: Custom Dimensions over {selected_x_axis}", markers=True)
            fig_trend.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.warning("⚠️ Please select at least one feature for the Trend Line chart.")
        
        # 2. Statistical Distribution Chart (Fully Dynamic)
        fig_dist = px.histogram(plot_df, x=selected_y_features if selected_y_features else num_cols[:1], barmode="overlay", title="3. Distribution Density Profile")
        fig_dist.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dist, use_container_width=True)

    with v_col2:
        # 3. Dimension Comparison Chart (Fully Dynamic)
        features_to_average = selected_y_features if selected_y_features else num_cols
        mean_df = plot_df[features_to_average].mean().reset_index()
        mean_df.columns = ["Feature", "Mean Magnitude"]
        fig_bar = px.bar(mean_df, x="Feature", y="Mean Magnitude", color="Feature", title="2. Dimension Comparison Matrix (Active Selections)")
        fig_bar.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 4. Dimensional Scatter Chart (Fully Dynamic)
        fig_scatter = px.scatter(plot_df, x=selected_x_axis, y=scatter_target_y, title=f"4. Dimensional Relationship ({selected_x_axis} vs {scatter_target_y})")
        fig_scatter.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_scatter.update_traces(marker=dict(size=12, symbol="diamond", color="crimson"))
        st.plotly_chart(fig_scatter, use_container_width=True)
