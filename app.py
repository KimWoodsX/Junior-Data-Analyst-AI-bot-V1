import streamlit as st
import os
import glob
import time
import pandas as pd
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

# Initialize unique runtime state monitoring tokens to track visual files across sessions
if "session_token" not in st.session_state:
    st.session_state.session_token = str(int(time.time()))
if "active_matrix_path" not in st.session_state:
    st.session_state.active_matrix_path = None

# --- 2. SIDEBAR CONTROL PANELS: SECURE INGESTION LAYER ---
st.sidebar.title("📁 Control Panel Workspace")
st.sidebar.write("Ingest structured data arrays securely (CSV formatting only).")

uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"], label_visibility="collapsed")
st.sidebar.text_input("🔍 Search active folder workspace", placeholder="Search filenames...")
human_approval = st.sidebar.checkbox("Require Manual Human Approval", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Infrastructure Status: **Cloud API Inversion Layer Active (Groq Node)**")

# --- 3. MAIN BLOCK AREA: HERO HEADERS ---
st.title("✨ Enterprise Local Data Analyst & File Co-Pilot")
st.write("Orchestrating state-of-the-art analytical data pipelines via LangChain Expression Language (LCEL)")
st.markdown("---")

col1, col2 = st.columns([0.5, 0.5])

# Defensive initialization tracking token
runtime_csv_target = None

# --- COLUMN 1: ACTIVE DATASET ISOLATION & PREVIEW ---
with col1:
    st.header("📊 Active Dataset Preview")
    if uploaded_file is not None:
        try:
            # Synchronize thread to read the uploaded data buffer metrics safely
            df = pd.read_csv(uploaded_file)
            st.success(f"📦 Source Data Ingested: '{uploaded_file.name}' — Discovered {len(df)} discrete matrices.")
            st.dataframe(df.head(12), use_container_width=True)
            
            # Formulate a bulletproof path bound strictly to this individual file entity
            runtime_csv_target = f"secure_target_{st.session_state.session_token}.csv"
            with open(runtime_csv_target, "wb") as secure_stream:
                secure_stream.write(uploaded_file.getbuffer())
                
        except Exception as ingestion_error:
            st.error(f"🚨 Structural Ingestion Fault: Unable to parse binary headers. Details: {ingestion_error}")
    else:
        st.info("💡 Storage Core Standby: Please upload a .csv data array in the sidebar container to initialize validation.")

# --- COLUMN 2: ANALYTICAL AGENT CONTROL SYSTEM ---
with col2:
    st.header("⚙️ AI Agent Control Center")
    st.write("Transmit high-priority optimization directives to your analytics execution layer:")
    
    user_directive = st.text_area(
        "Directives",
        placeholder="e.g., Run a deep statistical variance audit on the active rows, map feature correlations, and output the visual arrays.",
        height=150,
        label_visibility="collapsed"
    )
    
    execute_pipeline = st.button("Execute Pipeline Analysis 🚀", use_container_width=True)
    
    if execute_pipeline:
        if runtime_csv_target is None or not os.path.exists(runtime_csv_target):
            st.error("❌ Execution Refused: The processing matrix pipeline is empty. Upload a CSV source array first.")
        elif not user_directive.strip():
            st.warning("⚠️ Direction Missing: Please provide explicit text instructions for the model layers to compile.")
        else:
            with st.spinner("Processing deep analytical arrays via high-speed Groq cloud nodes..."):
                try:
                    # 1. Trigger backend engine logic to extract statistical deltas and plot charts
                    raw_payload = expert_data_analyst_engine.invoke({"csv_filename": runtime_csv_target})
                    
                    # --- AGGRESSIVE TRUE-CLOUD PAYLOAD TRUNCATION ---
                    # Slicing at 8000 characters caps data text at ~2,000 tokens, crushing the 413 exception.
                    stringified_payload = str(raw_payload)
                    if len(stringified_payload) > 8000:
                        stringified_payload = stringified_payload[:8000] + "\n\n[...System Warning: Analytical matrix truncated for cloud constraints...]"
                    
                    st.info("🧠 Syncing metrics to cloud infrastructure for executive synthesis...")
                    
                    # 2. Extract secure environment secrets securely
                    groq_vault_key = st.secrets["GROQ_API_KEY"]
                    
                    # 3. Spin up high-performance cloud inference layer
                    model_layer = ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=groq_vault_key, temperature=0.1)
                    
                    # --- FLAT SYSTEM PROMPT DESIGN (Reduces invisible token overhead) ---
                    report_prompt = ChatPromptTemplate.from_template("""SYSTEM: You are a Lead Data Scientist. Compile a concise engineering report using the following headers and bullets:
1. EXECUTIVE DATA SUMMARY
2. CORE DIFFERENTIAL ANALYSIS (Quantified values)
3. STATISTICAL OBSERVATIONS (Non-basic variance/skewness)
4. CORRELATIONS & RELATIONSHIPS
5. STRATEGIC INSIGHTS

USER DIRECTIVE: {directive}

DATASET PAYLOAD:
{payload}""")
                    
                    # 4. Invoke secure operational LCEL completion pipeline
                    orchestration_chain = report_prompt | model_layer | StrOutputParser()
                    compiled_insights = orchestration_chain.invoke({
                        "directive": user_directive,
                        "payload": stringified_payload
                    })
                    
                    st.success("🎯 Cloud Analytics Cycle Complete!")
                    st.text_area("📋 Compiled Lead Data Scientist Report:", compiled_insights, height=320)
                    
                    # 5. GARBAGE COLLECTION & CACHE INVALIDATION RULE
                    if os.path.exists("advanced_analytics_matrix.png"):
                        fresh_matrix_id = f"matrix_{st.session_state.session_token}_{int(time.time())}.png"
                        for stale_chart in glob.glob("matrix_*.png"):
                            try: os.remove(stale_chart)
                            except: pass
                        os.rename("advanced_analytics_matrix.png", fresh_matrix_id)
                        st.session_state.active_matrix_path = fresh_matrix_id
                        
                except Exception as runtime_fault:
                    st.error(f"🚨 Cloud Pipeline Exception: {runtime_fault}")

# --- 5. HIGH-RESOLUTION FULL-WIDTH VISUAL DISPLAY BLOCK ---
if st.session_state.active_matrix_path and os.path.exists(st.session_state.active_matrix_path):
    st.markdown("---")
    st.header("📈 Headless Analytics Matrix Profile (2x2 Statistical Grid Output)")
    st.write(f"Active Grid Asset Token ID: `{st.session_state.active_matrix_path}`")
    st.image(st.session_state.active_matrix_path, use_container_width=True)
