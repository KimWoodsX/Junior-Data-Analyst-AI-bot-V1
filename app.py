import streamlit as st
import os
import pandas as pd
from data_analyst import expert_data_analyst_engine

# --- 1. ENTERPRISE PAGE CONFIGURATION ---
st.set_page_config(page_title="Local Data Analyst Hub", layout="wide")

# --- 2. SIDEBAR: CONTROL PANEL WORKSPACE ---
st.sidebar.title("📁 Control Panel Workspace")
st.sidebar.write("Upload analytical workspace datasets (CSV only)")

uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"], label_visibility="collapsed")
st.sidebar.text_input("🔍 Search active folder workspace", placeholder="Search filenames...")
human_approval = st.sidebar.checkbox("Require Manual Human Approval", value=True)


# --- 3. MAIN AREA: HERO HEADERS ---
st.title("✨ Local Junior Data Analyst & File Co-Pilot")
st.write("Powered by LangChain & Llama 3.2 — 100% Offline, Secure & Private Workspace")
st.markdown("---")

# --- 4. TWO-COLUMN OPERATIONAL DASHBOARD LAYOUT ---
col1, col2 = st.columns([0.5, 0.5])

# --- COLUMN 1: ACTIVE DATASET PREVIEW ---
with col1:
    st.header("📊 Active Dataset Preview")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Previewing: {uploaded_file.name} ({len(df)} records discovered)")
            st.dataframe(df.head(10), use_container_width=True)
            
            with open("marketing_sales.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as file_error:
            st.error(f"Error reading file parameters: {file_error}")
    else:
        st.info("💡 Please upload a .csv file in the left control panel sidebar to preview your dataset.")

# --- COLUMN 2: AI AGENT CONTROL CENTER ---
with col2:
    st.header("⚙️ AI Agent Control Center")
    st.write("Enter analytical directives for your local agent:")
    
    user_directive = st.text_area(
        "Directives",
        placeholder="e.g., Run an advanced analytics operation on marketing_sales.csv...",
        height=150,
        label_visibility="collapsed"
    )
    
    if st.button("Execute Pipeline Analysis 🚀", use_container_width=True):
        if uploaded_file is None and not os.path.exists("marketing_sales.csv"):
            st.error("❌ Execution Aborted: No active dataset matrix located in the workspace.")
        else:
            with st.spinner("Processing statistical matrices... (Authorize 'yes' in your terminal if prompted!)"):
                try:
                    # ─── 🛠️ THE CRITICAL BROWSER ENGINE FIX ──────────────────────────
                    # Instead of calling it like a raw function, we must use .invoke()
                    # and pass the input as a structured dictionary format:
                    report_output = expert_data_analyst_engine.invoke({"csv_filename": "marketing_sales.csv"})
                    # ────────────────────────────────────────────────────────────────
                    
                    st.success("✅ Analysis Complete!")
                    st.text_area("📋 Compiled Lead Data Scientist Report:", report_output, height=250)
                    
                    if os.path.exists("advanced_analytics_matrix.png"):
                        st.image("advanced_analytics_matrix.png", caption="Headless Analytics Matrix Profile (2x2 Statistical Grid Output)")
                except Exception as e:
                    st.error(f"Runtime Pipeline Fault: {e}")


