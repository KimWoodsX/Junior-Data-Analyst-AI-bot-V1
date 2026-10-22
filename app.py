import streamlit as st
import os
import pandas as pd
from data_analyst import expert_data_analyst_engine
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
st.write("Powered by LangChain & Groq Architecture — 100% Secure Private Workspace")
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
    st.write("Enter analytical directives for your agent:")
    
    user_directive = st.text_area(
        "Directives",
        placeholder="e.g., Run an advanced analytics operation on marketing_sales.csv...",
        height=150,
        label_visibility="collapsed"
    )
    
    execute_pipeline = st.button("Execute Pipeline Analysis 🚀", use_container_width=True)
    
    if execute_pipeline:
        if uploaded_file is None and not os.path.exists("marketing_sales.csv"):
            st.error("❌ Execution Aborted: No active dataset matrix located in the workspace.")
        else:
            with st.spinner("Processing statistical matrices..."):
                try:
                    raw_tool_payload = expert_data_analyst_engine.invoke({"csv_filename": "marketing_sales.csv"})
                    
                    groq_key = st.secrets["GROQ_API_KEY"]
                    model = ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=groq_key, temperature=0.2)
                    
                    report_prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are an Advanced Lead Data Scientist. Take the raw mathematical aggregates provided and compile a formal engineering report.
                        Your report must strictly contain the following explicit sections:
                        1. EXECUTIVE DATA SUMMARY
                        2. CORE DIFFERENTIAL ANALYSIS
                        3. STATISTICAL OBSERVATIONS (NON-BASIC)
                        4. CORRELATIONS & RELATIONSHIPS
                        5. STRATEGIC IN-DEPTH INSIGHTS
                        Format with clear engineering headers and structured bullet points."""),
                        ("user", "Directive: {directive}\n\nRaw Data Payload:\n{payload}")
                    ])
                    
                    report_chain = report_prompt | model | StrOutputParser()
                    final_report = report_chain.invoke({"directive": user_directive, "payload": raw_tool_payload})
                    
                    st.success("✅ Analysis Complete!")
                    st.text_area("📋 Compiled Lead Data Scientist Report:", final_report, height=300)
                except Exception as e:
                    st.error(f"Runtime Pipeline Fault: {e}")

# --- 5. FULL-WIDTH VISUALIZATION GRID DISPLAY ---
# This breaks out of the column containers to render your charts across the entire screen width
if os.path.exists("advanced_analytics_matrix.png"):
    st.markdown("---")
    st.header("📈 Headless Analytics Matrix Profile (2x2 Statistical Grid Output)")
    st.image("advanced_analytics_matrix.png", use_container_width=True)
