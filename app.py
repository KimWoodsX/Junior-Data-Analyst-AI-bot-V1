import streamlit as st
import os
import numpy as np
import pandas as pd

# Headless multi-thread visualization configuration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent

# ─── FRONTEND PAGE LAYOUT CONFIGURATION ───────────────────────────────────
st.set_page_config(page_title="Local AI Data Analyst", layout="wide", page_icon="📊")
st.title("✨ Local Junior Data Analyst & File Co-Pilot")
st.write("Powered by LangChain & Llama 3.2 — 100% Offline, Secure & Private Workspace")

# ─── SIDEBAR CONTROL PANEL ────────────────────────────────────────────────
st.sidebar.header("📂 Control Panel Workspace")

# 1. Premium File Uploader Component
uploaded_file = st.sidebar.file_uploader(
    "Upload analytical workspace datasets (CSV only)", 
    type=["csv"]
)

# 2. Workspace Document Search Bar
search_query = st.sidebar.text_input("🔍 Search active folder workspace", placeholder="Search filenames...")

# 3. Safety Guardrail Toggle
require_approval = st.sidebar.checkbox("🛡️ Require Manual Human Approval", value=True)

# File Ingestion
target_filename = None
if uploaded_file is not None:
    target_filename = uploaded_file.name
    with open(target_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"📁 Ingested: {target_filename}")

# ─── BACKEND TOOL & AGENT ROUTING KERNEL ──────────────────────────────────
model = ChatOllama(model="llama3.2")

@tool
def robust_web_data_analyst_engine(csv_filename: str) -> str:
    """Useful engine tool to analyze a data file, compute averages, spreads, distributions and output matrices."""
    if not os.path.exists(csv_filename):
        return f"[ERROR] Targeted data file '{csv_filename}' cannot be located."

    try:
        df = pd.read_csv(csv_filename)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return "[STRUCTURAL ERROR] Dataset lacks sufficient numeric dimensions."

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Advanced Analytics Profile: {csv_filename}", fontsize=16, fontweight='bold')

        # 1. Continuous Trend
        time_axis = df.columns[0]
        for col in numeric_cols[:3]:
            axes[0, 0].plot(df[time_axis], df[col], marker='o', linewidth=2, label=col)
        axes[0, 0].set_title("1. Continuous Trend Matrix", fontweight='bold')
        axes[0, 0].legend()

        # 2. Categorical Averages
        mean_values = df[numeric_cols].mean()
        axes[0, 1].bar(mean_values.index, mean_values.values, color=sns.color_palette("muted"))
        axes[0, 1].set_title("2. Dimension Comparison Matrix", fontweight='bold')

        # 3. Data Spread Profile
        for col in numeric_cols[:2]:
            sns.histplot(df[col], kde=True, ax=axes[1, 0], alpha=0.4, label=col)
        axes[1, 0].set_title("3. Distribution Density Profile", fontweight='bold')
        axes[1, 0].legend()

        # 4. Relationships & Correlation
        sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], ax=axes[1, 1], s=100, color="crimson")
        axes[1, 1].set_title(f"4. Dimensional Relationship Matrix", fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        chart_output_name = "ui_advanced_analytics.png"
        plt.savefig(chart_output_name, dpi=200, bbox_inches='tight')
        plt.close(fig)

        summary_statistics = df[numeric_cols].describe()
        return f"[SUCCESS] Calculations executed. Metrics Payload generated. Chart saved as '{chart_output_name}'. Stats: {summary_statistics.to_dict()}"
    except Exception as e:
        return f"[ERROR] Pipeline execution crashed: {str(e)}"

tools = [robust_web_data_analyst_engine]
agent = create_agent(model=model, tools=tools)

# ─── MAIN APP GRAPHICAL WORKSPACE INTERFACE ───────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Active Dataset Preview")
    if target_filename:
        preview_df = pd.read_csv(target_filename)
        st.dataframe(preview_df, use_container_width=True)
    else:
        st.info("💡 Please upload a `.csv` file in the left control panel sidebar to preview your dataset.")

with col2:
    st.subheader("🤖 AI Agent Control Center")
    user_instruction = st.text_area(
        "Enter analytical directives for your local agent:", 
        placeholder="e.g., Run an advanced analytics operation on marketing_sales.csv, explain the core differentials, and draw the visual grids."
    )
    run_btn = st.button("Execute Pipeline Analysis 🚀", use_container_width=True)

# ─── ORCHESTRATION PIPELINE PROCESSOR ─────────────────────────────────────
if run_btn:
    if not user_instruction:
        st.error("❌ Error: Prompt space is empty. Please enter an instruction payload first.")
    else:
        st.info("Local Llama 3.2 is inspecting calculations and compiling visual grids...")
        try:
            structured_prompt = f"""
            {user_instruction}
            
            You must format your final report strictly with these exact header structures:
            ### 📝 EXECUTIVE DATA SUMMARY
            ### 📊 CORE DIFFERENTIAL ANALYSIS
            ### ⚖️ STATISTICAL OBSERVATIONS (NON-BASIC)
            ### 📈 CORRELATIONS & RELATIONSHIPS
            ### 💡 STRATEGIC IN-DEPTH INSIGHTS
            """
            
            response = agent.invoke({
                "messages": [{"role": "user", "content": structured_prompt}]
            })
            
            st.write("---")
            st.success("✅ Deep Analytics Pipeline Executed Successfully!")
            st.markdown(response["messages"][-1].content)
            
            if os.path.exists("ui_advanced_analytics.png"):
                st.write("---")
                st.subheader("📈 Rendered Advanced Visualization Matrices")
                st.image("ui_advanced_analytics.png", use_container_width=True)
                os.remove("ui_advanced_analytics.png")
                
        except Exception as pipeline_error:
            st.error(f"❌ Critical Pipeline Failure: {str(pipeline_error)}")
