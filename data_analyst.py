import os
import numpy as np
import pandas as pd

# ─── CRITICAL MULTI-THREAD FIX ───────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')  # Forces Matplotlib to run in safe, headless mode
import matplotlib.pyplot as plt
import seaborn as sns
# ─────────────────────────────────────────────────────────────────────────

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

print("⚡ Initializing Headless-Safe Data Analyst Backend Architecture...")
model = ChatOllama(model="llama3.2", temperature=0.1)

@tool
def expert_data_analyst_engine(csv_filename: str) -> str:
    """
    Executes deep analytical operations on a local CSV dataset.
    Generates multi-dimensional charts (Trend, Comparison, Distribution, Correlation)
    and extracts statistical variance insights without basic reasoning.
    """
    # ─── 🛡️ SECURITY LAYER: HUMAN-IN-THE-LOOP CHECK ───────────────────────────
    print(f"\n⚠️  [SECURITY TRAP] Agent requesting read/write access on: '{csv_filename}'")
    user_auth = input("Authorize data pipeline execution? (type 'yes' to proceed): ")
    if user_auth.strip().lower() != 'yes':
        return "[EXECUTION ABORTED] Security guardrail tripped: User denied access."
    # ──────────────────────────────────────────────────────────────────────────

    if not os.path.exists(csv_filename):
        return f"[ERROR] Targeted data file '{csv_filename}' cannot be located."

    try:
        df = pd.read_csv(csv_filename)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return "[STRUCTURAL ERROR] Dataset lacks sufficient numeric dimensions for comprehensive analysis."

        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Advanced Analytical Profile: {csv_filename}", fontsize=18, fontweight='bold')

        # 1. Trend Over Time
        time_axis = df.columns[0]
        for col in numeric_cols[:3]:
            axes[0, 0].plot(df[time_axis], df[col], marker='o', linewidth=2, label=f"Trend: {col}")
        axes[0, 0].set_title("1. Continuous Trend Over Time Matrix", fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel(str(time_axis))
        axes[0, 0].set_ylabel("Scalar Value")
        axes[0, 0].legend()

        # 2. Comparison Metrics
        mean_values = df[numeric_cols].mean()
        axes[0, 1].bar(mean_values.index, mean_values.values, color=sns.color_palette("muted"))
        axes[0, 1].set_title("2. Dimension Comparison Matrix (Mean Aggregates)", fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel("Tracked Features")
        axes[0, 1].set_ylabel("Mean Magnitude")

        # 3. Statistical Distribution and Spread
        for col in numeric_cols[:2]:
            sns.histplot(df[col], kde=True, ax=axes[1, 0], alpha=0.4, label=f"Spread: {col}")
        axes[1, 0].set_title("3. Distribution Density & Data Spread Profiles", fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel("Value Interval")
        axes[1, 0].set_ylabel("Frequency Count")
        axes[1, 0].legend()

        # 4. Relationships & Feature Correlations
        sns.scatterplot(data=df, x=numeric_cols[0], y=numeric_cols[1], ax=axes[1, 1], s=100, color="crimson", marker="D")
        axes[1, 1].set_title(f"4. Dimensional Relationship Matrix ({numeric_cols[0]} vs {numeric_cols[1]})", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel(numeric_cols[0])
        axes[1, 1].set_ylabel(numeric_cols[1])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_chart_path = "advanced_analytics_matrix.png"
        plt.savefig(output_chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Explicitly clear graphic context memory

        summary_statistics = df[numeric_cols].describe()
        correlation_matrix = df[numeric_cols].corr()
        
        insights_payload = {
            "file_metadata": {"total_records": len(df), "dimensions": list(df.columns)},
            "statistical_deltas": summary_statistics.to_dict(),
            "correlation_coefficients": correlation_matrix.to_dict(),
            "visual_status": f"Charts saved successfully as {output_chart_path}"
        }
        return f"[SYSTEM EXECUTION OUTPUT]\nData Payload sent to LLM for final report assembly:\n{insights_payload}"

    except Exception as data_fault:
        return f"[RUNTIME CRITICAL ERROR] Pipeline execution failed: {str(data_fault)}"

# --- 🛠️ MODERN AGENT ORCHESTRATION LAYER FIX ---
tools = [expert_data_analyst_engine]

# Create a prompt structured with placeholder arrays required by modern tool agents
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are acting as an Advanced Lead Data Scientist. Always use your engine tool first to gather data analytics parameters before writing your final executive report."),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Assemble the tool calling agent using modern LangChain paradigms
base_agent = create_tool_calling_agent(model=model, tools=tools, prompt=agent_prompt)
agent_executor = AgentExecutor(agent=base_agent, tools=tools, verbose=False)

expert_instruction = """
Analyze the local file 'marketing_sales.csv' using your engine tool.
When the tool sends back the raw mathematical aggregates and confirmation of charts generation, compile a formal engineering report.

Your report must strictly contain the following explicit sections:
1. EXECUTIVE DATA SUMMARY: State what the data structurally represents.
2. CORE DIFFERENTIAL ANALYSIS: Highlight the specific, quantified numerical differences between tracking variables.
3. STATISTICAL OBSERVATIONS (NON-BASIC): Identify anomalies, feature variance, and skewness. Do not state obvious or basic points.
4. CORRELATIONS & RELATIONSHIPS: Detail how variables interact based on mathematical dependencies.
5. STRATEGIC IN-DEPTH INSIGHTS: Provide advanced conclusions regarding trends or patterns observed across cycles.

Ensure all outputs are formatted with clear engineering headers and structured bullet points.
"""

print("🚀 Launching Data Analytics Pipeline...")
# Invoke the modern executor block using standard formatting inputs
pipeline_result = agent_executor.invoke({
    "messages": [("user", expert_instruction)]
})

print("\n📊 === FINAL BACKEND ANALYST REPORT ===")
print(pipeline_result["output"])
print("=======================================")

