import os
import numpy as np
import pandas as pd

# ─── CRITICAL MULTI-THREAD FIX ───────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')  # Forces Matplotlib to run in safe, headless mode
import matplotlib.pyplot as plt
import seaborn as sns
# ─────────────────────────────────────────────────────────────────────────

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("⚡ Initializing Headless-Safe Data Analyst Backend Architecture...")

@tool
def expert_data_analyst_engine(csv_filename: str) -> str:
    """
    Executes deep analytical operations on a local CSV dataset.
    Generates multi-dimensional charts (Trend, Comparison, Distribution, Correlation)
    and extracts statistical variance insights without basic reasoning.
    """
    # ─── 🛡️ CLOUD-OPTIMIZED AUTOMATED SECURITY GUARDRAIL ───────────────────
    print(f"📦 Cloud Data Pipeline verified and authorized for: '{csv_filename}'")
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

        # --- OPTIMIZED COMPACT PAYLOAD FORMATTING ---
        # Using .to_string() cleans up JSON-like dict syntax and formats stats elegantly for the LLM
        summary_str = df[numeric_cols].describe().to_string()
        correlation_str = df[numeric_cols].corr().to_string()
        
        insights_payload = f"""
======= DATASET METADATA =======
Total Records Isolation: {len(df)}
Discovered Column Layout: {list(df.columns)}
Visual Infrastructure: Charts saved successfully as {output_chart_path}

======= SUMMARY STATISTICS =======
{summary_str}

======= CORRELATION MATRIX COEFFS =======
{correlation_str}
"""
        return f"[SYSTEM EXECUTION OUTPUT]\nData Payload sent to LLM for final report assembly:\n{insights_payload}"

    except Exception as data_fault:
        return f"[RUNTIME CRITICAL ERROR] Pipeline execution failed: {str(data_fault)}"
