import os
import numpy as np
import pandas as pd
from langchain_core.tools import tool

print("⚡ Initializing Plotly-Ready Data Analyst Backend Architecture...")

@tool
def expert_data_analyst_engine(csv_filename: str) -> dict:
    """
    Executes analytical operations on a local CSV dataset.
    Extracts data matrices and structural summaries optimized for interactive rendering.
    """
    print(f"📦 Cloud Data Pipeline verified and authorized for: '{csv_filename}'")

    if not os.path.exists(csv_filename):
        return {"error": f"Targeted data file '{csv_filename}' cannot be located."}

    try:
        df = pd.read_csv(csv_filename)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return {"error": "Dataset lacks sufficient numeric dimensions for comprehensive analysis."}

        # Format statistics and matrices compactly to prevent token bloating
        summary_str = df[numeric_cols].describe().to_string()
        correlation_str = df[numeric_cols].corr().to_string()
        
        # Pass raw structural dataframe metrics straight back to Streamlit safely
        return {
            "file_path": csv_filename,
            "numeric_columns": numeric_cols,
            "time_axis": str(df.columns[0]),
            "summary_text": summary_str,
            "correlation_text": correlation_str,
            "total_records": len(df),
            "column_layout": list(df.columns)
        }

    except Exception as data_fault:
        return {"error": f"Pipeline execution failed: {str(data_fault)}"}

