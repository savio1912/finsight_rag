# src/evaluation/metrics.py

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def save_results(df: pd.DataFrame, output_dir: str = "data/processed"):
    """Save evaluation results to disk for the Streamlit UI to load."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = output_dir / f"ragas_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    return csv_path


def print_comparison_table(df: pd.DataFrame):
    """Print a clean comparison table across strategies."""
    metrics = ["faithfulness", "answer_relevancy",
               "context_precision", "context_recall"]

    summary = df.groupby("strategy")[metrics].mean().round(3)
    summary["avg_score"] = summary[metrics].mean(axis=1).round(3)
    summary = summary.sort_values("avg_score", ascending=False)

    print("\n" + "="*65)
    print("CHUNKING STRATEGY COMPARISON — RAGAS SCORES")
    print("="*65)
    print(summary.to_string())
    print("="*65)
    print(f"\nWinner: {summary.index[0]} chunking")
    return summary