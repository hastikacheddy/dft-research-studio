"""
visualization/noise_sensitivity.py
------------------------------------
Noise-collapse line chart and failure-mode taxonomy bar chart.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _ensure_system_type(df: pd.DataFrame) -> pd.DataFrame:
    if "system_type" not in df.columns:
        df = df.copy()
        df["system_type"] = df["experiment_type"].apply(lambda x: x.split("_")[0])
    return df


def plot_noise_sensitivity(df_metrics: pd.DataFrame) -> None:
    """Line chart: Correctness vs Distractor Ratio per architecture."""
    df = _ensure_system_type(df_metrics)

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="distractor_ratio",
        y="correctness",
        hue="system_type",
        marker="o",
        linewidth=2.5,
    )
    plt.title(
        "Robustness Profile: Correctness vs. Noise Level",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Distractor Ratio (Noise)", fontsize=12)
    plt.ylabel("Mean Correctness (1-5)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Architecture", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("noise_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_failure_modes(df_metrics: pd.DataFrame) -> None:
    """Stacked horizontal bar showing failure mode proportions per architecture."""
    df = _ensure_system_type(df_metrics.copy())

    def _categorize(row):
        if row.get("correctness", 0) >= 4.0:
            return "Success"
        chunks = row.get("retrieved_text_chunks", "[]")
        if not chunks or chunks == "[]":
            return "Retrieval Failure (No Context)"
        return "Reasoning / Synthesis Failure"

    df["Failure_Mode"] = df.apply(_categorize, axis=1)
    ct = pd.crosstab(df["system_type"], df["Failure_Mode"], normalize="index")

    ct.plot(
        kind="barh",
        stacked=True,
        figsize=(11, 6),
        color=["#e74c3c", "#f39c12", "#2ecc71"],
    )
    plt.title("Failure Mode Taxonomy by Architecture", fontsize=14, fontweight="bold")
    plt.xlabel("Proportion of Queries", fontsize=12)
    plt.ylabel("Architecture", fontsize=12)
    plt.legend(title="Status", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("failure_modes.png", dpi=150, bbox_inches="tight")
    plt.show()
