"""
visualization/heatmap.py
-------------------------
Stratified multi-subplot heatmaps (metric × question-level).
"""

from __future__ import annotations

import ast
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import SVG, display


def plot_experiment_heatmap(
    results_df: pd.DataFrame,
    metric_name: str,
    ax: plt.Axes,
    title_suffix: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Render a single heatmap panel on *ax* for *metric_name*.

    Parameters
    ----------
    results_df   : DataFrame slice (usually one question level)
    metric_name  : key inside the 'metrics' dict column
    ax           : Matplotlib Axes to draw on
    title_suffix : appended to the subplot title (e.g. "for L2")
    vmin / vmax  : global colour-scale limits for cross-panel comparability
    """
    if "metrics" not in results_df.columns:
        ax.set_visible(False)
        return

    df = results_df.copy()

    # Ensure metrics column is a dict
    if df["metrics"].iloc[0] and isinstance(df["metrics"].iloc[0], str):
        df["metrics"] = df["metrics"].apply(ast.literal_eval)

    col = metric_name.lower().replace(" ", "_")
    df[col] = df["metrics"].apply(lambda x: x.get(metric_name, 0.0))

    pivot = df.pivot_table(
        index="experiment_type",
        columns="distractor_ratio",
        values=col,
        aggfunc="mean",
    )
    pivot.columns = [f"{int(c * 100)}%" for c in pivot.columns]

    sns.heatmap(
        pivot,
        annot=True,
        cmap="spring_r",
        fmt=".2f",
        linewidths=0.5,
        linecolor="black",
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"shrink": 1.0},
    )
    ax.set_title(
        f"Mean {metric_name.replace('_', ' ').title()} {title_suffix}",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlabel("Distractor Ratio", fontsize=8)
    ax.set_ylabel("Experiment Type", fontsize=8)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)


def generate_multi_level_heatmaps(
    df_metrics: pd.DataFrame,
    question_id_to_level: dict,
    metrics_to_plot: list,
    output_dir: str = ".",
) -> None:
    """
    For each metric, create a figure with one subplot per question level
    and save as SVG.
    """
    import os

    df_metrics = df_metrics.copy()
    df_metrics["question_level"] = df_metrics["question_id"].map(
        question_id_to_level
    )

    unique_levels = sorted(df_metrics["question_level"].dropna().unique())
    if not unique_levels:
        print("[Heatmap] No question levels found.")
        return

    # Ensure metrics are parsed
    if df_metrics["metrics"].iloc[0] and isinstance(df_metrics["metrics"].iloc[0], str):
        df_metrics["metrics"] = df_metrics["metrics"].apply(ast.literal_eval)

    for metric in metrics_to_plot:
        df_metrics[metric] = df_metrics["metrics"].apply(
            lambda x: x.get(metric, 0.0)
        )
        g_min = df_metrics[metric].min()
        g_max = df_metrics[metric].max()

        n = len(unique_levels)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 9), sharey=True)
        if n == 1:
            axes = [axes]

        for i, level in enumerate(unique_levels):
            subset = df_metrics[df_metrics["question_level"] == level].copy()
            if subset.empty:
                axes[i].set_visible(False)
                continue
            plot_experiment_heatmap(
                subset,
                metric_name=metric,
                ax=axes[i],
                title_suffix=f"for {level}",
                vmin=g_min,
                vmax=g_max,
            )
            if i > 0:
                axes[i].set_ylabel("")

        fig.suptitle(
            f"Comparison of {metric.replace('_', ' ').title()} across Question Levels",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        path = os.path.join(output_dir, f"heatmap_{metric}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Heatmap] Saved → '{path}'")
        plt.close(fig)
        display(SVG(path))
