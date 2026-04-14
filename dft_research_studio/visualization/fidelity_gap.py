"""
visualization/fidelity_gap.py
------------------------------
Publication-quality Generative Fidelity Gap heatmap.
"""

from __future__ import annotations

import ast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import Image, Markdown, display


def _clean(name: str) -> str:
    return (
        name.replace("GraphRAG", "Graph-Anchored")
        .replace("GraphDeterministic", "Graph-Deterministic")
        .replace("MultiAgent", "Multi-Agent")
        .split("_Ratio_")[0]
    )


def plot_fidelity_gap(
    df: pd.DataFrame,
    output_prefix: str = "fidelity_gap_heatmap",
    display_width: int = 600,
) -> None:
    """
    Compute and render the Generative Fidelity Gap heatmap.

    Parameters
    ----------
    df             : final_experiment_metrics DataFrame
    output_prefix  : base filename (saved as .png and .svg)
    display_width  : pixel width for inline notebook display
    """
    df = df.copy()

    # Parse correctness from metrics dict if not already a column
    if "correctness" not in df.columns or df["correctness"].isnull().all():
        def _correctness(m):
            try:
                d = ast.literal_eval(m) if isinstance(m, str) else m
                return d.get("correctness", 0)
            except Exception:
                return 0
        df["correctness"] = df["metrics"].apply(_correctness)

    df["system_type"] = df["experiment_type"].apply(_clean)

    # Ceiling = best mean correctness
    ceiling = df.groupby("experiment_type")["correctness"].mean().max() or 5.0

    rows = []
    for (sys, ratio), grp in df.groupby(["system_type", "distractor_ratio"]):
        gfg = 1 - (grp["correctness"].mean() / ceiling)
        rows.append({"Model": sys, "Distractor Ratio": ratio, "GFG": gfg})

    gap_df = pd.DataFrame(rows)
    pivot = gap_df.pivot(index="Model", columns="Distractor Ratio", values="GFG")

    order = [
        "Baseline", "StandardRAG", "Graph-Anchored",
        "Graph-Deterministic", "Multi-Agent",
    ]
    pivot = pivot.reindex([m for m in order if m in pivot.index])

    # --- Plot ---
    plt.rcParams["font.family"] = "sans-serif"
    fig = plt.figure(figsize=(5, 4), dpi=400)
    with sns.axes_style("white"):
        ax = sns.heatmap(
            pivot,
            annot=True,
            fmt=".1%",
            cmap="magma_r",
            vmin=0, vmax=1,
            square=True,
            linewidths=1.0,
            linecolor="white",
            cbar=True,
            cbar_kws={
                "label": "Fidelity Gap (Lower is Better)",
                "shrink": 0.6,
                "fraction": 0.04,
                "pad": 0.03,
                "ticks": [0, 0.25, 0.5, 0.75, 1.0],
            },
            annot_kws={"size": 8},
        )

    plt.ylabel("", fontsize=0)
    plt.yticks(fontsize=9, weight="bold", rotation=0)
    plt.xlabel("Distractor Ratio (Noise Level)", fontsize=9, weight="bold", labelpad=8)
    plt.xticks(fontsize=9, rotation=0)
    plt.title("Generative Fidelity Gap", fontsize=11, weight="bold", pad=12)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.set_label("Fidelity Gap", fontsize=8, weight="bold", labelpad=8)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.svg", format="png", bbox_inches="tight")
    plt.close()

    display(Markdown("### Generative Fidelity Gap Heatmap"))
    display(Image(filename=f"{output_prefix}.png", width=display_width))
