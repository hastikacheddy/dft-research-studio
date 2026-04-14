"""
visualization/radar.py
-----------------------
Architecture capability radar chart.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_radar_comparison(df_metrics: pd.DataFrame) -> None:
    """Radar chart comparing normalised mean metrics per architecture."""
    categories = ["correctness", "relevance", "groundedness", "rougeL_fmeasure"]
    df = df_metrics.copy()
    df["correctness"] = df["correctness"] / 5.0  # normalise to [0,1]

    if "system_type" not in df.columns:
        df["system_type"] = df["experiment_type"].apply(lambda x: x.split("_")[0])

    summary = df.groupby("system_type")[categories].mean()
    label_locs = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    for model, row in summary.iterrows():
        vals = row.values.tolist() + row.values.tolist()[:1]
        locs = np.append(label_locs, label_locs[0])
        ax.plot(locs, vals, label=model, linewidth=2)
        ax.fill(locs, vals, alpha=0.1)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(label_locs, [c.replace("_", " ").title() for c in categories])
    plt.title("Architecture Capability Comparison (Normalised)", size=15, y=1.1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("radar_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
