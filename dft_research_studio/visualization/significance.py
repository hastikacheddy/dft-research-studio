"""
visualization/significance.py
------------------------------
Welch's T-test p-value matrix heatmap.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


def plot_significance_matrix(
    df_metrics: pd.DataFrame,
    metric: str = "correctness",
) -> None:
    """Heatmap of pairwise Welch T-test p-values between architectures."""
    if "system_type" not in df_metrics.columns:
        df_metrics = df_metrics.copy()
        df_metrics["system_type"] = df_metrics["experiment_type"].apply(
            lambda x: x.split("_")[0]
        )

    systems = df_metrics["system_type"].unique()
    matrix = pd.DataFrame(index=systems, columns=systems, dtype=float)

    for s1 in systems:
        for s2 in systems:
            if s1 == s2:
                matrix.loc[s1, s2] = 1.0
            else:
                g1 = df_metrics[df_metrics["system_type"] == s1][metric]
                g2 = df_metrics[df_metrics["system_type"] == s2][metric]
                _, p = stats.ttest_ind(g1, g2, equal_var=False)
                matrix.loc[s1, s2] = p

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix.astype(float), annot=True, cmap="viridis_r", fmt=".3f")
    plt.title(f"Statistical Significance (p-values) for {metric.title()}")
    plt.tight_layout()
    plt.savefig("significance_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Note: p < 0.05 indicates statistical significance.")
