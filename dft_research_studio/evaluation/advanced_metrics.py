"""
evaluation/advanced_metrics.py
--------------------------------
Bootstrap CI computation, Generative Fidelity Gap (GFG), full stats table,
and bar / noise-collapse line chart grids.
"""

from __future__ import annotations

import ast
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..config import Config


class AdvancedMetrics:
    """
    Loads final_experiment_metrics.csv and provides:
    - Bootstrap 95 % CIs
    - Generative Fidelity Gap (GFG)
    - Full scientific results table
    - Bar-chart grid and noise-collapse line-chart grid
    """

    _METRIC_LABELS: Dict[str, str] = {
        "correctness": "Correctness (1-5)",
        "relevance": "Relevance (1-5)",
        "groundedness": "Groundedness (0-1)",
        "gold_retrieval_recall_at_k": "Recall@K",
        "gold_retrieval_precision_at_k": "Precision@K",
        "gold_retrieval_mrr": "MRR",
        "rouge1_fmeasure": "ROUGE-1 F1",
        "rougeL_fmeasure": "ROUGE-L F1",
    }

    _SYSTEM_PALETTE: Dict[str, tuple] = {}  # filled lazily

    _MARKERS: Dict[str, str] = {
        "Baseline": "o",
        "StandardRAG": "s",
        "GraphRAG": "^",
        "GraphDeterministic": "D",
        "MultiAgent": "v",
    }

    def __init__(
        self,
        metrics_path: str = "final_experiment_metrics.csv",
        results_path: str | None = None,
    ) -> None:
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}")

        self.df = pd.read_csv(metrics_path)
        self.df["metrics_dict"] = self.df["metrics"].apply(self._parse_metrics)

        # Expand metric columns
        self.available_metrics: List[str] = []
        for key in self._METRIC_LABELS:
            self.df[key] = self.df["metrics_dict"].apply(
                lambda x, k=key: float(x.get(k, 0.0))
            )
            if self.df[key].mean() > 0 or key.startswith("gold_"):
                self.available_metrics.append(key)

        # Normalise system_type
        self.df["system_type"] = self.df["experiment_type"].apply(
            lambda x: x.split("_Ratio_")[0]
        )

        # Optionally load raw results JSON
        self.results: List[Dict] = []
        if results_path and os.path.exists(results_path):
            with open(results_path) as f:
                self.results = json.load(f)

        # Build colour palette
        cp = sns.color_palette("colorblind")
        self._SYSTEM_PALETTE = {
            "Baseline": cp[3],
            "StandardRAG": cp[0],
            "GraphRAG": cp[2],
            "GraphDeterministic": cp[4],
            "MultiAgent": cp[1],
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_metrics(raw) -> Dict:
        if isinstance(raw, dict):
            return raw
        try:
            return ast.literal_eval(raw)
        except Exception:
            return {}

    @staticmethod
    def _clean_name(name: str) -> str:
        return (
            name.replace("GraphRAG", "Graph-Anchored")
            .replace("GraphDeterministic", "Graph-Deterministic")
            .replace("MultiAgent", "Multi-Agent")
        )

    def get_bootstrap_ci(
        self, data: np.ndarray, n_boot: int = 1_000, ci: float = 0.95
    ) -> Tuple[float, float, float, float, float]:
        """Returns (mean, lower, upper, err_lower, err_upper)."""
        if len(data) < 2:
            m = float(np.mean(data)) if len(data) else 0.0
            return m, m, m, 0.0, 0.0
        rng = np.random.default_rng(42)
        boot_means = [
            rng.choice(data, size=len(data), replace=True).mean()
            for _ in range(n_boot)
        ]
        mean = float(np.mean(data))
        lower = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
        upper = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
        return mean, lower, upper, mean - lower, upper - mean

    # ------------------------------------------------------------------ #
    # GFG                                                                  #
    # ------------------------------------------------------------------ #

    def calculate_generative_gap(self) -> pd.DataFrame:
        print("\n[AdvancedMetrics] Computing Generative Fidelity Gap …")
        stats = self.df.groupby("experiment_type")["correctness"].agg(["mean"])
        ceiling = stats["mean"].max() or 5.0

        rows = []
        for (sys_type, ratio), group in self.df.groupby(
            ["system_type", "distractor_ratio"]
        ):
            m = group["correctness"].mean()
            s = group["correctness"].std()
            n = group["correctness"].count()
            gfg = 1 - (m / ceiling)
            gfg_err = ((s / math.sqrt(max(n, 1))) / ceiling) if n > 1 else 0.0
            rows.append(
                {
                    "Model": self._clean_name(sys_type),
                    "Distractor Ratio": ratio,
                    "Score (μ ± σ)": f"{m:.2f} ± {s:.2f}",
                    "Fidelity Gap": gfg,
                    "Gap Display": f"{gfg:.2%} (±{gfg_err:.2%})",
                    "Interpretation": (
                        "Critical Loss" if gfg > 0.25 else "High Fidelity"
                    ),
                }
            )

        gfg_df = pd.DataFrame(rows).sort_values(
            ["Distractor Ratio", "Fidelity Gap"]
        )
        print(gfg_df.drop(columns=["Fidelity Gap"]).to_markdown(index=False))
        gfg_df.to_csv("table3_generative_gap.csv", index=False)
        return gfg_df

    # ------------------------------------------------------------------ #
    # Stats table                                                          #
    # ------------------------------------------------------------------ #

    def generate_stats_table(self) -> pd.DataFrame:
        print("\n[AdvancedMetrics] Building statistics table …")
        rows = []
        for (sys_type, ratio), group in self.df.groupby(
            ["system_type", "distractor_ratio"]
        ):
            row: Dict = {
                "System": self._clean_name(sys_type),
                "Distractor Ratio": ratio,
            }
            for m in self.available_metrics:
                vals = group[m].values
                if len(vals):
                    mean, lower, upper, *_ = self.get_bootstrap_ci(vals)
                    label = self._METRIC_LABELS.get(m, m)
                    row[f"{label} (95% CI)"] = f"{mean:.2f} [{lower:.2f}, {upper:.2f}]"
                else:
                    row[f"{self._METRIC_LABELS.get(m, m)} (95% CI)"] = "N/A"
            rows.append(row)

        summary = pd.DataFrame(rows)
        print("\n" + "=" * 80)
        print("FULL SCIENTIFIC RESULTS TABLE")
        print("=" * 80)
        print(summary.to_markdown(index=False))
        summary.to_csv("final_results_with_ci.csv", index=False)
        print("[AdvancedMetrics] Saved 'final_results_with_ci.csv'.")
        return summary

    # ------------------------------------------------------------------ #
    # Plots                                                                #
    # ------------------------------------------------------------------ #

    def _make_plot_grid(self, plot_type: str) -> None:
        system_types = sorted(self.df["system_type"].unique())
        clean_systems = [self._clean_name(s) for s in system_types]
        colors = [
            self._SYSTEM_PALETTE.get(s, sns.color_palette("tab10")[i % 10])
            for i, s in enumerate(system_types)
        ]

        n = len(self.available_metrics)
        cols = 4
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes_flat = axes.flatten() if n > 1 else [axes]

        for i, ax in enumerate(axes_flat):
            if i >= n:
                ax.set_visible(False)
                continue

            metric = self.available_metrics[i]

            if plot_type == "bar":
                plot_data = []
                for sys in system_types:
                    vals = self.df[self.df["system_type"] == sys][metric].values
                    if len(vals):
                        mean, _, _, el, eu = self.get_bootstrap_ci(vals)
                        plot_data.append(
                            {
                                "System": self._clean_name(sys),
                                "Mean": mean,
                                "Error": [el, eu],
                            }
                        )
                pdf = pd.DataFrame(plot_data)
                if not pdf.empty:
                    x = np.arange(len(pdf))
                    yerr = np.array(pdf["Error"].tolist()).T
                    ax.bar(
                        x,
                        pdf["Mean"],
                        yerr=yerr,
                        capsize=5,
                        color=colors[: len(pdf)],
                        edgecolor="black",
                        alpha=0.9,
                    )
                    ax.set_xticks(x)
                    ax.set_xticklabels(
                        pdf["System"], rotation=45, ha="right", fontsize=9
                    )

            elif plot_type == "line":
                for sys, color in zip(system_types, colors):
                    line_rows = []
                    for ratio in sorted(self.df["distractor_ratio"].unique()):
                        sub = self.df[
                            (self.df["system_type"] == sys)
                            & (self.df["distractor_ratio"] == ratio)
                        ]
                        if len(sub):
                            m, lo, hi, *_ = self.get_bootstrap_ci(
                                sub[metric].values
                            )
                            line_rows.append(
                                {
                                    "Ratio": ratio,
                                    "Mean": m,
                                    "Lower": lo,
                                    "Upper": hi,
                                }
                            )
                    if line_rows:
                        ldf = pd.DataFrame(line_rows).sort_values("Ratio")
                        clean = self._clean_name(sys)
                        marker = self._MARKERS.get(sys, "o")
                        ax.plot(
                            ldf["Ratio"],
                            ldf["Mean"],
                            marker=marker,
                            label=clean,
                            color=color,
                            linewidth=2.5,
                            markersize=8,
                        )
                        ax.fill_between(
                            ldf["Ratio"],
                            ldf["Lower"],
                            ldf["Upper"],
                            color=color,
                            alpha=0.15,
                        )
                ax.set_xlabel("Distractor Ratio")
                ax.legend(
                    title="Architecture",
                    loc="upper right",
                    fontsize=8,
                    framealpha=0.9,
                )

            ax.set_title(
                self._METRIC_LABELS.get(metric, metric), fontweight="bold"
            )
            ax.margins(y=0.15)

        plt.tight_layout()
        fname = f"metrics_{plot_type}_grid.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"[AdvancedMetrics] Saved '{fname}'.")
        plt.close(fig)

    # ------------------------------------------------------------------ #
    # Entry point                                                          #
    # ------------------------------------------------------------------ #

    def generate_report(self) -> None:
        self.generate_stats_table()
        self.calculate_generative_gap()
        print("\n[AdvancedMetrics] Generating plots …")
        self._make_plot_grid("bar")
        self._make_plot_grid("line")
