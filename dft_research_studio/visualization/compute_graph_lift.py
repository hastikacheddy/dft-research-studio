"""
visualization/compute_graph_lift.py
-------------------------------------
Computes the Graph Lift (ΔG) metric for Auto-KGR architectures.

ΔG = (M_Auto-KGR − M_Hybrid) / M_Hybrid

where M represents the metric score (correctness, groundedness, etc.)
on multi-hop reasoning tasks.

Usage:
    python -m dft_research_studio.visualization.compute_graph_lift \
        --input results/full/full_experiment_metrics.csv \
        --output results/full/tables
"""
from __future__ import annotations
import argparse, json, os, re
import numpy as np
import pandas as pd

ARCH_ORDER = [
    "Baseline", "TemplatePrompting", "CoTPrompting",
    "BM25Retriever", "CrossEncoderReranker", "BM25RerankerRAG", "StandardRAG",
    "GraphRAG", "GraphDeterministic", "MultiAgent", "MultiAgentBM25",
]
GRAPH_ARCHS = ["GraphRAG", "GraphDeterministic", "MultiAgent", "MultiAgentBM25"]
BASELINE_ARCHS = ["Baseline", "TemplatePrompting", "CoTPrompting"]
IR_ARCHS = ["BM25Retriever", "CrossEncoderReranker", "BM25RerankerRAG", "StandardRAG"]
TIER_NAMES = {1: "T1: Factual", 2: "T2: Constraint", 3: "T3: MultiHop", 4: "T4: Comparative"}
METRICS = ["correctness", "groundedness", "relevance", "rougeL_fmeasure", "gold_retrieval_recall_at_k"]
REFERENCE = "BM25RerankerRAG"


def parse_metrics(s: str) -> dict:
    if not s:
        return {}
    try:
        s = re.sub(r"\bnan\b", "null", s)
        s = s.replace("'", '"').replace("None", "null").replace("True", "true").replace("False", "false")
        return json.loads(s)
    except Exception:
        return {}


def load_and_flatten(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "metrics" in df.columns:
        mdf = df["metrics"].apply(parse_metrics).apply(pd.Series)
        df = pd.concat([df.drop(columns=["metrics"]), mdf], axis=1)
    df["tier"] = df["question_id"].str.extract(r"Q(\d+)\.").astype(int)
    df["tier_name"] = df["tier"].map(TIER_NAMES)
    return df


def compute_lift(auto_kgr_score: float, hybrid_score: float) -> float:
    """ΔG = (M_Auto-KGR − M_Hybrid) / M_Hybrid × 100"""
    if hybrid_score == 0:
        return 0.0
    return (auto_kgr_score - hybrid_score) / hybrid_score * 100



def generate_charts(df: pd.DataFrame, ref: str, out: str) -> None:
    """Generate all Graph Lift charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    charts_dir = os.path.join(out, "..", "charts")
    os.makedirs(charts_dir, exist_ok=True)

    def _save(fig, name):
        path = os.path.join(charts_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Chart: {name}")

    # ─── 1. Dual bar: correctness vs groundedness lift ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for idx, (metric, title) in enumerate([("correctness", "Correctness"), ("groundedness", "Groundedness")]):
        lifts = []
        for arch in GRAPH_ARCHS:
            ma = df[df["experiment_type"] == arch][metric].mean()
            sr = df[df["experiment_type"] == ref][metric].mean()
            lifts.append(compute_lift(ma, sr) / 100)  # decimal form
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in lifts]
        axes[idx].barh(range(4), lifts, color=colors, edgecolor="black", linewidth=0.5)
        axes[idx].set_yticks(range(4))
        axes[idx].set_yticklabels(GRAPH_ARCHS, fontsize=11)
        axes[idx].set_xlabel("ΔG (Graph Lift)", fontsize=12)
        axes[idx].set_title(f"Graph Lift: {title}", fontsize=14, fontweight="bold")
        axes[idx].axvline(x=0, color="black", linewidth=1)
        for i, v in enumerate(lifts):
            axes[idx].text(v + (0.01 if v >= 0 else -0.06), i, f"{v:+.3f}", va="center", fontsize=11, fontweight="bold")
    fig.suptitle(f"ΔG = (M_Auto-KGR − M_Hybrid) / M_Hybrid    |    Reference: {ref}",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "graph_lift_dual.png")

    # ─── 2. Per-tier lift (MultiAgent) ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for idx, (metric, title) in enumerate([("correctness", "Correctness"), ("groundedness", "Groundedness")]):
        tier_lifts = []
        for t in [1, 2, 3, 4]:
            ma = df[(df["experiment_type"] == "MultiAgent") & (df["tier"] == t)][metric].mean()
            sr = df[(df["experiment_type"] == ref) & (df["tier"] == t)][metric].mean()
            tier_lifts.append(compute_lift(ma, sr) / 100)
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in tier_lifts]
        tier_labels = [TIER_NAMES[t] for t in [1, 2, 3, 4]]
        axes[idx].barh(range(4), tier_lifts, color=colors, edgecolor="black", linewidth=0.5)
        axes[idx].set_yticks(range(4))
        axes[idx].set_yticklabels(tier_labels, fontsize=11)
        axes[idx].set_xlabel("ΔG", fontsize=12)
        axes[idx].set_title(f"MultiAgent {title} Lift by Tier", fontsize=13, fontweight="bold")
        axes[idx].axvline(x=0, color="black", linewidth=1)
        for i, v in enumerate(tier_lifts):
            axes[idx].text(v + (0.01 if v >= 0 else -0.06), i, f"{v:+.3f}", va="center", fontsize=11, fontweight="bold")
    fig.suptitle(f"Graph Lift by Question Tier (MultiAgent vs {ref})", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "graph_lift_by_tier.png")

    # ─── 3. Heatmap: all graph archs × all metrics ───
    fig, ax = plt.subplots(figsize=(14, 8))
    metric_labels = ["Correctness", "Groundedness", "Relevance", "ROUGE-L", "Recall@K"]
    lift_matrix = []
    for arch in GRAPH_ARCHS:
        row = []
        for m in METRICS:
            ma = df[df["experiment_type"] == arch][m].mean()
            sr = df[df["experiment_type"] == ref][m].mean()
            row.append(compute_lift(ma, sr) / 100)
        lift_matrix.append(row)
    lift_df = pd.DataFrame(lift_matrix, index=GRAPH_ARCHS, columns=metric_labels)
    sns.heatmap(lift_df, annot=True, fmt="+.3f", cmap="RdYlGn", center=0, ax=ax,
                linewidths=0.5, annot_kws={"fontsize": 12, "fontweight": "bold"}, vmin=-0.5, vmax=0.5)
    ax.set_title(f"Graph Lift ΔG Heatmap: All Graph Architectures × All Metrics\n(Reference: {ref})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "graph_lift_heatmap.png")

    # ─── 4. Tradeoff scatter: correctness cost vs groundedness gain ───
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_archs = GRAPH_ARCHS + [ref, "Baseline"]
    for arch in plot_archs:
        ma_c = df[df["experiment_type"] == arch]["correctness"].mean()
        sr_c = df[df["experiment_type"] == ref]["correctness"].mean()
        ma_g = df[df["experiment_type"] == arch]["groundedness"].mean()
        sr_g = df[df["experiment_type"] == ref]["groundedness"].mean()
        dg_c = compute_lift(ma_c, sr_c) / 100
        dg_g = compute_lift(ma_g, sr_g) / 100
        color = "#e74c3c" if arch in GRAPH_ARCHS else "#3498db" if arch == ref else "#95a5a6"
        marker = "s" if arch in GRAPH_ARCHS else "D" if arch == ref else "o"
        size = 300 if arch in ["MultiAgent", ref] else 200
        ax.scatter(dg_c, dg_g, s=size, c=color, marker=marker, zorder=5, edgecolors="black", linewidth=1)
        ax.annotate(arch, (dg_c, dg_g), fontsize=9, ha="center", va="bottom",
                    xytext=(0, 10), textcoords="offset points", fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("ΔG Correctness (cost →)", fontsize=13)
    ax.set_ylabel("ΔG Groundedness (gain ↑)", fontsize=13)
    ax.set_title("The Correctness–Groundedness Tradeoff\n(Graph Lift relative to " + ref + ")", fontsize=14, fontweight="bold")
    ax.text(0.15, 0.25, "IDEAL\n(higher on both)", fontsize=10, color="green", alpha=0.5, ha="center", style="italic")
    ax.text(-0.3, 0.25, "GROUNDED\nbut less correct", fontsize=10, color="#e67e22", alpha=0.5, ha="center", style="italic")
    ax.text(0.15, -0.15, "MORE CORRECT\nbut less grounded", fontsize=10, color="#3498db", alpha=0.5, ha="center", style="italic")
    ax.text(-0.3, -0.15, "WORSE\n(lower on both)", fontsize=10, color="red", alpha=0.5, ha="center", style="italic")
    fig.tight_layout()
    _save(fig, "graph_lift_tradeoff.png")

    # ─── 5. Line chart: ΔG by distractor ratio ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ratios = [0.0, 0.5, 1.0, 2.0, 3.0]
    for arch in GRAPH_ARCHS:
        corr_r, grnd_r = [], []
        for r in ratios:
            ma_c = df[(df["experiment_type"] == arch) & (df["distractor_ratio"] == r)]["correctness"].mean()
            sr_c = df[(df["experiment_type"] == ref) & (df["distractor_ratio"] == r)]["correctness"].mean()
            corr_r.append(compute_lift(ma_c, sr_c) / 100)
            ma_g = df[(df["experiment_type"] == arch) & (df["distractor_ratio"] == r)]["groundedness"].mean()
            sr_g = df[(df["experiment_type"] == ref) & (df["distractor_ratio"] == r)]["groundedness"].mean()
            grnd_r.append(compute_lift(ma_g, sr_g) / 100)
        axes[0].plot(ratios, corr_r, "o-", label=arch, linewidth=2, markersize=7)
        axes[1].plot(ratios, grnd_r, "o-", label=arch, linewidth=2, markersize=7)
    for idx, (title, ylabel) in enumerate([("Correctness Graph Lift vs Noise", "ΔG Correctness"),
                                            ("Groundedness Graph Lift vs Noise", "ΔG Groundedness")]):
        axes[idx].axhline(y=0, color="black", linewidth=1, linestyle="--")
        axes[idx].set_xlabel("Distractor Ratio", fontsize=12)
        axes[idx].set_ylabel(ylabel, fontsize=12)
        axes[idx].set_title(title, fontsize=13, fontweight="bold")
        axes[idx].legend(fontsize=10)
        axes[idx].grid(alpha=0.3)
    fig.suptitle(f"Graph Lift Stability Across Distractor Ratios (Reference: {ref})", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "graph_lift_by_ratio.png")


def main():
    parser = argparse.ArgumentParser(description="Compute Graph Lift (ΔG)")
    parser.add_argument("--input", default="results/full/full_experiment_metrics.csv")
    parser.add_argument("--output", default="results/full/tables")
    parser.add_argument("--reference", default=REFERENCE, help="Reference architecture (default: BM25RerankerRAG)")
    parser.add_argument("--charts", action="store_true", help="Also generate charts")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading {args.input} ...")
    df = load_and_flatten(args.input)
    ref = args.reference
    print(f"Loaded {len(df)} rows. Reference: {ref}\n")

    # Print tables
    print("=" * 90)
    print("1. OVERALL GRAPH LIFT")
    print("=" * 90)
    print(f"  {'Metric':<25} {'MultiAgent':>12} {'MultiAgentBM25':>15} {ref:>15} {'ΔG (MA)':>10} {'ΔG (MABM)':>10}")
    print(f"  {'-' * 88}")
    for metric in METRICS:
        ma = df[df["experiment_type"] == "MultiAgent"][metric].mean()
        mabm = df[df["experiment_type"] == "MultiAgentBM25"][metric].mean()
        sr = df[df["experiment_type"] == ref][metric].mean()
        dg_ma = compute_lift(ma, sr) / 100
        dg_mabm = compute_lift(mabm, sr) / 100
        print(f"  {metric:<25} {ma:>12.3f} {mabm:>15.3f} {sr:>15.3f} {dg_ma:>+9.3f} {dg_mabm:>+9.3f}")

    print(f"\n{'=' * 90}")
    print("2. GRAPH LIFT BY TIER (correctness)")
    print(f"{'=' * 90}")
    print(f"  {'Tier':<20} {'MultiAgent':>12} {ref:>15} {'ΔG':>10} {'Note':>20}")
    print(f"  {'-' * 78}")
    for t in [1, 2, 3, 4]:
        ma = df[(df["experiment_type"] == "MultiAgent") & (df["tier"] == t)]["correctness"].mean()
        sr = df[(df["experiment_type"] == ref) & (df["tier"] == t)]["correctness"].mean()
        dg = compute_lift(ma, sr) / 100
        note = "← KEY METRIC" if t == 3 else ""
        print(f"  {TIER_NAMES[t]:<18} {ma:>12.2f} {sr:>15.2f} {dg:>+9.3f} {note}")

    print(f"\n{'=' * 90}")
    print("3. GRAPH LIFT BY TIER (groundedness)")
    print(f"{'=' * 90}")
    for t in [1, 2, 3, 4]:
        ma = df[(df["experiment_type"] == "MultiAgent") & (df["tier"] == t)]["groundedness"].mean()
        sr = df[(df["experiment_type"] == ref) & (df["tier"] == t)]["groundedness"].mean()
        dg = compute_lift(ma, sr) / 100
        print(f"  {TIER_NAMES[t]:<18} {ma:>12.2f} {sr:>15.2f} {dg:>+9.3f}")

    # Save CSV
    rows = []
    for arch in GRAPH_ARCHS:
        for metric in METRICS:
            for t in [0, 1, 2, 3, 4]:
                if t == 0:
                    ma = df[df["experiment_type"] == arch][metric].mean()
                    sr = df[df["experiment_type"] == ref][metric].mean()
                    tier_label = "Overall"
                else:
                    ma = df[(df["experiment_type"] == arch) & (df["tier"] == t)][metric].mean()
                    sr = df[(df["experiment_type"] == ref) & (df["tier"] == t)][metric].mean()
                    tier_label = TIER_NAMES[t]
                dg = compute_lift(ma, sr) / 100
                rows.append({
                    "Architecture": arch, "Metric": metric, "Tier": tier_label,
                    "Auto-KGR Score": round(ma, 3), f"{ref} Score": round(sr, 3),
                    "Graph Lift (dG)": round(dg, 4),
                })
    out_path = os.path.join(args.output, "table_graph_lift.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n✅ Saved: {out_path} ({len(rows)} rows)")

    if args.charts:
        print(f"\nGenerating charts...")
        generate_charts(df, ref, args.output)
        print("✅ All Graph Lift charts generated.")


if __name__ == "__main__":
    main()
