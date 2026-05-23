"""
visualization/generate_all_charts.py
--------------------------------------
Generates all publication-quality charts for the Auto-KGR ablation study.
Reads the flat metrics CSV and produces 27+ charts.

Usage:
    python -m dft_research_studio.visualization.generate_all_charts \
        --input results/full/full_experiment_metrics.csv \
        --output results/full/charts
"""
from __future__ import annotations
import argparse, json, os, re, sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────
ARCH_ORDER = [
    "Baseline", "TemplatePrompting", "CoTPrompting",
    "BM25Retriever", "CrossEncoderReranker", "BM25RerankerRAG", "StandardRAG",
    "GraphRAG", "GraphDeterministic", "MultiAgent", "MultiAgentBM25",
]

ARCH_GROUPS = {
    "Baselines\n(no retrieval)": ["Baseline", "TemplatePrompting", "CoTPrompting"],
    "IR\n(dense/sparse)": ["BM25Retriever", "CrossEncoderReranker", "BM25RerankerRAG", "StandardRAG"],
    "Graph\n(KG-based)": ["GraphRAG", "GraphDeterministic", "MultiAgent", "MultiAgentBM25"],
}

TIER_NAMES = {1: "T1: Factual", 2: "T2: Constraint", 3: "T3: MultiHop", 4: "T4: Comparative"}

NORM_RANGES = {
    "correctness": (1, 5), "relevance": (1, 5), "groundedness": (0, 1),
    "rougeL_fmeasure": (0, 0.5), "gold_retrieval_recall_at_k": (0, 1),
}


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
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
    df["q_base"] = df["question_id"].str.extract(r"(Q\d+\.\d+)")
    df["q_var"] = df["question_id"].str.extract(r"(v\d+)")
    df["word_count"] = df["generated_answer"].apply(lambda x: len(str(x).split()))
    return df


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────
# Chart generators
# ────────────────────────────────────────────────────────────────────

# 1. Normalised radar chart
def chart_radar_normalized(df: pd.DataFrame, out: str) -> None:
    metrics = ["correctness", "relevance", "groundedness", "rougeL_fmeasure", "gold_retrieval_recall_at_k"]
    labels = ["Correctness\n(norm)", "Relevance\n(norm)", "Groundedness", "ROUGE-L", "Recall@K"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist() + [0]
    colors = plt.cm.tab10(np.linspace(0, 1, 11))
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    for i, arch in enumerate(ARCH_ORDER):
        ad = df[df["experiment_type"] == arch]
        values = []
        for m in metrics:
            v = ad[m].mean()
            lo, hi = NORM_RANGES[m]
            values.append((v - lo) / (hi - lo))
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=1.5, label=arch, color=colors[i], markersize=4)
        ax.fill(angles, values, alpha=0.05, color=colors[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Architecture Capability Radar (Normalised 0-1)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    _save(fig, os.path.join(out, "01_radar_normalized.png"))
    print("  01. Radar normalised")


# 2. Correctness heatmap
def chart_correctness_heatmap(df: pd.DataFrame, out: str) -> None:
    pivot = df.pivot_table(values="correctness", index="experiment_type", columns="distractor_ratio", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot.reindex(ARCH_ORDER), annot=True, fmt=".2f", cmap="RdYlGn", vmin=1, vmax=4, ax=ax, linewidths=0.5)
    ax.set_title("Mean Correctness by Architecture × Distractor Ratio", fontsize=14, fontweight="bold")
    _save(fig, os.path.join(out, "02_heatmap_correctness.png"))
    print("  02. Heatmap correctness")


# 3. Groundedness heatmap
def chart_groundedness_heatmap(df: pd.DataFrame, out: str) -> None:
    pivot = df.pivot_table(values="groundedness", index="experiment_type", columns="distractor_ratio", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot.reindex(ARCH_ORDER), annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Mean Groundedness by Architecture × Distractor Ratio", fontsize=14, fontweight="bold")
    _save(fig, os.path.join(out, "03_heatmap_groundedness.png"))
    print("  03. Heatmap groundedness")


# 4. Relevance heatmap
def chart_relevance_heatmap(df: pd.DataFrame, out: str) -> None:
    pivot = df.pivot_table(values="relevance", index="experiment_type", columns="distractor_ratio", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot.reindex(ARCH_ORDER), annot=True, fmt=".2f", cmap="RdYlGn", vmin=1, vmax=5, ax=ax, linewidths=0.5)
    ax.set_title("Mean Relevance by Architecture × Distractor Ratio", fontsize=14, fontweight="bold")
    _save(fig, os.path.join(out, "04_heatmap_relevance.png"))
    print("  04. Heatmap relevance")


# 5. Dual-axis correctness vs groundedness
def chart_dual_axis(df: pd.DataFrame, out: str) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 7))
    x = np.arange(len(ARCH_ORDER))
    width = 0.35
    corr = [df[df["experiment_type"] == a]["correctness"].mean() for a in ARCH_ORDER]
    grnd = [df[df["experiment_type"] == a]["groundedness"].mean() for a in ARCH_ORDER]
    ax1.bar(x - width / 2, corr, width, label="Correctness (1-5)", color="#3498db", alpha=0.8)
    ax1.set_ylabel("Correctness (1-5)", fontsize=12, color="#3498db"); ax1.set_ylim(0, 5)
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, grnd, width, label="Groundedness (0-1)", color="#e74c3c", alpha=0.8)
    ax2.set_ylabel("Groundedness (0-1)", fontsize=12, color="#e74c3c"); ax2.set_ylim(0, 1.2)
    ax1.set_xticks(x); ax1.set_xticklabels(ARCH_ORDER, rotation=45, ha="right", fontsize=9)
    ax1.set_title("The Correctness–Groundedness Tradeoff", fontsize=14, fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=10)
    _save(fig, os.path.join(out, "05_dual_axis_tradeoff.png"))
    print("  05. Dual axis tradeoff")


# 6. Correctness vs Groundedness scatter
def chart_scatter_tradeoff(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for arch in df["experiment_type"].unique():
        ad = df[df["experiment_type"] == arch]
        ax.scatter(ad["correctness"].mean(), ad["groundedness"].mean(), s=200, label=arch, zorder=5)
        ax.annotate(arch, (ad["correctness"].mean(), ad["groundedness"].mean()),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 8), textcoords="offset points")
    ax.set_xlabel("Mean Correctness (1-5)"); ax.set_ylabel("Mean Groundedness (0-1)")
    ax.set_title("Correctness vs Groundedness Tradeoff", fontsize=14, fontweight="bold")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    _save(fig, os.path.join(out, "06_correctness_vs_groundedness.png"))
    print("  06. Scatter tradeoff")


# 7. Bootstrap CI
def chart_bootstrap_ci(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ci_data = []
    for arch in ARCH_ORDER:
        vals = df[df["experiment_type"] == arch]["correctness"].values
        vals = vals[vals >= 0]
        boots = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(1000)]
        ci_data.append((arch, np.mean(vals), np.percentile(boots, 2.5), np.percentile(boots, 97.5)))
    archs_ci, means, los, his = zip(*ci_data)
    colors = ["#e74c3c" if m < 2.0 else "#f39c12" if m < 2.5 else "#2ecc71" for m in means]
    ax.barh(range(len(archs_ci)), means,
            xerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
            color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(archs_ci))); ax.set_yticklabels(archs_ci, fontsize=10)
    ax.set_xlabel("Mean Correctness (95% Bootstrap CI)")
    ax.set_title("Correctness with 95% Confidence Intervals", fontsize=14, fontweight="bold")
    for i, (m, lo, hi) in enumerate(zip(means, los, his)):
        ax.text(hi + 0.03, i, f"{m:.2f} [{lo:.2f}, {hi:.2f}]", va="center", fontsize=8)
    _save(fig, os.path.join(out, "07_bootstrap_ci.png"))
    print("  07. Bootstrap CI")


# 8. Cohen's d effect sizes
def chart_cohens_d(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    base = df[df["experiment_type"] == "Baseline"]["correctness"].values
    ds = []
    for arch in ARCH_ORDER:
        if arch == "Baseline":
            ds.append(0); continue
        av = df[df["experiment_type"] == arch]["correctness"].values
        ps = np.sqrt(((len(base)-1)*np.var(base,ddof=1)+(len(av)-1)*np.var(av,ddof=1))/(len(base)+len(av)-2))
        ds.append((np.mean(av)-np.mean(base))/ps if ps > 0 else 0)
    colors = ["#2ecc71" if d > 0.2 else "#3498db" if d > -0.2 else "#f39c12" if d > -0.5 else "#e74c3c" for d in ds]
    ax.barh(range(len(ARCH_ORDER)), ds, color=colors)
    ax.set_yticks(range(len(ARCH_ORDER))); ax.set_yticklabels(ARCH_ORDER, fontsize=10)
    ax.set_xlabel("Cohen's d (vs Baseline)")
    ax.set_title("Effect Size: Each Architecture vs Zero-Shot Baseline", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=1)
    for x_val, label, col in [(0.2,"Small +","green"),(-0.2,"Small -","red"),(0.5,"Med +","green"),(-0.5,"Med -","red")]:
        ax.axvline(x=x_val, color=col, linestyle="--" if abs(x_val)==0.2 else ":", alpha=0.3)
    for i, d in enumerate(ds):
        ax.text(d+(0.02 if d >= 0 else -0.08), i, f"{d:+.2f}", va="center", fontsize=9)
    _save(fig, os.path.join(out, "08_cohens_d.png"))
    print("  08. Cohen's d")


# 9. Retrieval metrics (Recall, Precision, MRR)
def chart_retrieval_metrics(df: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for idx, (metric, title) in enumerate([
        ("gold_retrieval_recall_at_k", "Recall@K"),
        ("gold_retrieval_precision_at_k", "Precision@K"),
        ("gold_retrieval_mrr", "MRR"),
    ]):
        means = df.groupby("experiment_type")[metric].mean().reindex(ARCH_ORDER)
        colors = ["#e74c3c" if v == 0 else "#2ecc71" if v > 0.5 else "#f39c12" for v in means.values]
        axes[idx].barh(range(len(means)), means.values, color=colors)
        axes[idx].set_yticks(range(len(means))); axes[idx].set_yticklabels(means.index, fontsize=9)
        axes[idx].set_xlabel(title); axes[idx].set_title(title, fontsize=13, fontweight="bold"); axes[idx].set_xlim(0, 1)
        for j, v in enumerate(means.values):
            axes[idx].text(v + 0.02, j, f"{v:.2f}", va="center", fontsize=9)
    fig.suptitle("Information Retrieval Metrics by Architecture", fontsize=15, fontweight="bold")
    _save(fig, os.path.join(out, "09_retrieval_metrics.png"))
    print("  09. Retrieval metrics")


# 10. Inter-rater agreement
def chart_inter_rater(df: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    c1, c2 = df["correctness"].values, df["correctness_secondary"].values
    mask = (c1 >= 0) & (c2 >= 0); c1v, c2v = c1[mask], c2[mask]
    axes[0].scatter(c1v, c2v, alpha=0.1, s=10, c="#3498db")
    axes[0].plot([0,5],[0,5],"r--",alpha=0.5)
    r_val = np.corrcoef(c1v, c2v)[0, 1]
    from numpy.polynomial.polynomial import polyfit
    b, m = polyfit(c1v, c2v, 1)
    axes[0].plot(np.linspace(1,5,100), b+m*np.linspace(1,5,100), "g-", linewidth=2, label=f"r={r_val:.3f}")
    axes[0].set_xlabel("Primary (qwen2.5:14b)"); axes[0].set_ylabel("Secondary (gemma2:9b)")
    axes[0].set_title("Inter-Rater Agreement", fontsize=13, fontweight="bold"); axes[0].legend()
    # Per-arch agreement
    arch_r = []
    for arch in ARCH_ORDER:
        ad = df[df["experiment_type"]==arch]
        a1, a2 = ad["correctness"].values, ad["correctness_secondary"].values
        m2 = (a1>=0)&(a2>=0)
        arch_r.append(np.corrcoef(a1[m2],a2[m2])[0,1] if m2.sum()>5 else 0)
    colors = ["#2ecc71" if r>0.7 else "#f39c12" if r>0.4 else "#e74c3c" for r in arch_r]
    axes[1].barh(range(11), arch_r, color=colors)
    axes[1].set_yticks(range(11)); axes[1].set_yticklabels(ARCH_ORDER, fontsize=9)
    axes[1].set_xlabel("Pearson r"); axes[1].set_title("Agreement by Architecture", fontsize=13, fontweight="bold")
    axes[1].axvline(x=0.7, color="gray", linestyle="--", alpha=0.5)
    # Confusion
    bins = [0.5,1.5,2.5,3.5,4.5,5.5]
    conf = np.zeros((5,5))
    for a,b_val in zip(np.digitize(c1v,bins), np.digitize(c2v,bins)):
        if 1<=a<=5 and 1<=b_val<=5: conf[a-1][b_val-1] += 1
    conf = conf/conf.sum()*100
    sns.heatmap(conf, annot=True, fmt=".1f", cmap="Blues", ax=axes[2],
                xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
    axes[2].set_xlabel("Secondary"); axes[2].set_ylabel("Primary")
    axes[2].set_title("Score Distribution (%)", fontsize=13, fontweight="bold")
    fig.suptitle("Dual-Judge Inter-Rater Reliability Analysis", fontsize=15, fontweight="bold")
    _save(fig, os.path.join(out, "10_inter_rater_agreement.png"))
    print("  10. Inter-rater agreement")


# 11. Stats table
def chart_stats_table(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 9)); ax.axis("off")
    base = df[df["experiment_type"]=="Baseline"]["correctness"].values
    rows = []
    for arch in ARCH_ORDER:
        ad = df[df["experiment_type"]==arch]
        av = ad["correctness"].values
        ps = np.sqrt(((len(base)-1)*np.var(base,ddof=1)+(len(av)-1)*np.var(av,ddof=1))/(len(base)+len(av)-2))
        cd = (np.mean(av)-np.mean(base))/ps if ps>0 else 0
        rows.append([arch, len(ad), f"{ad['correctness'].mean():.2f}±{ad['correctness'].std():.2f}",
                     f"{ad['correctness_secondary'].mean():.2f}", f"{ad['relevance'].mean():.2f}",
                     f"{ad['groundedness'].mean():.2f}", f"{ad['rougeL_fmeasure'].mean():.3f}",
                     f"{ad['gold_retrieval_recall_at_k'].mean():.2f}", f"{ad['gold_retrieval_mrr'].mean():.2f}", f"{cd:+.2f}"])
    cols = ["Architecture","N","Correctness\n(μ±σ)","Corr.\n2nd","Relev.","Grnd.","ROUGE-L","Recall","MRR","Cohen's d"]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.6)
    for j in range(len(cols)):
        table[(0,j)].set_facecolor("#2c3e50"); table[(0,j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)+1):
        cv = float(rows[i-1][2].split("±")[0])
        c = "#d5f5e3" if cv>=2.7 else "#fef9e7" if cv>=2.0 else "#fadbd8"
        for j in range(len(cols)): table[(i,j)].set_facecolor(c)
    ax.set_title("Table: Complete Ablation Study Results (N=6,600)", fontsize=14, fontweight="bold", pad=20)
    _save(fig, os.path.join(out, "11_stats_table.png"))
    print("  11. Stats table")


# 12. Tier difficulty heatmap
def chart_tier_difficulty(df: pd.DataFrame, out: str) -> None:
    pivot = df.pivot_table(values="correctness", index="experiment_type", columns="tier_name", aggfunc="mean")
    pivot = pivot.reindex(index=ARCH_ORDER, columns=["T1: Factual","T2: Constraint","T3: MultiHop","T4: Comparative"])
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=1, vmax=4, ax=ax, linewidths=0.5)
    ax.set_title("Correctness by Architecture × Question Tier", fontsize=14, fontweight="bold")
    _save(fig, os.path.join(out, "12_tier_difficulty.png"))
    print("  12. Tier difficulty")


# 13. Question variation consistency
def chart_variation(df: pd.DataFrame, out: str) -> None:
    var_scores = df.groupby(["experiment_type","q_base"])["correctness"].agg(["mean","std"]).reset_index()
    var_scores.columns = ["experiment_type","q_base","mean_corr","std_corr"]
    var_scores["std_corr"] = var_scores["std_corr"].fillna(0)
    consistency = var_scores.groupby("experiment_type")["std_corr"].mean().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = ["#2ecc71" if v<0.3 else "#f39c12" if v<0.5 else "#e74c3c" for v in consistency.values]
    axes[0].barh(range(len(consistency)), consistency.values, color=colors)
    axes[0].set_yticks(range(len(consistency))); axes[0].set_yticklabels(consistency.index)
    axes[0].set_xlabel("Mean Std Dev (lower = more consistent)")
    axes[0].set_title("Paraphrase Robustness (v0/v1/v2)", fontsize=13, fontweight="bold")
    axes[0].axvline(x=0.3, color="gray", linestyle="--", alpha=0.5)
    sample = []
    for arch in df["experiment_type"].unique():
        for var in ["v0","v1","v2"]:
            e = df[(df["experiment_type"]==arch)&(df["q_var"]==var)]
            if len(e)>0: sample.append({"Architecture":arch,"Variation":var,"Correctness":e["correctness"].mean()})
    pv = pd.DataFrame(sample).pivot_table(values="Correctness", index="Architecture", columns="Variation", aggfunc="mean")
    sns.heatmap(pv, annot=True, fmt=".2f", cmap="RdYlGn", vmin=1, vmax=4, ax=axes[1], linewidths=0.5)
    axes[1].set_title("Correctness by Variation", fontsize=13, fontweight="bold")
    _save(fig, os.path.join(out, "13_question_variation.png"))
    print("  13. Question variation")


# 14. Latency comparison
def chart_latency(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    lat = df.groupby("experiment_type")["latency_ms"].agg(["mean","std"]).reindex(ARCH_ORDER)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 11))
    ax.barh(range(11), lat["mean"]/1000, xerr=lat["std"]/1000, color=colors, capsize=3)
    ax.set_yticks(range(11)); ax.set_yticklabels(ARCH_ORDER, fontsize=10)
    ax.set_xlabel("Mean Latency (seconds)")
    ax.set_title("Answer Generation Latency by Architecture", fontsize=14, fontweight="bold")
    for i, v in enumerate(lat["mean"]/1000):
        ax.text(v+0.1, i, f"{v:.1f}s", va="center", fontsize=9)
    _save(fig, os.path.join(out, "14_latency_comparison.png"))
    print("  14. Latency")


# 15. Answer length boxplot
def chart_answer_length(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    data = [df[df["experiment_type"]==a]["word_count"].values for a in ARCH_ORDER]
    bp = ax.boxplot(data, labels=ARCH_ORDER, patch_artist=True, vert=True)
    for patch, c in zip(bp["boxes"], plt.cm.Set3(np.linspace(0,1,11))):
        patch.set_facecolor(c)
    ax.set_xticklabels(ARCH_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Answer Word Count"); ax.set_title("Answer Length Distribution", fontsize=14, fontweight="bold")
    _save(fig, os.path.join(out, "15_answer_length_boxplot.png"))
    print("  15. Answer length")


# 16. Correctness violin plot
def chart_violin(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    pd_data = pd.DataFrame([{"Architecture":d["experiment_type"],"Correctness":d["correctness"]}
                             for _, d in df.iterrows()])
    sns.violinplot(data=pd_data, x="Architecture", y="Correctness", ax=ax, palette="Set2", cut=0, order=ARCH_ORDER)
    ax.set_xticklabels(ARCH_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_title("Correctness Score Distribution (Violin)", fontsize=14, fontweight="bold")
    _save(fig, os.path.join(out, "16_correctness_violin.png"))
    print("  16. Violin")


# 17. Correctness CDF
def chart_cdf(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for arch in ARCH_ORDER:
        vals = sorted(df[df["experiment_type"]==arch]["correctness"].values)
        ax.plot(vals, np.arange(1,len(vals)+1)/len(vals), label=arch, linewidth=1.5)
    ax.set_xlabel("Correctness Score"); ax.set_ylabel("Cumulative Proportion")
    ax.set_title("Cumulative Distribution of Correctness", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    _save(fig, os.path.join(out, "17_correctness_cdf.png"))
    print("  17. CDF")


# 18. ROUGE-L comparison
def chart_rouge(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    means = df.groupby("experiment_type")["rougeL_fmeasure"].mean().reindex(ARCH_ORDER)
    stds = df.groupby("experiment_type")["rougeL_fmeasure"].std().reindex(ARCH_ORDER)
    ax.barh(range(11), means.values, xerr=stds.values, color=plt.cm.Greens(np.linspace(0.3,0.9,11)), capsize=3)
    ax.set_yticks(range(11)); ax.set_yticklabels(ARCH_ORDER, fontsize=10)
    ax.set_xlabel("ROUGE-L F-measure"); ax.set_title("ROUGE-L Scores", fontsize=14, fontweight="bold")
    for i, v in enumerate(means.values):
        ax.text(v+0.005, i, f"{v:.3f}", va="center", fontsize=9)
    _save(fig, os.path.join(out, "18_rouge_l.png"))
    print("  18. ROUGE-L")


# 19. Metric correlation matrix
def chart_correlation(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    cols = ["correctness","correctness_secondary","relevance","groundedness","rouge1_fmeasure","rougeL_fmeasure","gold_retrieval_recall_at_k","gold_retrieval_mrr"]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Metric Correlation Matrix", fontsize=14, fontweight="bold")
    _save(fig, os.path.join(out, "19_metric_correlation.png"))
    print("  19. Correlation matrix")


# 20. Error type pies
def chart_error_pies(df: pd.DataFrame, out: str) -> None:
    def classify(row):
        c, g, w = row.get("correctness",0), row.get("groundedness",0), row.get("word_count",0)
        if w < 5: return "Type I: Retrieval Collapse"
        if c <= 1.5 and g > 0.5: return "Type III: Over-Constraint"
        if c <= 2: return "Type II: Hallucination"
        return "Correct"
    df["error_type"] = df.apply(classify, axis=1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    cols_pie = {"Correct":"#2ecc71","Type I: Retrieval Collapse":"#e74c3c","Type II: Hallucination":"#f39c12","Type III: Over-Constraint":"#9b59b6"}
    for idx, arch in enumerate(["Baseline","StandardRAG","GraphRAG","MultiAgent","BM25Retriever","CrossEncoderReranker"]):
        ax = axes[idx//3][idx%3]
        counts = df[df["experiment_type"]==arch]["error_type"].value_counts()
        ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=90,
               colors=[cols_pie.get(c,"#95a5a6") for c in counts.index], textprops={"fontsize":8})
        ax.set_title(arch, fontsize=12, fontweight="bold")
    fig.suptitle("Error Type Distribution (Type I/II/III)", fontsize=15, fontweight="bold")
    _save(fig, os.path.join(out, "20_error_type_pies.png"))
    print("  20. Error type pies")


# 21. Paradigm comparison
def chart_paradigm(df: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, metric in enumerate(["correctness","groundedness","rougeL_fmeasure"]):
        means = [df[df["experiment_type"].isin(a)][metric].mean() for a in ARCH_GROUPS.values()]
        stds = [df[df["experiment_type"].isin(a)][metric].std() for a in ARCH_GROUPS.values()]
        axes[idx].bar(range(3), means, yerr=stds, color=["#95a5a6","#3498db","#e74c3c"], capsize=5, edgecolor="black", linewidth=0.5)
        axes[idx].set_xticks(range(3)); axes[idx].set_xticklabels(ARCH_GROUPS.keys(), fontsize=11)
        axes[idx].set_title(metric.replace("_"," ").title(), fontsize=13, fontweight="bold")
    fig.suptitle("Paradigm Comparison: Baselines vs IR vs Graph", fontsize=15, fontweight="bold")
    _save(fig, os.path.join(out, "21_paradigm_comparison.png"))
    print("  21. Paradigm comparison")


# 22. Question difficulty scatter
def chart_q_difficulty(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    qs = df.groupby("question_id")["correctness"].agg(["mean","std"]).reset_index()
    qs["tier"] = qs["question_id"].str.extract(r"Q(\d+)\.").astype(int)
    for t in [1,2,3,4]:
        td = qs[qs["tier"]==t]
        ax.scatter(td["mean"], td["std"], label=TIER_NAMES[t], s=60, alpha=0.7)
    ax.set_xlabel("Mean Correctness (easier →)"); ax.set_ylabel("Std Dev (more variable →)")
    ax.set_title("Question Difficulty Map", fontsize=14, fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, os.path.join(out, "22_question_difficulty.png"))
    print("  22. Question difficulty")


# 23. Groundedness by tier
def chart_grnd_tier(df: pd.DataFrame, out: str) -> None:
    pivot = df.pivot_table(values="groundedness", index="experiment_type", columns="tier_name", aggfunc="mean")
    pivot = pivot.reindex(index=ARCH_ORDER, columns=["T1: Factual","T2: Constraint","T3: MultiHop","T4: Comparative"])
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Groundedness by Architecture × Tier", fontsize=14, fontweight="bold")
    _save(fig, os.path.join(out, "23_groundedness_by_tier.png"))
    print("  23. Groundedness by tier")


# 24. Groundedness noise sensitivity
def chart_grnd_noise(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    for arch in ARCH_ORDER:
        means = df[df["experiment_type"]==arch].groupby("distractor_ratio")["groundedness"].mean()
        ax.plot(means.index, means.values, "o-", label=arch, linewidth=1.5, markersize=5)
    ax.set_xlabel("Distractor Ratio"); ax.set_ylabel("Mean Groundedness")
    ax.set_title("Groundedness Robustness Profile", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, os.path.join(out, "24_groundedness_noise.png"))
    print("  24. Groundedness noise")


# 25. Hardest questions heatmap
def chart_hardest_qs(df: pd.DataFrame, out: str) -> None:
    q_arch = df.pivot_table(values="correctness", index="question_id", columns="experiment_type", aggfunc="mean")
    q_arch = q_arch.reindex(columns=ARCH_ORDER)
    q_means = q_arch.mean(axis=1).sort_values()
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(q_arch.loc[q_means.index[:20]], annot=True, fmt=".1f", cmap="RdYlGn", vmin=1, vmax=5,
                ax=ax, linewidths=0.3, annot_kws={"fontsize": 7})
    ax.set_title("20 Hardest Questions × Architecture", fontsize=14, fontweight="bold")
    _save(fig, os.path.join(out, "25_hardest_questions.png"))
    print("  25. Hardest questions")


# 26. Variation by tier
def chart_var_tier(df: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, tier in enumerate([1, 2, 3, 4]):
        ax = axes[idx // 2][idx % 2]
        td = df[df["tier"] == tier]
        pv = td.pivot_table(values="correctness", index="experiment_type", columns="q_var", aggfunc="mean").reindex(ARCH_ORDER)
        sns.heatmap(pv, annot=True, fmt=".2f", cmap="RdYlGn", vmin=1, vmax=4, ax=ax, linewidths=0.5)
        ax.set_title(f"T{tier} Variation Scores", fontsize=12, fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    fig.suptitle("Paraphrase Robustness by Tier (v0/v1/v2)", fontsize=15, fontweight="bold")
    _save(fig, os.path.join(out, "26_variation_by_tier.png"))
    print("  26. Variation by tier")


# 27. Composite score (stacked)
def chart_composite(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    metrics_s = ["correctness", "relevance", "groundedness"]
    norm_s = {"correctness": 5, "relevance": 5, "groundedness": 1}
    bottoms = np.zeros(11)
    colors_s = ["#3498db", "#2ecc71", "#e74c3c"]
    for mi, m in enumerate(metrics_s):
        vals = [df[df["experiment_type"] == a][m].mean() / norm_s[m] for a in ARCH_ORDER]
        ax.barh(range(11), vals, left=bottoms, color=colors_s[mi], label=m)
        bottoms += vals
    ax.set_yticks(range(11)); ax.set_yticklabels(ARCH_ORDER, fontsize=10)
    ax.set_xlabel("Normalised Composite Score")
    ax.set_title("Multi-Metric Composite Score", fontsize=14, fontweight="bold"); ax.legend(fontsize=10)
    _save(fig, os.path.join(out, "27_composite_score.png"))
    print("  27. Composite score")


# 28. Key comparison
def chart_key_compare(df: pd.DataFrame, out: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    ca = ["Baseline", "StandardRAG", "GraphRAG", "MultiAgent", "MultiAgentBM25"]
    for idx, (m, t) in enumerate([("correctness","Correctness"),("groundedness","Groundedness"),
                                   ("rougeL_fmeasure","ROUGE-L"),("gold_retrieval_recall_at_k","Recall@K")]):
        vals = [df[df["experiment_type"]==a][m].mean() for a in ca]
        axes[idx].bar(range(5), vals, color=["#95a5a6","#3498db","#e67e22","#e74c3c","#9b59b6"])
        axes[idx].set_xticks(range(5)); axes[idx].set_xticklabels([a[:10] for a in ca], rotation=45, ha="right")
        axes[idx].set_title(t, fontsize=12, fontweight="bold")
        for i, v in enumerate(vals): axes[idx].text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.suptitle("Key Architecture Comparison", fontsize=14, fontweight="bold")
    _save(fig, os.path.join(out, "28_key_comparison.png"))
    print("  28. Key comparison")


# 29. Experiment summary
def chart_summary(df: pd.DataFrame, out: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 8)); ax.axis("off")
    r_val = np.corrcoef(df["correctness"].values, df["correctness_secondary"].values)[0, 1]
    data = [
        ["Total Experiments","6,600"],["Architectures","11"],["Questions","120 (4 tiers × 30)"],
        ["Distractor Ratios","5 (0.0–3.0)"],["KG Nodes","19,726"],["KG Edges","97,924"],
        ["Primary Judge","qwen2.5:14b"],["Secondary Judge","gemma2:9b"],["Generator","llama3.1:8b"],
        ["Inter-Rater r",f"{r_val:.3f}"],
        ["Best Correctness",f"StandardRAG ({df[df['experiment_type']=='StandardRAG']['correctness'].mean():.2f})"],
        ["Best Groundedness",f"GraphRAG ({df[df['experiment_type']=='GraphRAG']['groundedness'].mean():.2f})"],
    ]
    table = ax.table(cellText=data, colLabels=["Metric","Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 1.8)
    for j in range(2):
        table[(0,j)].set_facecolor("#2c3e50"); table[(0,j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(data)+1):
        table[(i,0)].set_facecolor("#eaf2f8"); table[(i,0)].set_text_props(fontweight="bold")
    ax.set_title("Experiment Summary", fontsize=16, fontweight="bold", pad=20)
    _save(fig, os.path.join(out, "29_experiment_summary.png"))
    print("  29. Summary")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────
ALL_CHARTS = [
    chart_radar_normalized, chart_correctness_heatmap, chart_groundedness_heatmap,
    chart_relevance_heatmap, chart_dual_axis, chart_scatter_tradeoff, chart_bootstrap_ci,
    chart_cohens_d, chart_retrieval_metrics, chart_inter_rater, chart_stats_table,
    chart_tier_difficulty, chart_variation, chart_latency, chart_answer_length,
    chart_violin, chart_cdf, chart_rouge, chart_correlation, chart_error_pies,
    chart_paradigm, chart_q_difficulty, chart_grnd_tier, chart_grnd_noise,
    chart_hardest_qs, chart_var_tier, chart_composite, chart_key_compare, chart_summary,
]


def main():
    parser = argparse.ArgumentParser(description="Generate all Auto-KGR charts")
    parser.add_argument("--input", default="results/full/full_experiment_metrics.csv")
    parser.add_argument("--output", default="results/full/charts")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading {args.input} ...")
    df = load_and_flatten(args.input)
    print(f"Loaded {len(df)} rows. Generating {len(ALL_CHARTS)} charts to {args.output}/\n")

    for fn in ALL_CHARTS:
        try:
            fn(df, args.output)
        except Exception as exc:
            print(f"  ❌ {fn.__name__}: {exc}")

    total = len([f for f in os.listdir(args.output) if f.endswith(".png")])
    print(f"\n✅ {total} charts generated in {args.output}/")


if __name__ == "__main__":
    main()

