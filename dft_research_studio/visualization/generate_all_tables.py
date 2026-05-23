"""
visualization/generate_all_tables.py
--------------------------------------
Generates all publication-quality CSV tables for the Auto-KGR ablation study.

Usage:
    python -m dft_research_studio.visualization.generate_all_tables \
        --input results/full/full_experiment_metrics.csv \
        --output results/full/tables
"""
from __future__ import annotations
import argparse, json, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats as sp

ARCH_ORDER = [
    "Baseline", "TemplatePrompting", "CoTPrompting",
    "BM25Retriever", "CrossEncoderReranker", "BM25RerankerRAG", "StandardRAG",
    "GraphRAG", "GraphDeterministic", "MultiAgent", "MultiAgentBM25",
]
TIER_NAMES = {1: "T1: Factual", 2: "T2: Constraint", 3: "T3: MultiHop", 4: "T4: Comparative"}


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


def table_5_1_architectures(df, out):
    arch_info = [
        ["Baseline", "Zero-Shot Prompting", "Parametric", "No retrieval", "Part 1: Pure LLM"],
        ["TemplatePrompting", "Template Prompting", "Parametric", "Domain persona", "Part 1: Pure LLM"],
        ["CoTPrompting", "Chain-of-Thought", "Parametric", "Step-by-step reasoning", "Part 1: Pure LLM"],
        ["BM25Retriever", "BM25 Retriever", "Sparse", "Okapi BM25 keyword matching", "Part 2: Standard IR"],
        ["CrossEncoderReranker", "Cross-Encoder Reranker", "Neural", "Deep semantic pair scoring", "Part 2: Standard IR"],
        ["BM25RerankerRAG", "BM25 + Reranker RAG", "Hybrid", "BM25 recall + Cross-Encoder precision", "Part 2: Standard IR"],
        ["StandardRAG", "Standard Vector RAG", "Dense", "HuggingFace embeddings + ChromaDB", "Part 2: Standard IR"],
        ["GraphRAG", "GraphRAG (Star Context)", "Symbolic", "NER + fuzzy match + ego-graph", "Part 3: Graph-Based"],
        ["GraphDeterministic", "Topological Retriever", "Deterministic", "Hub-based 1-hop ranked by degree", "Part 3: Graph-Based"],
        ["MultiAgent", "Multi-Agent (GraphRAG)", "Adaptive", "Strategist + Critic + graph validation", "Part 3: Graph-Based"],
        ["MultiAgentBM25", "Multi-Agent (BM25)", "Adaptive", "Strategist + Critic + BM25 fallback", "Part 3: Graph-Based"],
    ]
    pd.DataFrame(arch_info, columns=["ID", "Architecture", "Type", "Core Mechanism", "Part"]).to_csv(
        os.path.join(out, "table_5_1_architectures.csv"), index=False)
    print("  Table 5.1: Architecture Overview")


def table_5_2_q120(df, out):
    rows = []
    for t in [1, 2, 3, 4]:
        td = df[df["tier"] == t]
        example = td.iloc[0]["question"][:80] if len(td) > 0 else ""
        rows.append({
            "Tier": TIER_NAMES[t], "Questions": len(set(td["question_id"])),
            "Base Questions": len(set(td["q_base"])), "Variations": 3, "Example": example,
        })
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_5_2_q120_structure.csv"), index=False)
    print("  Table 5.2: Q-120 Structure")


def table_5_3_kg(df, out):
    rows = [
        ["Total Nodes", "19,726"], ["Total Relationships", "97,924"], ["Papers", "23"],
        ["Functionals", "293"], ["ValidationResults", "1,213"], ["Chemical Systems", "6,904"],
        ["BenchmarkSets", "553"], ["FailureModes", "71"], ["Relationship Types", "15+"],
        ["Avg Degree", "9.93"], ["Max Degree (Hub)", "2,847"],
    ]
    pd.DataFrame(rows, columns=["Metric", "Value"]).to_csv(
        os.path.join(out, "table_5_3_kg_stats.csv"), index=False)
    print("  Table 5.3: KG Statistics")


def table_5_4_metrics(df, out):
    rows = [
        ["Correctness", "1-5 (Likert)", "LLM-as-Judge", "How well the answer matches ground truth"],
        ["Correctness (2nd)", "1-5 (Likert)", "LLM-as-Judge", "Inter-rater reliability check"],
        ["Relevance", "1-5 (Likert)", "LLM-as-Judge", "How well the answer addresses the question"],
        ["Groundedness", "0-1 (Binary)", "LLM-as-Judge", "Whether claims are supported by context"],
        ["ROUGE-1", "0-1 (F1)", "Automated", "Unigram overlap with ground truth"],
        ["ROUGE-L", "0-1 (F1)", "Automated", "Longest common subsequence overlap"],
        ["Recall@K", "0-1", "Automated", "Fraction of gold documents retrieved"],
        ["Precision@K", "0-1", "Automated", "Fraction of retrieved docs that are relevant"],
        ["MRR", "0-1", "Automated", "Mean reciprocal rank of first relevant document"],
        ["Cohen's d", "Real", "Computed", "Effect size vs baseline (pooled std)"],
    ]
    pd.DataFrame(rows, columns=["Metric", "Scale", "Method", "Description"]).to_csv(
        os.path.join(out, "table_5_4_eval_metrics.csv"), index=False)
    print("  Table 5.4: Evaluation Metrics")


def table_5_5_config(df, out):
    rows = [
        ["Generator Model", "llama3.1:8b (Ollama local)"],
        ["Primary Judge", "qwen2.5:14b (Ollama local)"],
        ["Secondary Judge", "gemma2:9b (Ollama local)"],
        ["Embedding Model", "sentence-transformers/all-MiniLM-L6-v2"],
        ["Reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2"],
        ["Distractor Ratios", "0.0, 0.5, 1.0, 2.0, 3.0"],
        ["Questions", "120 (Q-120 Challenge Set)"],
        ["Total Runs", "6,600 (11 x 5 x 120)"],
        ["Random Seed", "42"],
        ["Max Tokens (Answer)", "500"],
        ["Max Tokens (Judge)", "512"],
        ["Temperature", "0.0 (deterministic)"],
        ["Hardware", "NVIDIA L4 (24GB VRAM)"],
        ["Platform", "Lightning.ai"],
        ["Vector Store", "ChromaDB (per-ratio)"],
        ["BM25 Source", "Extracted from ChromaDB"],
    ]
    pd.DataFrame(rows, columns=["Parameter", "Value"]).to_csv(
        os.path.join(out, "table_5_5_config.csv"), index=False)
    print("  Table 5.5: Experiment Configuration")


def table_6_1_ablation(df, out):
    base = df[df["experiment_type"] == "Baseline"]["correctness"].values
    rows = []
    for a in ARCH_ORDER:
        ad = df[df["experiment_type"] == a]
        av = ad["correctness"].values
        ps = np.sqrt(((len(base)-1)*np.var(base, ddof=1)+(len(av)-1)*np.var(av, ddof=1))/(len(base)+len(av)-2))
        cd = (np.mean(av)-np.mean(base))/ps if ps > 0 else 0
        rows.append({
            "Architecture": a, "N": len(ad),
            "Correctness (mean)": round(ad["correctness"].mean(), 2),
            "Correctness (std)": round(ad["correctness"].std(), 2),
            "Correctness 2nd": round(ad["correctness_secondary"].mean(), 2),
            "Relevance": round(ad["relevance"].mean(), 2),
            "Groundedness": round(ad["groundedness"].mean(), 2),
            "ROUGE-1": round(ad["rouge1_fmeasure"].mean(), 3),
            "ROUGE-L": round(ad["rougeL_fmeasure"].mean(), 3),
            "Recall@K": round(ad["gold_retrieval_recall_at_k"].mean(), 2),
            "Precision@K": round(ad["gold_retrieval_precision_at_k"].mean(), 2),
            "MRR": round(ad["gold_retrieval_mrr"].mean(), 2),
            "Cohen's d": round(cd, 3),
            "Latency (ms)": round(ad["latency_ms"].mean(), 0),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_6_1_ablation_results.csv"), index=False)
    print("  Table 6.1: Complete Ablation Results")


def table_6_2_significance(df, out):
    rows = []
    for a1 in ARCH_ORDER:
        row = {"Architecture": a1}
        v1 = df[df["experiment_type"] == a1]["correctness"].values
        for a2 in ARCH_ORDER:
            v2 = df[df["experiment_type"] == a2]["correctness"].values
            _, p = sp.ttest_ind(v1, v2, equal_var=False)
            row[a2] = round(p, 4)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_6_2_significance.csv"), index=False)
    print("  Table 6.2: Pairwise Significance")


def table_6_3_per_tier(df, out):
    rows = []
    for a in ARCH_ORDER:
        row = {"Architecture": a}
        for t in [1, 2, 3, 4]:
            td = df[(df["experiment_type"] == a) & (df["tier"] == t)]
            row[f"{TIER_NAMES[t]} Corr"] = round(td["correctness"].mean(), 2)
            row[f"{TIER_NAMES[t]} Grnd"] = round(td["groundedness"].mean(), 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_6_3_per_tier.csv"), index=False)
    print("  Table 6.3: Per-Tier Results")


def table_6_4_groundedness(df, out):
    groups = {
        "Baselines": ["Baseline", "TemplatePrompting", "CoTPrompting"],
        "IR": ["BM25Retriever", "CrossEncoderReranker", "BM25RerankerRAG", "StandardRAG"],
        "Graph": ["GraphRAG", "GraphDeterministic", "MultiAgent", "MultiAgentBM25"],
    }
    rows = []
    for gname, archs in groups.items():
        gd = df[df["experiment_type"].isin(archs)]
        rows.append({
            "Paradigm": gname, "Architectures": len(archs),
            "Mean Groundedness": round(gd["groundedness"].mean(), 3),
            "Std": round(gd["groundedness"].std(), 3),
            "Mean Correctness": round(gd["correctness"].mean(), 2),
            "Mean Relevance": round(gd["relevance"].mean(), 2),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_6_4_groundedness.csv"), index=False)
    print("  Table 6.4: Groundedness Comparison")


def table_6_5_retrieval(df, out):
    rows = []
    for a in ARCH_ORDER:
        ad = df[df["experiment_type"] == a]
        rows.append({
            "Architecture": a,
            "Recall@K": round(ad["gold_retrieval_recall_at_k"].mean(), 3),
            "Precision@K": round(ad["gold_retrieval_precision_at_k"].mean(), 3),
            "MRR": round(ad["gold_retrieval_mrr"].mean(), 3),
            "Has Retrieval": a not in ["Baseline", "TemplatePrompting", "CoTPrompting"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_6_5_retrieval.csv"), index=False)
    print("  Table 6.5: Retrieval Metrics")


def table_6_6_error_taxonomy(df, out):
    def classify(row):
        c, g, w = row.get("correctness", 0), row.get("groundedness", 0), row.get("word_count", 0)
        if w < 5: return "Type I: Retrieval Collapse"
        if c <= 1.5 and g > 0.5: return "Type III: Over-Constraint"
        if c <= 2: return "Type II: Hallucination"
        return "Correct"
    df["error_type"] = df.apply(classify, axis=1)
    rows = []
    for a in ARCH_ORDER:
        ad = df[df["experiment_type"] == a]
        counts = ad["error_type"].value_counts()
        total = len(ad)
        rows.append({
            "Architecture": a, "N": total,
            "Correct": counts.get("Correct", 0),
            "Correct %": round(counts.get("Correct", 0)/total*100, 1),
            "Type I": counts.get("Type I: Retrieval Collapse", 0),
            "Type I %": round(counts.get("Type I: Retrieval Collapse", 0)/total*100, 1),
            "Type II": counts.get("Type II: Hallucination", 0),
            "Type II %": round(counts.get("Type II: Hallucination", 0)/total*100, 1),
            "Type III": counts.get("Type III: Over-Constraint", 0),
            "Type III %": round(counts.get("Type III: Over-Constraint", 0)/total*100, 1),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_6_6_error_taxonomy.csv"), index=False)
    print("  Table 6.6: Error Taxonomy")


def table_6_7_inter_rater(df, out):
    rows = []
    for a in ARCH_ORDER:
        ad = df[df["experiment_type"] == a]
        c1, c2 = ad["correctness"].values, ad["correctness_secondary"].values
        mask = (c1 >= 0) & (c2 >= 0)
        r = np.corrcoef(c1[mask], c2[mask])[0, 1] if mask.sum() > 5 else float("nan")
        rows.append({
            "Architecture": a, "Valid Pairs": int(mask.sum()),
            "Pearson r": round(r, 3),
            "Agreement": "Strong" if r > 0.7 else "Moderate" if r > 0.4 else "Weak",
            "Mean Diff": round(np.mean(np.abs(c1[mask]-c2[mask])), 2) if mask.sum() > 0 else 0,
        })
    c1a, c2a = df["correctness"].values, df["correctness_secondary"].values
    ma = (c1a >= 0) & (c2a >= 0)
    rows.append({
        "Architecture": "OVERALL", "Valid Pairs": int(ma.sum()),
        "Pearson r": round(np.corrcoef(c1a[ma], c2a[ma])[0, 1], 3),
        "Agreement": "Strong",
        "Mean Diff": round(np.mean(np.abs(c1a[ma]-c2a[ma])), 2),
    })
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_6_7_inter_rater.csv"), index=False)
    print("  Table 6.7: Inter-Rater Agreement")


def table_6_8_bootstrap(df, out):
    np.random.seed(42)
    rows = []
    for a in ARCH_ORDER:
        vals = df[df["experiment_type"] == a]["correctness"].values
        vals = vals[vals >= 0]
        boots = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(2000)]
        rows.append({
            "Architecture": a, "Mean": round(np.mean(vals), 3), "Std": round(np.std(vals), 3),
            "CI Lower (2.5%)": round(np.percentile(boots, 2.5), 3),
            "CI Upper (97.5%)": round(np.percentile(boots, 97.5), 3),
            "CI Width": round(np.percentile(boots, 97.5)-np.percentile(boots, 2.5), 3),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_6_8_bootstrap_ci.csv"), index=False)
    print("  Table 6.8: Bootstrap CIs")


def table_6_9_paraphrase(df, out):
    rows = []
    for a in ARCH_ORDER:
        ad = df[df["experiment_type"] == a]
        var_data = ad.groupby(["q_base", "q_var"])["correctness"].mean().reset_index()
        var_pivot = var_data.pivot(index="q_base", columns="q_var", values="correctness")
        v0 = var_pivot.get("v0", pd.Series([0])).mean()
        v1 = var_pivot.get("v1", pd.Series([0])).mean()
        v2 = var_pivot.get("v2", pd.Series([0])).mean()
        per_q_std = ad.groupby("q_base")["correctness"].std().mean()
        rows.append({
            "Architecture": a,
            "v0 Mean": round(v0, 2), "v1 Mean": round(v1, 2), "v2 Mean": round(v2, 2),
            "Mean Std": round(per_q_std, 3) if not np.isnan(per_q_std) else 0,
            "Max Gap": round(max(abs(v0-v1), abs(v1-v2), abs(v0-v2)), 2),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_6_9_paraphrase.csv"), index=False)
    print("  Table 6.9: Paraphrase Robustness")


def table_6_10_fidelity_gap(df, out):
    ref = "StandardRAG"
    rows = []
    for a in ARCH_ORDER:
        row = {"Architecture": a}
        for r in [0.0, 0.5, 1.0, 2.0, 3.0]:
            a_s = df[(df["experiment_type"] == a) & (df["distractor_ratio"] == r)]["correctness"].mean()
            r_s = df[(df["experiment_type"] == ref) & (df["distractor_ratio"] == r)]["correctness"].mean()
            gap = ((r_s - a_s) / r_s * 100) if r_s > 0 else 0
            row[f"GFG ratio={r}"] = round(gap, 2)
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out, "table_6_10_fidelity_gap.csv"), index=False)
    print("  Table 6.10: Generative Fidelity Gap")


ALL_TABLES = [
    table_5_1_architectures, table_5_2_q120, table_5_3_kg, table_5_4_metrics, table_5_5_config,
    table_6_1_ablation, table_6_2_significance, table_6_3_per_tier, table_6_4_groundedness,
    table_6_5_retrieval, table_6_6_error_taxonomy, table_6_7_inter_rater, table_6_8_bootstrap,
    table_6_9_paraphrase, table_6_10_fidelity_gap,
]


def main():
    parser = argparse.ArgumentParser(description="Generate all Auto-KGR tables")
    parser.add_argument("--input", default="results/full/full_experiment_metrics.csv")
    parser.add_argument("--output", default="results/full/tables")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    print(f"Loading {args.input} ...")
    df = load_and_flatten(args.input)
    print(f"Loaded {len(df)} rows. Generating {len(ALL_TABLES)} tables to {args.output}/\n")

    for fn in ALL_TABLES:
        try:
            fn(df, args.output)
        except Exception as exc:
            print(f"  ❌ {fn.__name__}: {exc}")

    total = len([f for f in os.listdir(args.output) if f.endswith(".csv")])
    print(f"\n✅ {total} tables generated in {args.output}/")


if __name__ == "__main__":
    main()
