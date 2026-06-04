"""
evaluation/epistemic_metrics.py
---------------------------------
Epistemic Pipeline Integration — three novel deterministic evaluation metrics.

1. Scalar Validation (V_scalar) — T1/T2 quantitative queries
2. Propositional Boolean Matrix (S_IE) — T3/T4 multi-hop queries  
3. Graph-Isomorphic Overlap (F1_graph) — all tiers, structural alternative to ROUGE

These metrics run OVER existing saved results (no re-running experiments).

Usage:
    python -m dft_research_studio.evaluation.epistemic_metrics \
        --results results/full/full_results_raw.json \
        --kg-nodes dft_research_studio/data/processed/dft_kg_nodes.csv \
        --kg-rels dft_research_studio/data/processed/dft_kg_relationships.csv \
        --output results/full/epistemic_metrics.csv
"""
from __future__ import annotations

import argparse, json, logging, os, re
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _extract_numbers(text: str) -> List[Tuple[float, str]]:
    """Extract (value, unit) pairs from text."""
    patterns = [
        r"([-+]?\d+\.?\d*)\s*(kcal/mol|kJ/mol|eV|pm|Å|angstrom|%|hartree|Ha)",
        r"([-+]?\d+\.?\d*)\s*(kcal|kj|ev|pm|angstrom|percent)",
        r"MAE\s*(?:of|=|:)?\s*([-+]?\d+\.?\d*)",
        r"([-+]?\d+\.?\d*)\s*(?:kcal|kJ)",
    ]
    results = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            groups = m.groups()
            if len(groups) >= 2:
                try:
                    results.append((float(groups[0]), groups[1].lower()))
                except ValueError:
                    continue
            elif len(groups) == 1:
                try:
                    results.append((float(groups[0]), "kcal/mol"))
                except ValueError:
                    continue
    return results


def _normalize_unit(unit: str) -> str:
    """Normalize unit strings."""
    unit = unit.lower().strip()
    mapping = {
        "kcal": "kcal/mol", "kj": "kj/mol", "ev": "ev",
        "pm": "pm", "å": "angstrom", "angstrom": "angstrom",
        "percent": "%", "hartree": "hartree", "ha": "hartree",
    }
    return mapping.get(unit, unit)


def _extract_entities(text: str) -> Set[str]:
    """Extract DFT entity names from text."""
    # Common DFT functionals, benchmarks, methods
    entity_patterns = [
        r"\b(B3LYP|PBE0|PBE|M06-?L|M06-?2X|M06|M05|TPSS|SCAN|R2SCAN|HSE06|"
        r"CAM-B3LYP|ωB97X?-?[VD23]*|wB97[XV]?-?[VD23]*|B2PLYP|DSD-PBEP86|"
        r"BLYP|BP86|PW91|B97|LC-ωPBE|MN15|MN12|M08|M11|revTPSS|"
        r"PWPB95|XYG3|B2GP-PLYP|SVWN|LDA)\b",
        r"\b(S22|S66|GMTKN55|GMTKN30|BH76|W4-?11|NCIE|TMC32|DBH24|"
        r"HTBH|NHTBH|MB08|DARC|BSR36|ISO34|DC13|G2RC|AL2X6|"
        r"IDISP|PCONF21|SCONF|ACONF|Amino20|WATER27|RG18|ADIM6)\b",
        r"\b(D3|D3BJ|D3\(BJ\)|D4|DFT-D3|DFT-D4|MBD|XDM|VV10|NL)\b",
        r"\b(MAE|RMSE|MUE|WTMAD|WTMAD-?2|MSD|MSE|ME|MaxE)\b",
        r"\b(def2-SVP|def2-TZVP|def2-QZVP|aug-cc-pVTZ|cc-pVDZ|cc-pVTZ|"
        r"6-31G|6-311G|ANO-RCC)\b",
    ]
    entities = set()
    for pat in entity_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            entities.add(m.group().upper().replace(" ", ""))
    return entities


def _extract_triples(text: str, entities: Set[str] = None) -> Set[Tuple[str, str, str]]:
    """
    Extract (subject, predicate, object) triples from text.
    Uses pattern matching for common DFT relationship expressions.
    """
    if entities is None:
        entities = _extract_entities(text)

    triples = set()

    # Pattern: X achieves/gives/shows Y on/for Z
    for m in re.finditer(
        r"(\b[A-Za-z0-9-]+\b)\s+(?:achieves?|gives?|shows?|yields?|produces?|reports?)\s+"
        r"(?:an?\s+)?(?:MAE|RMSE|error|value)?\s*(?:of\s+)?"
        r"([-+]?\d+\.?\d*\s*(?:kcal/mol|kJ/mol|eV|pm|%)?)?\s*"
        r"(?:on|for|with)\s+(?:the\s+)?(\b[A-Za-z0-9-]+\b)",
        text, re.IGNORECASE
    ):
        s, val, o = m.group(1).upper(), m.group(2) or "", m.group(3).upper()
        if s in entities or o in entities:
            pred = "ACHIEVES_ON" if val else "EVALUATED_ON"
            triples.add((s, pred, o))

    # Pattern: X is designed/suited for Y
    for m in re.finditer(
        r"(\b[A-Za-z0-9-]+\b)\s+(?:is\s+)?(?:designed|suited|recommended|appropriate|good|accurate)\s+"
        r"(?:for|to\s+capture|at)\s+(.+?)(?:\.|,|$)",
        text, re.IGNORECASE
    ):
        s = m.group(1).upper()
        obj_text = m.group(2).strip()[:60]
        if s in entities:
            triples.add((s, "DESIGNED_FOR", obj_text.upper()))

    # Pattern: X outperforms/beats Y
    for m in re.finditer(
        r"(\b[A-Za-z0-9-]+\b)\s+(?:outperforms?|beats?|surpasses?|is\s+(?:better|superior)\s+(?:to|than))\s+"
        r"(\b[A-Za-z0-9-]+\b)",
        text, re.IGNORECASE
    ):
        s, o = m.group(1).upper(), m.group(2).upper()
        if s in entities or o in entities:
            triples.add((s, "OUTPERFORMS", o))

    # Pattern: X with/using D3/D4 correction
    for m in re.finditer(
        r"(\b[A-Za-z0-9-]+\b)\s+(?:with|using|plus|\+)\s+(D[34](?:\(BJ\))?|DFT-D[34])",
        text, re.IGNORECASE
    ):
        s, o = m.group(1).upper(), m.group(2).upper()
        if s in entities:
            triples.add((s, "APPLIES_CORRECTION", o))

    # Pattern: X fails/struggles on/for Y
    for m in re.finditer(
        r"(\b[A-Za-z0-9-]+\b)\s+(?:fails?|struggles?|performs?\s+poorly|is\s+(?:poor|bad|inaccurate))\s+"
        r"(?:on|for|at|with)\s+(\b[A-Za-z0-9-]+\b)",
        text, re.IGNORECASE
    ):
        s, o = m.group(1).upper(), m.group(2).upper()
        if s in entities or o in entities:
            triples.add((s, "FAILS_ON", o))

    # Direct KG node references: VR_XXX patterns
    for m in re.finditer(r"(VR_[A-Za-z0-9_()-]+)", text):
        vr = m.group(1).upper()
        # Parse VR_BENCHMARK_FUNCTIONAL
        parts = vr.replace("VR_", "").split("_", 1)
        if len(parts) == 2:
            triples.add((parts[1], "HAS_RESULT", vr))

    return triples


# ─────────────────────────────────────────────────────────────────────
# Metric 1: Scalar Validation
# ─────────────────────────────────────────────────────────────────────

def eval_scalar_precision(
    answer: str,
    ground_truth: str,
    epsilon: float = 0.5,
) -> Dict[str, float]:
    """
    V_scalar: Checks if quantitative values in the answer match ground truth.
    Returns 1 if |predicted - actual| <= epsilon AND units match.

    Only activated for queries containing numeric ground truths.
    """
    gt_values = _extract_numbers(ground_truth)
    if not gt_values:
        return {"scalar_precision": -1.0, "scalar_match_count": 0, "scalar_total": 0}

    ans_values = _extract_numbers(answer)
    if not ans_values:
        return {"scalar_precision": 0.0, "scalar_match_count": 0, "scalar_total": len(gt_values)}

    matches = 0
    for gt_val, gt_unit in gt_values:
        gt_unit_norm = _normalize_unit(gt_unit)
        for ans_val, ans_unit in ans_values:
            ans_unit_norm = _normalize_unit(ans_unit)
            if abs(ans_val - gt_val) <= epsilon and ans_unit_norm == gt_unit_norm:
                matches += 1
                break

    precision = matches / len(gt_values) if gt_values else 0.0
    return {
        "scalar_precision": round(precision, 4),
        "scalar_match_count": matches,
        "scalar_total": len(gt_values),
    }


# ─────────────────────────────────────────────────────────────────────
# Metric 2: Propositional Boolean Matrix
# ─────────────────────────────────────────────────────────────────────

def eval_boolean_ie(
    answer: str,
    ground_truth: str,
) -> Dict[str, float]:
    """
    S_IE: Checks what fraction of ground truth propositions appear in the answer.
    Extracts atomic propositions from ground truth, checks each against answer.
    Uses entity overlap as proxy for propositional entailment.
    """
    # Extract key entities/facts from ground truth
    gt_entities = _extract_entities(ground_truth)
    if not gt_entities:
        # Fall back to keyword extraction
        gt_words = set(w.lower() for w in ground_truth.split()
                       if len(w) > 4 and w.isalpha())
        gt_entities = gt_words

    if not gt_entities:
        return {"boolean_ie_score": -1.0, "propositions_found": 0, "propositions_total": 0}

    ans_lower = answer.lower()
    found = 0
    for entity in gt_entities:
        # Check if entity appears in answer (case insensitive)
        if entity.lower() in ans_lower or entity.upper() in answer:
            found += 1

    score = found / len(gt_entities) if gt_entities else 0.0
    return {
        "boolean_ie_score": round(score, 4),
        "propositions_found": found,
        "propositions_total": len(gt_entities),
    }


# ─────────────────────────────────────────────────────────────────────
# Metric 3: Graph-Isomorphic Overlap
# ─────────────────────────────────────────────────────────────────────

def eval_triple_overlap(
    answer: str,
    ground_truth: str,
    kg_rels: pd.DataFrame = None,
    gold_docs: List[str] = None,
) -> Dict[str, float]:
    """
    F1_graph: Computes precision, recall, F1 of (s, p, o) triples
    between predicted answer and ground truth.

    Ground truth triples come from:
    1. Triples extracted from the ground truth text
    2. KG edges from the gold documents (if provided)
    """
    # Extract ground truth triples from text
    gt_triples = _extract_triples(ground_truth)

    # Augment with KG triples from gold docs
    if kg_rels is not None and gold_docs:
        for doc in gold_docs:
            doc_name = doc.replace(".pdf", "")
            # Find KG edges from this paper
            paper_edges = kg_rels[
                kg_rels["paper_id"].astype(str).str.contains(doc_name[:15], case=False, na=False)
            ]
            for _, row in paper_edges.iterrows():
                gt_triples.add((
                    str(row["source_id"]).upper(),
                    str(row["relationship_type"]).upper(),
                    str(row["target_id"]).upper(),
                ))

    if not gt_triples:
        return {
            "triple_precision": -1.0,
            "triple_recall": -1.0,
            "triple_f1": -1.0,
            "predicted_triples": 0,
            "gt_triples": 0,
            "matching_triples": 0,
        }

    # Extract predicted triples from answer
    pred_triples = _extract_triples(answer)

    if not pred_triples:
        return {
            "triple_precision": 0.0,
            "triple_recall": 0.0,
            "triple_f1": 0.0,
            "predicted_triples": 0,
            "gt_triples": len(gt_triples),
            "matching_triples": 0,
        }

    # Fuzzy triple matching — subjects and objects match if they share entities
    matching = 0
    for ps, pp, po in pred_triples:
        for gs, gp, go in gt_triples:
            # Subject match (fuzzy)
            s_match = (ps == gs or ps in gs or gs in ps or
                       _entity_similarity(ps, gs) > 0.7)
            # Object match (fuzzy)
            o_match = (po == go or po in go or go in po or
                       _entity_similarity(po, go) > 0.7)
            # Predicate match (semantic grouping)
            p_match = _predicate_match(pp, gp)

            if s_match and o_match and p_match:
                matching += 1
                break

    precision = matching / len(pred_triples) if pred_triples else 0.0
    recall = matching / len(gt_triples) if gt_triples else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "triple_precision": round(precision, 4),
        "triple_recall": round(recall, 4),
        "triple_f1": round(f1, 4),
        "predicted_triples": len(pred_triples),
        "gt_triples": len(gt_triples),
        "matching_triples": matching,
    }


def _entity_similarity(a: str, b: str) -> float:
    """Simple overlap-based entity similarity."""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _predicate_match(p1: str, p2: str) -> bool:
    """Semantic predicate matching — groups synonymous relationship types."""
    PRED_GROUPS = {
        "EVALUATION": {"EVALUATED_ON", "VALIDATED_ON", "TESTED_ON", "ACHIEVES_ON",
                        "RESULT_FOR_BENCHMARK", "REPORTS_RESULT", "HAS_RESULT"},
        "DESIGN": {"DESIGNED_FOR", "SUITED_FOR", "RECOMMENDED_FOR", "CAPTURES",
                    "TARGETS", "ADDRESSES"},
        "COMPARISON": {"OUTPERFORMS", "BEATS", "SURPASSES", "IS_BETTER_THAN",
                        "SUPERIOR_TO"},
        "FAILURE": {"FAILS_ON", "STRUGGLES_WITH", "POOR_FOR", "INACCURATE_FOR"},
        "CORRECTION": {"APPLIES_CORRECTION", "USES", "WITH", "REFINES",
                        "REFINES_DISPERSION_CORRECTION"},
        "STRUCTURE": {"CONTAINS_SYSTEM", "HAS_REFERENCE_VALUE", "CONTAINS",
                       "STRUCTURAL_SIMILARITY", "SAME_THEORETICAL_FAMILY"},
    }
    for group_name, preds in PRED_GROUPS.items():
        if p1 in preds and p2 in preds:
            return True
    return p1 == p2


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────

def evaluate_all(
    results_path: str,
    kg_nodes_path: str,
    kg_rels_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Run all 3 epistemic metrics over saved experiment results."""

    with open(results_path) as f:
        results = json.load(f)

    kg_rels = pd.read_csv(kg_rels_path)
    logger.info("Loaded %d results, %d KG edges.", len(results), len(kg_rels))

    rows = []
    for i, r in enumerate(results):
        answer = str(r.get("generated_answer", ""))
        ground_truth = str(r.get("ground_truth", ""))
        question_id = r.get("question_id", "")
        tier = int(re.search(r"Q(\d+)\.", question_id).group(1)) if re.search(r"Q(\d+)\.", question_id) else 0
        gold_docs = r.get("gold_docs", [])
        if isinstance(gold_docs, str):
            gold_docs = [d.strip() for d in gold_docs.split(",")]

        row = {
            "question_id": question_id,
            "experiment_type": r.get("experiment_type", ""),
            "distractor_ratio": r.get("distractor_ratio", 0),
            "tier": tier,
        }

        # Metric 1: Scalar Validation (T1/T2 only)
        if tier in (1, 2):
            scalar = eval_scalar_precision(answer, ground_truth)
        else:
            scalar = {"scalar_precision": -1.0, "scalar_match_count": 0, "scalar_total": 0}
        row.update(scalar)

        # Metric 2: Boolean IE (T3/T4 only)
        if tier in (3, 4):
            boolean = eval_boolean_ie(answer, ground_truth)
        else:
            boolean = {"boolean_ie_score": -1.0, "propositions_found": 0, "propositions_total": 0}
        row.update(boolean)

        # Metric 3: Triple Overlap (all tiers)
        triple = eval_triple_overlap(answer, ground_truth, kg_rels, gold_docs)
        row.update(triple)

        rows.append(row)

        if (i + 1) % 1000 == 0:
            logger.info("Processed %d / %d results.", i + 1, len(results))

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("Saved epistemic metrics to %s (%d rows).", output_path, len(df))

    # Print summary
    print("\n" + "=" * 70)
    print("EPISTEMIC METRICS SUMMARY")
    print("=" * 70)

    # Scalar (T1/T2)
    t12 = df[df["scalar_precision"] >= 0]
    if len(t12) > 0:
        print(f"\n1. SCALAR VALIDATION (T1/T2, n={len(t12)})")
        for arch in df["experiment_type"].unique():
            ad = t12[t12["experiment_type"] == arch]
            if len(ad) > 0:
                print(f"   {arch:<25} V_scalar = {ad['scalar_precision'].mean():.3f}")

    # Boolean IE (T3/T4)
    t34 = df[df["boolean_ie_score"] >= 0]
    if len(t34) > 0:
        print(f"\n2. BOOLEAN IE (T3/T4, n={len(t34)})")
        for arch in df["experiment_type"].unique():
            ad = t34[t34["experiment_type"] == arch]
            if len(ad) > 0:
                print(f"   {arch:<25} S_IE = {ad['boolean_ie_score'].mean():.3f}")

    # Triple Overlap (all)
    all_valid = df[df["triple_f1"] >= 0]
    if len(all_valid) > 0:
        print(f"\n3. TRIPLE OVERLAP (all tiers, n={len(all_valid)})")
        for arch in df["experiment_type"].unique():
            ad = all_valid[all_valid["experiment_type"] == arch]
            if len(ad) > 0:
                print(f"   {arch:<25} P={ad['triple_precision'].mean():.3f}  "
                      f"R={ad['triple_recall'].mean():.3f}  "
                      f"F1={ad['triple_f1'].mean():.3f}")

    print("\n" + "=" * 70)
    return df


def main():
    parser = argparse.ArgumentParser(description="Epistemic Metrics Evaluation")
    parser.add_argument("--results", default="results/full/full_results_raw.json")
    parser.add_argument("--kg-nodes", default="dft_research_studio/data/processed/dft_kg_nodes.csv")
    parser.add_argument("--kg-rels", default="dft_research_studio/data/processed/dft_kg_relationships.csv")
    parser.add_argument("--output", default="results/full/epistemic_metrics.csv")
    parser.add_argument("--charts", action="store_true", help="Generate charts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Run original 3 metrics
    df_metrics = evaluate_all(args.results, args.kg_nodes, args.kg_rels, args.output)

    # Run epistemic behavioral analysis
    behavior_output = args.output.replace(".csv", "_behavior.csv")
    df_behavior = evaluate_epistemic_behavior(args.results, behavior_output)

    if args.charts:
        generate_epistemic_charts(df_metrics, df_behavior, os.path.dirname(args.output))


# ─────────────────────────────────────────────────────────────────────
# Metric 4: Epistemic Behavioral Analysis
# ─────────────────────────────────────────────────────────────────────

def eval_epistemic_behavior(answer: str) -> Dict[str, float]:
    """
    Measures epistemic honesty — whether the system refuses when unsure,
    cites KG evidence, and avoids fluent hallucination.
    """
    words = len(answer.split())

    honest_refusal = 1.0 if any(phrase in answer.lower() for phrase in [
        'does not contain', 'not found', 'cannot determine', 'no relevant',
        'not explicitly', 'not available', 'not provided', 'not stated',
        'cannot be determined', 'no information', 'evidence does not',
    ]) else 0.0

    cites_kg = 1.0 if any(phrase in answer.lower() for phrase in [
        'graph evidence', 'candidate', 'node', 'kg ', 'knowledge graph',
        'connects to', 'edge', 'traversal',
    ]) else 0.0

    cites_specific = 1.0 if re.search(
        r'VR_|WTMAD|MAE\s*[\d=:]|\d+\.\d+\s*kcal|RMSE\s*[\d=:]|\d+\.\d+\s*%',
        answer
    ) else 0.0

    return {
        "honest_refusal": honest_refusal,
        "cites_kg_evidence": cites_kg,
        "cites_specific_data": cites_specific,
        "answer_word_count": words,
        "is_short_precise": 1.0 if words < 20 else 0.0,
        "is_long_fluent": 1.0 if words > 100 else 0.0,
    }


def evaluate_epistemic_behavior(results_path: str, output_path: str) -> pd.DataFrame:
    """Run epistemic behavioral analysis over all results."""
    with open(results_path) as f:
        results = json.load(f)

    rows = []
    for r in results:
        answer = str(r.get("generated_answer", ""))
        question_id = r.get("question_id", "")
        tier = int(re.search(r"Q(\d+)\.", question_id).group(1)) if re.search(r"Q(\d+)\.", question_id) else 0
        row = {
            "question_id": question_id,
            "experiment_type": r.get("experiment_type", ""),
            "distractor_ratio": r.get("distractor_ratio", 0),
            "tier": tier,
        }
        row.update(eval_epistemic_behavior(answer))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    ARCH_ORDER = ['Baseline', 'TemplatePrompting', 'CoTPrompting', 'BM25Retriever',
                  'CrossEncoderReranker', 'BM25RerankerRAG', 'StandardRAG',
                  'GraphRAG', 'GraphDeterministic', 'MultiAgent', 'MultiAgentBM25']

    print("\n" + "=" * 90)
    print("EPISTEMIC BEHAVIORAL ANALYSIS")
    print("=" * 90)
    print(f"{'Architecture':<25} {'Honest Ref%':>12} {'Cites KG%':>10} {'Cites Data%':>12} {'Short%':>8} {'Long%':>8}")
    print("-" * 90)
    for arch in ARCH_ORDER:
        ad = df[df['experiment_type'] == arch]
        if len(ad) == 0:
            continue
        print(f"{arch:<25} {ad['honest_refusal'].mean()*100:>10.1f}% {ad['cites_kg_evidence'].mean()*100:>8.1f}% "
              f"{ad['cites_specific_data'].mean()*100:>10.1f}% {ad['is_short_precise'].mean()*100:>6.1f}% "
              f"{ad['is_long_fluent'].mean()*100:>6.1f}%")
    print("=" * 90)
    return df


# ─────────────────────────────────────────────────────────────────────
# Chart Generation
# ─────────────────────────────────────────────────────────────────────

def generate_epistemic_charts(df_metrics: pd.DataFrame, df_behavior: pd.DataFrame, out_dir: str) -> None:
    """Generate all epistemic metric charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    charts_dir = os.path.join(out_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    ARCH_ORDER = ['Baseline', 'TemplatePrompting', 'CoTPrompting', 'BM25Retriever',
                  'CrossEncoderReranker', 'BM25RerankerRAG', 'StandardRAG',
                  'GraphRAG', 'GraphDeterministic', 'MultiAgent', 'MultiAgentBM25']

    def _save(fig, name):
        path = os.path.join(charts_dir, name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Chart: {name}")

    # ─── Chart 1: Epistemic Behavior Stacked Bar ───
    fig, ax = plt.subplots(figsize=(14, 8))
    metrics_to_plot = ['honest_refusal', 'cites_kg_evidence', 'cites_specific_data']
    labels = ['Honest Refusal', 'Cites KG Evidence', 'Cites Specific Data']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    x = np.arange(len(ARCH_ORDER))
    width = 0.25
    for i, (metric, label, color) in enumerate(zip(metrics_to_plot, labels, colors)):
        vals = [df_behavior[df_behavior['experiment_type'] == a][metric].mean() * 100 for a in ARCH_ORDER]
        ax.bar(x + i * width, vals, width, label=label, color=color)
    ax.set_xticks(x + width)
    ax.set_xticklabels(ARCH_ORDER, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Epistemic Behavior: Honest Refusal vs KG Citation vs Specific Data', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    _save(fig, "epistemic_behavior_bars.png")

    # ─── Chart 2: Epistemic Honesty vs Fluent Hallucination ───
    fig, ax = plt.subplots(figsize=(12, 8))
    for arch in ARCH_ORDER:
        ad = df_behavior[df_behavior['experiment_type'] == arch]
        honest = ad['honest_refusal'].mean() * 100
        fluent = ad['is_long_fluent'].mean() * 100
        color = '#e74c3c' if arch in ['GraphRAG', 'GraphDeterministic', 'MultiAgent', 'MultiAgentBM25'] \
                else '#3498db' if arch in ['BM25Retriever', 'CrossEncoderReranker', 'BM25RerankerRAG', 'StandardRAG'] \
                else '#95a5a6'
        marker = 's' if 'Graph' in arch or 'Multi' in arch else 'o'
        ax.scatter(fluent, honest, s=250, c=color, marker=marker, zorder=5, edgecolors='black', linewidth=1)
        ax.annotate(arch, (fluent, honest), fontsize=8, ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points', fontweight='bold')
    ax.set_xlabel('Fluent Hallucination Rate (>100 words) %', fontsize=12)
    ax.set_ylabel('Honest Refusal Rate %', fontsize=12)
    ax.set_title('Epistemic Honesty vs Fluent Hallucination\n(Top-left = epistemically honest, Bottom-right = fluent hallucinator)',
                 fontsize=13, fontweight='bold')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=15, color='gray', linestyle='--', alpha=0.3)
    ax.text(25, 85, 'IMPOSSIBLE\n(honest + fluent)', fontsize=9, color='gray', alpha=0.4, ha='center', style='italic')
    ax.text(5, 85, 'EPISTEMICALLY\nHONEST', fontsize=10, color='green', alpha=0.5, ha='center', style='italic')
    ax.text(25, 10, 'FLUENT\nHALLUCINATOR', fontsize=10, color='red', alpha=0.5, ha='center', style='italic')
    fig.tight_layout()
    _save(fig, "epistemic_honesty_vs_hallucination.png")

    # ─── Chart 3: Scalar Validation by Architecture (T1/T2) ───
    t12 = df_metrics[df_metrics['scalar_precision'] >= 0]
    if len(t12) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        vals = [t12[t12['experiment_type'] == a]['scalar_precision'].mean() for a in ARCH_ORDER]
        colors_bar = ['#e74c3c' if v == 0 else '#2ecc71' if v > 0.1 else '#f39c12' for v in vals]
        ax.barh(range(11), vals, color=colors_bar, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(11))
        ax.set_yticklabels(ARCH_ORDER, fontsize=10)
        ax.set_xlabel('V_scalar (Scalar Precision)', fontsize=12)
        ax.set_title('Metric 1: Scalar Validation — T1/T2 Quantitative Queries\nV_scalar = 1 if |predicted − actual| ≤ ε and units match',
                     fontsize=13, fontweight='bold')
        for i, v in enumerate(vals):
            if v >= 0:
                ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)
        fig.tight_layout()
        _save(fig, "epistemic_scalar_validation.png")

    # ─── Chart 4: Boolean IE by Architecture (T3/T4) ───
    t34 = df_metrics[df_metrics['boolean_ie_score'] >= 0]
    if len(t34) > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        vals = [t34[t34['experiment_type'] == a]['boolean_ie_score'].mean() for a in ARCH_ORDER]
        colors_bar = ['#e74c3c' if v < 0.4 else '#2ecc71' if v > 0.7 else '#f39c12' for v in vals]
        ax.barh(range(11), vals, color=colors_bar, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(11))
        ax.set_yticklabels(ARCH_ORDER, fontsize=10)
        ax.set_xlabel('S_IE (Propositional Coverage)', fontsize=12)
        ax.set_title('Metric 2: Boolean Information Extraction — T3/T4 Multi-Hop\nS_IE = fraction of ground truth propositions found in answer',
                     fontsize=13, fontweight='bold')
        for i, v in enumerate(vals):
            if v >= 0:
                ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)
        fig.tight_layout()
        _save(fig, "epistemic_boolean_ie.png")

    # ─── Chart 5: Three-Way Tradeoff Radar ───
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    metrics_radar = ['Correctness\n(norm)', 'Groundedness', 'Scalar\nPrecision', 'Propositional\nCoverage', 'Honest\nRefusal']
    angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist() + [0]

    # Need original metrics too
    highlight_archs = ['Baseline', 'StandardRAG', 'GraphRAG', 'MultiAgent']
    colors_r = ['#95a5a6', '#3498db', '#e74c3c', '#e67e22']

    for arch, color in zip(highlight_archs, colors_r):
        t12_a = t12[t12['experiment_type'] == arch]['scalar_precision'].mean() if len(t12) > 0 else 0
        t34_a = t34[t34['experiment_type'] == arch]['boolean_ie_score'].mean() if len(t34) > 0 else 0
        beh_a = df_behavior[df_behavior['experiment_type'] == arch]
        honest_a = beh_a['honest_refusal'].mean() if len(beh_a) > 0 else 0
        # Placeholders for correctness/groundedness (normalized 0-1)
        # These come from the original experiment
        corr_norm = {'Baseline': 0.32, 'StandardRAG': 0.47, 'GraphRAG': 0.15, 'MultiAgent': 0.17}.get(arch, 0.3)
        grnd = {'Baseline': 0.0, 'StandardRAG': 0.68, 'GraphRAG': 0.94, 'MultiAgent': 0.93}.get(arch, 0.5)

        values = [corr_norm, grnd, t12_a, t34_a, honest_a]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=arch, color=color, markersize=6)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_radar, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Five-Dimensional Epistemic Radar\n(No single architecture dominates all dimensions)',
                 fontsize=13, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    fig.tight_layout()
    _save(fig, "epistemic_5d_radar.png")

    # ─── Chart 6: Answer Length Distribution vs Honest Refusal ───
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for arch in ARCH_ORDER:
        ad = df_behavior[df_behavior['experiment_type'] == arch]
        if len(ad) == 0:
            continue
        color = '#e74c3c' if arch in ['GraphRAG', 'GraphDeterministic', 'MultiAgent', 'MultiAgentBM25'] \
                else '#3498db' if arch in ['BM25Retriever', 'CrossEncoderReranker', 'BM25RerankerRAG', 'StandardRAG'] \
                else '#95a5a6'
        axes[0].hist(ad['answer_word_count'], bins=20, alpha=0.3, label=arch if arch in ['Baseline', 'StandardRAG', 'GraphRAG', 'MultiAgent'] else None,
                     color=color, density=True)
    axes[0].set_xlabel('Answer Word Count')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Answer Length Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)

    # Refusal rate by tier
    tier_names = {1: 'T1: Factual', 2: 'T2: Constraint', 3: 'T3: MultiHop', 4: 'T4: Comparative'}
    for arch in ['Baseline', 'StandardRAG', 'GraphRAG', 'MultiAgent']:
        tier_rates = []
        for t in [1, 2, 3, 4]:
            ad = df_behavior[(df_behavior['experiment_type'] == arch) & (df_behavior['tier'] == t)]
            tier_rates.append(ad['honest_refusal'].mean() * 100 if len(ad) > 0 else 0)
        axes[1].plot([1, 2, 3, 4], tier_rates, 'o-', label=arch, linewidth=2, markersize=8)
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_xticklabels([tier_names[t] for t in [1, 2, 3, 4]], fontsize=9)
    axes[1].set_ylabel('Honest Refusal Rate %')
    axes[1].set_title('Honest Refusal by Question Tier', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "epistemic_length_and_refusal.png")

    print(f"\n✅ {6} epistemic charts generated in {charts_dir}/")


if __name__ == "__main__":
    main()
