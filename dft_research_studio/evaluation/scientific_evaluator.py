from __future__ import annotations
import json, logging, os, re, time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from groq import Groq
from rouge_score import rouge_scorer
from scipy import stats
from tqdm import tqdm
from ..config import Config
from .schemas import ExperimentResult

logger = logging.getLogger(__name__)
_JUDGE_RETRIES = 3
_JUDGE_DELAY   = 2.0


class ScientificEvaluator:
    """
    LLM-as-a-Judge with multi-judge support.
    Set JUDGE_MODEL and SECONDARY_JUDGE_MODEL in .env to avoid circular evaluation.
    """

    def __init__(self, results_file="experiment_results_raw.json", config=None):
        cfg = config or Config()
        self.cfg   = cfg
        self.top_k = cfg.top_k_retrieval

        with open(results_file) as f:
            raw = json.load(f)

        self.data: List[ExperimentResult] = []
        for entry in raw:
            try:
                self.data.append(ExperimentResult(**entry))
            except Exception as exc:
                logger.warning("Skipping malformed entry: %s", exc)

        logger.info("Loaded %d validated results.", len(self.data))

        self.judge_model           = os.getenv("JUDGE_MODEL", cfg.models_to_test[0])
        self.secondary_judge_model = os.getenv("SECONDARY_JUDGE_MODEL", "")

        if self.judge_model in cfg.models_to_test:
            logger.warning(
                "JUDGE_MODEL '%s' is the same as the generator — circular evaluation. "
                "Set JUDGE_MODEL=llama-3.3-70b-versatile in .env.",
                self.judge_model,
            )

        self._client = Groq(api_key=cfg.groq_api_key)
        self.rouge   = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        self._primary_scores:   List[float] = []
        self._secondary_scores: List[float] = []
        self._relevance_primary: List[float] = []
        self._relevance_secondary: List[float] = []
        self._ground_primary: List[float] = []
        self._ground_secondary: List[float] = []

    def _call_judge(self, prompt, model):
        import requests as _req
        for attempt in range(_JUDGE_RETRIES):
            try:
                resp = _req.post('http://localhost:11434/api/chat', json={
                    'model': model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'stream': False,
                    'options': {'temperature': 0, 'num_predict': 256},
                }, timeout=300)
                data = resp.json()
                return data.get('message', {}).get('content', '')
            except Exception as exc:
                logger.warning('Judge call failed (attempt %d): %s', attempt+1, exc)
                time.sleep(_JUDGE_DELAY * (2 ** attempt))
        return ""

    def _parse_score(self, text, default, scale=5.0):
        try:
            return float(json.loads(text).get("score", default))
        except Exception:
            pass
        m = re.search(r"\{.*?\}", text, re.DOTALL)
        if m:
            try:
                return float(json.loads(m.group()).get("score", default))
            except Exception:
                pass
        m = re.search(r"\b([0-9](?:\.[0-9]+)?)\b", text)
        if m:
            val = float(m.group(1))
            if 0 <= val <= scale:
                return val
        return default

    @staticmethod
    def _to_str(obj):
        if isinstance(obj, (dict, list, tuple)):
            return str(obj)
        return str(obj) if obj is not None else ""

    def evaluate_correctness(self, ground_truth, answer):
        prompt = f"""You are an expert quantum chemistry evaluator comparing a predicted answer against the ground truth.
Rate CORRECTNESS on a scale of 1-5:
1 = Completely wrong or irrelevant answer
2 = Mentions the right topic but key facts are wrong
3 = Partially correct, some important details missing or wrong
4 = Mostly correct with minor omissions
5 = Perfectly correct, matches ground truth in all key facts

Ground Truth: {ground_truth[:600]}
Predicted Answer: {answer[:600]}

Compare the predicted answer against the ground truth carefully. Check if specific values, functional names, and scientific claims match.
Respond ONLY with JSON: {{"score": <int 1-5>, "reason": "<one sentence explaining your score>"}}"""
        primary = self._parse_score(self._call_judge(prompt, self.judge_model), -1.0)
        secondary = -1.0
        if self.secondary_judge_model:
            secondary = self._parse_score(
                self._call_judge(prompt, self.secondary_judge_model), -1.0
            )
        return primary, secondary

    def evaluate_relevance(self, question, answer):
        prompt = f"""Rate how well the answer addresses the specific question asked, on a scale of 1-5:
1 = Does not address the question at all
2 = Tangentially related but doesn't answer the question
3 = Partially answers the question
4 = Answers the question well but misses some aspects
5 = Fully and precisely answers the question

Question: {question}
Answer: {answer[:600]}

Respond ONLY with JSON: {{"score": <int 1-5>}}"""
        primary = self._parse_score(self._call_judge(prompt, self.judge_model), 1.0)
        secondary = -1.0
        if self.secondary_judge_model:
            secondary = self._parse_score(self._call_judge(prompt, self.secondary_judge_model), -1.0)
        return primary, secondary


    @staticmethod
    def _clean_graph_context(context):
        """Reformat raw graph evidence into readable prose for groundedness evaluation."""
        if "GRAPH EVIDENCE" not in context and "CANDIDATE" not in context:
            return context  # Already clean (BM25/StandardRAG)
        import re
        lines = context.split("\n")
        facts = []
        current_id = ""
        current_type = ""
        current_connections = ""
        for line in lines:
            line = line.strip()
            if line.startswith("CANDIDATE") and "[ID:" in line:
                if current_id:
                    facts.append(f"{current_id} (type: {current_type}) is connected to {current_connections}.")
                m = re.search(r"\[ID:\s*(.+?)\]", line)
                current_id = m.group(1) if m else ""
                current_type = ""
                current_connections = ""
            elif line.startswith("- TYPE:"):
                current_type = line.replace("- TYPE:", "").strip()
            elif line.startswith("- CONNECTS TO:"):
                current_connections = line.replace("- CONNECTS TO:", "").strip()
            elif line.startswith("- VALIDATION:") or line.startswith("- VALUE:"):
                val = line.split(":", 1)[1].strip() if ":" in line else ""
                if val:
                    facts.append(f"{current_id} has validation data: {val}.")
        if current_id:
            facts.append(f"{current_id} (type: {current_type}) is connected to {current_connections}.")
        return " ".join(facts[:30]) if facts else context[:1200]

    def evaluate_groundedness(self, context, answer):
        prompt = f"""Rate GROUNDEDNESS 0 or 1.
0 = The answer makes factual claims that CONTRADICT or are NOT supported by the provided context
1 = The answer either (a) makes claims supported by the context, OR (b) honestly states that the evidence does not contain the answer

Context (retrieved evidence):
{context[:1200]}

Answer to evaluate:
{answer[:600]}

IMPORTANT: If the answer says the evidence does not contain the information, that is a GROUNDED response (score 1) because it accurately reflects the evidence limitations rather than hallucinating.
Respond ONLY with JSON: {{"score": <0 or 1>}}"""
        primary = self._parse_score(self._call_judge(prompt, self.judge_model), 0.0, scale=1.0)
        secondary = -1.0
        if self.secondary_judge_model:
            secondary = self._parse_score(
                self._call_judge(prompt, self.secondary_judge_model), -1.0, scale=1.0
            )
        return primary, secondary

    def evaluate_rouge(self, ground_truth, answer):
        s = self.rouge.score(ground_truth, answer)
        return {"rouge1_fmeasure": s["rouge1"].fmeasure, "rougeL_fmeasure": s["rougeL"].fmeasure}

    @staticmethod
    def _normalize_doc_name(name):
        """Normalize document names for fuzzy matching.
        Handles: PDF filenames, paper IDs (Caldeweyher2019), full titles."""
        import re
        n = str(name).lower().strip()
        # Remove file extensions
        n = re.sub(r'\.(pdf|txt|csv|json)$', '', n)
        # Remove common prefixes/suffixes
        n = n.replace('_', ' ').replace('-', ' ')
        # Extract author+year pattern (e.g. "caldeweyher2019")
        author_year = re.findall(r'[a-z]+\d{4}', n)
        return (n, set(author_year))

    def _docs_match(self, retrieved_name, gold_name):
        """Check if a retrieved doc matches a gold doc using fuzzy matching + paper_id mapping."""
        import json, os
        r_norm = str(retrieved_name).lower().strip()
        g_norm = str(gold_name).lower().strip().replace('.pdf', '')
        
        # Exact match
        if r_norm == g_norm or r_norm == g_norm + '.pdf':
            return True
        
        # Load paper_id mapping
        mapping_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'paper_id_mapping.json')
        try:
            mapping = json.load(open(mapping_file))
        except:
            mapping = {}
        
        # Check if retrieved is a paper_id that maps to a title substring
        title_substr = mapping.get(retrieved_name, mapping.get(r_norm, ""))
        if title_substr and title_substr.lower() in g_norm:
            return True
        
        # Check if gold doc filename contains the title substring
        if title_substr:
            # Gold doc is a filename, title_substr is from the paper
            g_words = set(w for w in g_norm.split() if len(w) >= 4)
            t_words = set(w for w in title_substr.lower().split() if len(w) >= 4)
            if t_words and g_words and len(t_words & g_words) >= 2:
                return True
        
        # Original fuzzy matching
        r_clean, r_ay = self._normalize_doc_name(retrieved_name)
        g_clean, g_ay = self._normalize_doc_name(gold_name)
        if r_ay and g_ay and r_ay & g_ay:
            return True
        if len(r_clean) > 4 and len(g_clean) > 4:
            r_words = set(w for w in r_clean.split() if len(w) >= 4)
            g_words = set(w for w in g_clean.split() if len(w) >= 4)
            if r_words and g_words and len(r_words & g_words) >= 2:
                return True
            if r_clean.replace(' ', '') in g_clean.replace(' ', ''):
                return True
            if g_clean.replace(' ', '') in r_clean.replace(' ', ''):
                return True
        return False

    def compute_gold_retrieval_metrics(self, retrieved, gold_docs):
        result = {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0}
        if not gold_docs:
            return result
        top_k = retrieved[:self.top_k]
        # Fuzzy matching: each gold doc checked against all retrieved
        gold_hits = 0
        for gold in gold_docs:
            for ret in top_k:
                if self._docs_match(ret, gold):
                    gold_hits += 1
                    break
        result["recall_at_k"] = gold_hits / len(gold_docs)
        # Precision: how many retrieved are relevant
        ret_hits = 0
        for ret in top_k:
            for gold in gold_docs:
                if self._docs_match(ret, gold):
                    ret_hits += 1
                    break
        result["precision_at_k"] = ret_hits / len(top_k) if top_k else 0.0
        # MRR: rank of first relevant
        for rank, ret in enumerate(retrieved, 1):
            for gold in gold_docs:
                if self._docs_match(ret, gold):
                    result["mrr"] = 1.0 / rank
                    return result
        return result

    @staticmethod
    def cohens_d(a, b):
        """Cohen's d (pooled SD). |d|<0.2 negligible, <0.5 small, <0.8 medium, >=0.8 large."""
        na, nb = len(a), len(b)
        if na < 2 or nb < 2:
            return float("nan")
        s = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
        return float((np.mean(a) - np.mean(b)) / s) if s > 0 else float("nan")

    def compute_inter_rater_reliability(self):
        if len(self._primary_scores) < 3 or len(self._secondary_scores) != len(self._primary_scores):
            return None
        r, p = stats.pearsonr(self._primary_scores, self._secondary_scores)
        logger.info(
            "Inter-rater reliability Pearson r=%.3f (p=%.4f) between '%s' and '%s'",
            r, p, self.judge_model, self.secondary_judge_model,
        )
        return float(r)

    def run_all_evaluations(self, output_csv="final_experiment_metrics.csv"):
        logger.info("Running evaluation (judge=%s) ...", self.judge_model)
        evaluated = []

        for entry in tqdm(self.data, desc="Evaluating"):
            ans    = self._to_str(entry.generated_answer)
            chunks = entry.retrieved_text_chunks
            if isinstance(chunks, str):
                chunks = [chunks] if chunks else []
            # Fallback: use context_used if chunks empty (graph architectures)
            if not chunks and hasattr(entry, 'context_used') and entry.context_used:
                ctx = entry.context_used
                if isinstance(ctx, str) and len(ctx) > 10:
                    chunks = [ctx]

            primary_corr, secondary_corr = self.evaluate_correctness(entry.ground_truth, ans)
            self._primary_scores.append(primary_corr)
            if secondary_corr >= 0:
                self._secondary_scores.append(secondary_corr)

            relevance_primary, relevance_secondary = self.evaluate_relevance(entry.question, ans)
            if chunks:
                ground_primary, ground_secondary = self.evaluate_groundedness(self._clean_graph_context("\n".join(chunks)), ans)
            else:
                ground_primary, ground_secondary = 0.0, 0.0
            metrics: Dict[str, Any] = {
                "correctness":              primary_corr,
                "correctness_secondary":    secondary_corr if secondary_corr >= 0 else None,
                "relevance":                relevance_primary,
                "relevance_secondary":      relevance_secondary if relevance_secondary >= 0 else None,
                "groundedness":             ground_primary,
                "groundedness_secondary":   ground_secondary if ground_secondary >= 0 else None,
                "judge_model":              self.judge_model,
            }
            self._relevance_primary.append(relevance_primary)
            if relevance_secondary >= 0:
                self._relevance_secondary.append(relevance_secondary)
            self._ground_primary.append(ground_primary)
            if ground_secondary >= 0:
                self._ground_secondary.append(ground_secondary)
            metrics.update(self.evaluate_rouge(entry.ground_truth, ans))
            gold = self.compute_gold_retrieval_metrics(
                entry.retrieved_source_filenames, entry.gold_docs
            )
            metrics.update({
                "gold_retrieval_recall_at_k":    gold["recall_at_k"],
                "gold_retrieval_precision_at_k": gold["precision_at_k"],
                "gold_retrieval_mrr":            gold["mrr"],
            })
            row = entry.model_dump()
            row["metrics"] = metrics
            evaluated.append(row)

        df = pd.DataFrame(evaluated)
        df["correctness_val"] = df["metrics"].apply(
            lambda x: x.get("correctness", 0.0) if isinstance(x, dict) else 0.0
        )
        baseline = df[df["experiment_type"] == "Baseline"]["correctness_val"].values

        if len(baseline) >= 2:
            for exp_type in df["experiment_type"].unique():
                if exp_type == "Baseline":
                    continue
                group = df[df["experiment_type"] == exp_type]["correctness_val"].values
                d  = self.cohens_d(group, baseline)
                ad = abs(d) if not np.isnan(d) else 0
                magnitude = ("negligible" if ad < 0.2 else "small" if ad < 0.5
                             else "medium" if ad < 0.8 else "large")
                df.loc[df["experiment_type"] == exp_type, "metrics"] = df.loc[
                    df["experiment_type"] == exp_type, "metrics"
                ].apply(lambda m: {**m, "cohens_d_vs_baseline": round(d, 4),
                                    "effect_size": magnitude})
                logger.info("Cohen's d (%s vs Baseline): %.3f (%s)", exp_type, d, magnitude)

        irr = self.compute_inter_rater_reliability()
        if irr is not None:
            df["inter_rater_r"] = irr

        df.drop(columns=["correctness_val"], errors="ignore").to_csv(output_csv, index=False)
        logger.info("Saved → '%s' (%d rows).", output_csv, len(df))
