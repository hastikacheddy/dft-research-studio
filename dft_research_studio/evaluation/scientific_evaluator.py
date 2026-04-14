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

    def _call_judge(self, prompt, model):
        for attempt in range(_JUDGE_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=256,
                )
                return resp.choices[0].message.content
            except Exception as exc:
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
        prompt = f"""You are an expert quantum chemistry evaluator.
Rate CORRECTNESS 1-5: 1=wrong, 3=mostly correct, 5=perfect.
Ground Truth: {ground_truth[:600]}
Predicted: {answer[:600]}
Respond ONLY with JSON: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""
        primary = self._parse_score(self._call_judge(prompt, self.judge_model), 1.0)
        secondary = -1.0
        if self.secondary_judge_model:
            secondary = self._parse_score(
                self._call_judge(prompt, self.secondary_judge_model), 1.0
            )
        return primary, secondary

    def evaluate_relevance(self, question, answer):
        prompt = f"""Rate RELEVANCE of the answer to the question 1-5.
Question: {question}
Answer: {answer[:600]}
Respond ONLY with JSON: {{"score": <int 1-5>}}"""
        return self._parse_score(self._call_judge(prompt, self.judge_model), 1.0)

    def evaluate_groundedness(self, context, answer):
        prompt = f"""Rate GROUNDEDNESS 0 or 1.
0=answer contains unsupported claims, 1=all claims supported by context.
Context: {context[:1200]}
Answer: {answer[:600]}
Respond ONLY with JSON: {{"score": <0 or 1>}}"""
        return self._parse_score(self._call_judge(prompt, self.judge_model), 0.0, scale=1.0)

    def evaluate_rouge(self, ground_truth, answer):
        s = self.rouge.score(ground_truth, answer)
        return {"rouge1_fmeasure": s["rouge1"].fmeasure, "rougeL_fmeasure": s["rougeL"].fmeasure}

    def compute_gold_retrieval_metrics(self, retrieved, gold_docs):
        result = {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0}
        if not gold_docs:
            return result
        gold_set = set(gold_docs)
        top_k    = retrieved[:self.top_k]
        hits     = len(gold_set & set(top_k))
        result["recall_at_k"]    = hits / len(gold_set)
        result["precision_at_k"] = hits / self.top_k if self.top_k else 0.0
        for rank, doc in enumerate(retrieved, 1):
            if doc in gold_set:
                result["mrr"] = 1.0 / rank
                break
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

            primary_corr, secondary_corr = self.evaluate_correctness(entry.ground_truth, ans)
            self._primary_scores.append(primary_corr)
            if secondary_corr >= 0:
                self._secondary_scores.append(secondary_corr)

            metrics: Dict[str, Any] = {
                "correctness":           primary_corr,
                "correctness_secondary": secondary_corr if secondary_corr >= 0 else None,
                "relevance":             self.evaluate_relevance(entry.question, ans),
                "groundedness":          self.evaluate_groundedness("\n".join(chunks), ans) if chunks else 0.0,
                "judge_model":           self.judge_model,
            }
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
