"""
tests/test_statistical_analysis.py
-------------------------------------
Tests for AdvancedMetrics: Bootstrap CI correctness, Generative Fidelity
Gap (GFG) formula, and ReproducibilityTracker data fingerprinting.

All assertions are made against known mathematical properties of the
algorithms — no LLM or external service is invoked.
"""

from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from dft_research_studio.evaluation.advanced_metrics import AdvancedMetrics
from dft_research_studio.evaluation.reproducibility_tracker import ReproducibilityTracker


# ─────────────────────────────────────────────────────────────────────
# Shared fixture: minimal but realistic metrics CSV
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def metrics_df(tmp_path_factory):
    """
    A 20-row metrics DataFrame covering 2 experiment types × 5 ratios × 2 questions.
    Correctness values are deliberately chosen so that:
      - GraphRAG mean correctness = 4.0
      - StandardRAG mean correctness = 2.5
    This makes GFG arithmetic verifiable by hand.
    """
    tmp_path = tmp_path_factory.mktemp("metrics")
    rows = []
    for exp, corr in [("GraphRAG", 4.0), ("StandardRAG", 2.5)]:
        for ratio in [0.0, 0.5, 1.0, 2.0, 3.0]:
            for qid in ["Q1.1v0", "Q1.2v0"]:
                rows.append({
                    "question_id": qid,
                    "question": "What is PBE0?",
                    "ground_truth": "PBE0 is a hybrid functional.",
                    "gold_docs": str(["p1.pdf"]),
                    "model": "llama-3.1-8b-instant",
                    "distractor_ratio": ratio,
                    "experiment_type": exp,
                    "mode": "Topological" if "Graph" in exp else "Dense",
                    "generated_answer": "PBE0 is a hybrid functional.",
                    "context_used": "graph context",
                    "retrieved_text_chunks": str(["chunk"]),
                    "retrieved_source_filenames": str(["p1.pdf"]),
                    "metrics": str({
                        "correctness": corr,
                        "relevance": corr - 0.5,
                        "groundedness": 1.0 if corr >= 4.0 else 0.5,
                        "rouge1_fmeasure": 0.8 if corr >= 4.0 else 0.4,
                        "rougeL_fmeasure": 0.75 if corr >= 4.0 else 0.35,
                        "gold_retrieval_recall_at_k": 1.0 if corr >= 4.0 else 0.5,
                        "gold_retrieval_precision_at_k": 1.0 if corr >= 4.0 else 0.5,
                        "gold_retrieval_mrr": 1.0 if corr >= 4.0 else 0.5,
                    }),
                })
    path = tmp_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def adv(metrics_df):
    return AdvancedMetrics(metrics_path=metrics_df)


# ─────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────

class TestBootstrapCI:
    """
    Properties of the 95% bootstrap CI that must hold for any valid estimator:
      1. lower ≤ mean ≤ upper
      2. CI width > 0 for non-degenerate data
      3. CI width → 0 as variance → 0
      4. Seeded bootstrap is deterministic
    """

    def test_mean_within_ci_bounds(self, adv):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, lower, upper, _, _ = adv.get_bootstrap_ci(data)
        assert lower <= mean <= upper

    def test_ci_width_positive_for_variable_data(self, adv):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, lower, upper, _, _ = adv.get_bootstrap_ci(data)
        assert upper > lower

    def test_ci_width_zero_for_constant_data(self, adv):
        """
        A constant array has zero variance → CI collapses to mean.
        """
        data = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
        mean, lower, upper, _, _ = adv.get_bootstrap_ci(data)
        assert lower == pytest.approx(mean, abs=0.01)
        assert upper == pytest.approx(mean, abs=0.01)

    def test_mean_is_arithmetic_mean(self, adv):
        data = np.array([2.0, 4.0, 6.0])
        mean, _, _, _, _ = adv.get_bootstrap_ci(data)
        assert mean == pytest.approx(4.0)

    def test_seeded_bootstrap_is_deterministic(self, adv):
        data = np.random.default_rng(0).uniform(1, 5, 50)
        r1 = adv.get_bootstrap_ci(data)
        r2 = adv.get_bootstrap_ci(data)
        assert r1 == r2

    def test_single_value_returns_triple_equal_to_value(self, adv):
        mean, lower, upper, _, _ = adv.get_bootstrap_ci(np.array([3.7]))
        assert mean == pytest.approx(3.7)

    def test_wider_ci_for_higher_variance(self, adv):
        low_var = np.array([3.0, 3.1, 2.9, 3.0, 3.05])
        high_var = np.array([1.0, 5.0, 1.0, 5.0, 3.0])
        _, lo1, hi1, _, _ = adv.get_bootstrap_ci(low_var)
        _, lo2, hi2, _, _ = adv.get_bootstrap_ci(high_var)
        assert (hi2 - lo2) > (hi1 - lo1)


# ─────────────────────────────────────────────────────────────────────
# Generative Fidelity Gap
# ─────────────────────────────────────────────────────────────────────

class TestGenerativeFidelityGap:
    """
    GFG = 1 − (observed_mean / ceiling_mean).
    With ceiling = 4.0 (GraphRAG mean):
      - GraphRAG GFG at ratio 0.0 = 1 − 4.0/4.0 = 0.0
      - StandardRAG GFG at ratio 0.0 = 1 − 2.5/4.0 = 0.375
    """

    def test_gfg_values_bounded_between_zero_and_one(self, adv):
        gfg_df = adv.calculate_generative_gap()
        assert (gfg_df["Fidelity Gap"] >= 0.0).all()
        assert (gfg_df["Fidelity Gap"] <= 1.0).all()

    def test_best_system_has_lowest_gfg(self, adv):
        gfg_df = adv.calculate_generative_gap()
        # GraphRAG has highest correctness → lowest GFG
        graphrag_mean_gfg = gfg_df[gfg_df["Model"] == "Graph-Anchored"]["Fidelity Gap"].mean()
        stdrag_mean_gfg = gfg_df[gfg_df["Model"] == "StandardRAG"]["Fidelity Gap"].mean()
        assert graphrag_mean_gfg < stdrag_mean_gfg

    def test_gfg_for_ceiling_system_is_zero(self, adv):
        """
        The system with the highest mean correctness defines the ceiling,
        so its GFG must be zero (or very close, given rounding).
        """
        gfg_df = adv.calculate_generative_gap()
        min_gfg = gfg_df["Fidelity Gap"].min()
        assert min_gfg == pytest.approx(0.0, abs=1e-6)

    def test_gfg_standard_rag_at_ratio_0_is_correct(self, adv):
        """
        With ceiling = 4.0, StandardRAG correctness = 2.5:
        GFG = 1 - 2.5/4.0 = 0.375
        """
        gfg_df = adv.calculate_generative_gap()
        row = gfg_df[
            (gfg_df["Model"] == "StandardRAG") &
            (gfg_df["Distractor Ratio"] == 0.0)
        ]
        assert not row.empty
        assert row["Fidelity Gap"].iloc[0] == pytest.approx(0.375, abs=1e-6)

    def test_gfg_dataframe_has_required_columns(self, adv):
        gfg_df = adv.calculate_generative_gap()
        for col in ("Model", "Distractor Ratio", "Fidelity Gap", "Gap Display", "Interpretation"):
            assert col in gfg_df.columns

    def test_high_gap_labelled_critical_loss(self, adv):
        gfg_df = adv.calculate_generative_gap()
        high_gap = gfg_df[gfg_df["Fidelity Gap"] > 0.25]
        if not high_gap.empty:
            assert (high_gap["Interpretation"] == "Critical Loss").all()

    def test_low_gap_labelled_high_fidelity(self, adv):
        gfg_df = adv.calculate_generative_gap()
        low_gap = gfg_df[gfg_df["Fidelity Gap"] <= 0.25]
        if not low_gap.empty:
            assert (low_gap["Interpretation"] == "High Fidelity").all()


# ─────────────────────────────────────────────────────────────────────
# ReproducibilityTracker — data fingerprinting
# ─────────────────────────────────────────────────────────────────────

class TestReproducibilityTrackerFingerprinting:
    """
    MD5 fingerprinting must be:
      - Deterministic: same file → same hash every time
      - Sensitive: any byte change → different hash
      - Honest: missing file reported as "MISSING"
    """

    @pytest.fixture()
    def tracker(self, config):
        return ReproducibilityTracker(config)

    def test_same_file_produces_same_hash(self, tracker, tmp_path):
        f = tmp_path / "nodes.csv"
        f.write_text("node_id,label\nPBE0,Functional\n")
        h1 = tracker._hash_files([str(f)])
        h2 = tracker._hash_files([str(f)])
        assert h1 == h2

    def test_different_content_produces_different_hash(self, tracker, tmp_path):
        f1 = tmp_path / "v1.csv"
        f2 = tmp_path / "v2.csv"
        f1.write_text("node_id,label\nPBE0,Functional\n")
        f2.write_text("node_id,label\nB3LYP,Functional\n")
        h1 = tracker._hash_files([str(f1)])
        h2 = tracker._hash_files([str(f2)])
        assert list(h1.values())[0] != list(h2.values())[0]

    def test_single_byte_change_detected(self, tracker, tmp_path):
        f = tmp_path / "data.csv"
        f.write_bytes(b"PBE0,Functional")
        h1 = tracker._hash_files([str(f)])[str(f)]
        f.write_bytes(b"PBE1,Functional")  # one character changed
        h2 = tracker._hash_files([str(f)])[str(f)]
        assert h1 != h2

    def test_missing_file_returns_missing_sentinel(self, tracker, tmp_path):
        result = tracker._hash_files([str(tmp_path / "nonexistent.csv")])
        assert list(result.values())[0] == "MISSING"

    def test_hash_length_is_32_hexdigits(self, tracker, tmp_path):
        f = tmp_path / "check.csv"
        f.write_text("a,b\n1,2\n")
        result = tracker._hash_files([str(f)])
        assert len(list(result.values())[0]) == 32  # MD5 produces 128-bit = 32 hex chars

    def test_hash_matches_independent_md5_computation(self, tracker, tmp_path):
        content = b"node_id,label\nPBE0,Functional\nS66,Dataset\n"
        f = tmp_path / "verify.csv"
        f.write_bytes(content)
        expected = hashlib.md5(content).hexdigest()
        result = tracker._hash_files([str(f)])[str(f)]
        assert result == expected

    def test_reproducibility_log_json_is_valid(self, config, tmp_path):
        tracker = ReproducibilityTracker(config)
        path = str(tmp_path / "log.json")
        tracker.save_reproducibility_log(path)
        with open(path) as f:
            data = json.load(f)
        required_keys = {
            "timestamp_start", "timestamp_end", "python_version",
            "os_platform", "hardware", "libraries",
            "total_runtime_seconds", "estimated_kwh", "estimated_co2_grams",
        }
        assert required_keys <= set(data.keys()), (
            f"Missing keys: {required_keys - set(data.keys())}"
        )

    def test_compute_cost_is_non_negative(self, config):
        tracker = ReproducibilityTracker(config)
        tracker.metadata["total_runtime_seconds"] = 3600
        tracker._estimate_compute_cost()
        assert tracker.metadata["estimated_kwh"] >= 0.0
        assert tracker.metadata["estimated_co2_grams"] >= 0.0

    def test_compute_cost_scales_with_duration(self, config):
        """Doubling runtime must double the estimated energy consumption."""
        t1 = ReproducibilityTracker(config)
        t1.metadata["total_runtime_seconds"] = 3600
        t1._estimate_compute_cost()

        t2 = ReproducibilityTracker(config)
        t2.metadata["total_runtime_seconds"] = 7200
        t2._estimate_compute_cost()

        assert math.isclose(
            t2.metadata["estimated_kwh"],
            t1.metadata["estimated_kwh"] * 2,
            rel_tol=1e-6,
        )
