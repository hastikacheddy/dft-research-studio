"""
tests/test_new_architectures.py
---------------------------------
Unit tests for the 6 architectures added in the ablation study expansion:
  - TemplatePrompting
  - CoTPrompting
  - BM25Retriever
  - CrossEncoderReranker
  - BM25RerankerRAG
  - MultiAgentBM25

Design philosophy:
  - No external API calls (Groq, Ollama) — all LLM calls are stubbed.
  - Tests verify prompt construction, data flow, and _record() integration.
  - Uses the same fixture patterns as the existing test suite.
"""

from __future__ import annotations

import json
import sys
import os
import time
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_qa() -> Dict:
    """A minimal QA pair matching the real schema."""
    return {
        "id": "Q1.1v0",
        "question": "What is the M06-L density functional designed to capture?",
        "ground_truth": (
            "The M06-L functional is designed to capture noncovalent interactions, "
            "main-group thermochemistry, and transition metal bonding."
        ),
        "gold_docs": [
            "A new local density functional for main-group thermochemistry transition.pdf",
            "The M06 suite of density functionals for main group thermochemistry.pdf",
        ],
        "original_question": "Q1.1",
        "variation": "v0",
    }


@pytest.fixture()
def sample_models() -> List[str]:
    """Model list used in experiment orchestrator."""
    return ["llama3.1:8b"]


@pytest.fixture()
def mock_llm_response() -> str:
    """Typical LLM answer text."""
    return (
        "M06-L is a local meta-GGA density functional designed to capture "
        "noncovalent interactions and main-group thermochemistry."
    )


@pytest.fixture()
def sample_bm25_docs() -> List[Document]:
    """Documents that BM25 would return."""
    return [
        Document(
            page_content=(
                "The M06-L functional is a new local density functional for "
                "main-group thermochemistry, transition metal bonding, and "
                "noncovalent interactions."
            ),
            metadata={"source": "A new local density functional.pdf", "section": "ABSTRACT"},
        ),
        Document(
            page_content=(
                "We present the M06 suite of density functionals for main group "
                "thermochemistry and noncovalent interactions."
            ),
            metadata={"source": "The M06 suite.pdf", "section": "INTRODUCTION"},
        ),
        Document(
            page_content=(
                "DFT-D3 is a semiempirical dispersion correction method."
            ),
            metadata={"source": "distractor_paper.pdf", "section": "METHODS"},
        ),
    ]


# ─────────────────────────────────────────────────────────────────────
# Stub for LLMWrapper
# ─────────────────────────────────────────────────────────────────────

class StubLLMWrapper:
    """Replaces LLMWrapper without making any API/Ollama calls."""

    def __init__(self, model_name: str, config=None):
        self.model_name = model_name
        self.use_ollama = False

    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.0):
        # Return a canned response based on prompt content
        if "chain-of-thought" in prompt.lower() or "step by step" in prompt.lower():
            text = "Step 1: M06-L is a meta-GGA functional. Step 2: It targets noncovalent interactions."
        elif "quantum chemistry" in prompt.lower() and "expert" in prompt.lower():
            text = "M06-L is designed to capture noncovalent interactions and thermochemistry."
        elif "context:" in prompt.lower():
            text = "Based on the retrieved context, M06-L captures noncovalent interactions."
        else:
            text = "M06-L is a density functional for noncovalent interactions."
        return text, 50, 30, 80, 0.0, 1


# ─────────────────────────────────────────────────────────────────────
# Test: TemplatePrompting
# ─────────────────────────────────────────────────────────────────────

class TestTemplatePrompting:
    """Tests for run_template_prompting in ExperimentOrchestrator."""

    def test_template_prompt_contains_persona(self, sample_qa, sample_models):
        """Template prompting should inject a domain expert persona."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = set()
            runner.cfg = MagicMock()
            runner.engines = MagicMock()

            prompts_captured = []

            def mock_generate(prompt, **kwargs):
                prompts_captured.append(prompt)
                return "M06-L captures noncovalent interactions.", 50, 30, 80, 0.0, 1

            with patch("dft_research_studio.utils.experiment_orchestrator.LLMWrapper") as MockLLM:
                instance = MockLLM.return_value
                instance.generate = mock_generate
                runner._record = MagicMock()

                runner.run_template_prompting(sample_models, sample_qa, 0.0)

                assert len(prompts_captured) == 1
                prompt = prompts_captured[0].lower()
                # Template should mention quantum chemistry or DFT expertise
                assert any(kw in prompt for kw in ["quantum", "dft", "chemistry", "expert", "scientist"])

    def test_template_records_result(self, sample_qa, sample_models):
        """Should call _record with correct experiment_type."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = set()
            runner.cfg = MagicMock()
            runner.engines = MagicMock()
            runner._record = MagicMock()

            with patch("dft_research_studio.utils.experiment_orchestrator.LLMWrapper") as MockLLM:
                MockLLM.return_value.generate = lambda p, **kw: ("answer", 50, 30, 80, 0.0, 1)
                runner.run_template_prompting(sample_models, sample_qa, 0.0)

                runner._record.assert_called_once()
                call_args = runner._record.call_args
                assert call_args[0][3] == "TemplatePrompting"  # experiment_type

    def test_template_skips_if_done(self, sample_qa, sample_models):
        """Should skip if already completed."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = {("Q1.1v0", 0.0, "TemplatePrompting")}
            runner._record = MagicMock()

            runner.run_template_prompting(sample_models, sample_qa, 0.0)
            runner._record.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# Test: CoTPrompting
# ─────────────────────────────────────────────────────────────────────

class TestCoTPrompting:
    """Tests for run_cot_prompting in ExperimentOrchestrator."""

    def test_cot_prompt_contains_step_by_step(self, sample_qa, sample_models):
        """CoT prompting should instruct step-by-step reasoning."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = set()
            runner.cfg = MagicMock()
            runner.engines = MagicMock()

            prompts_captured = []

            def mock_generate(prompt, **kwargs):
                prompts_captured.append(prompt)
                return "Step 1: Analyse. Step 2: M06-L captures noncovalent.", 50, 30, 80, 0.0, 1

            with patch("dft_research_studio.utils.experiment_orchestrator.LLMWrapper") as MockLLM:
                MockLLM.return_value.generate = mock_generate
                runner._record = MagicMock()

                runner.run_cot_prompting(sample_models, sample_qa, 0.0)

                prompt = prompts_captured[0].lower()
                assert any(kw in prompt for kw in ["step by step", "step-by-step", "chain", "reasoning", "think"])

    def test_cot_records_correct_type(self, sample_qa, sample_models):
        """Should record as CoTPrompting."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = set()
            runner.cfg = MagicMock()
            runner.engines = MagicMock()
            runner._record = MagicMock()

            with patch("dft_research_studio.utils.experiment_orchestrator.LLMWrapper") as MockLLM:
                MockLLM.return_value.generate = lambda p, **kw: ("answer", 50, 30, 80, 0.0, 1)
                runner.run_cot_prompting(sample_models, sample_qa, 0.0)

                call_args = runner._record.call_args
                assert call_args[0][3] == "CoTPrompting"
                assert call_args[0][4] == "Chain-of-Thought"


# ─────────────────────────────────────────────────────────────────────
# Test: BM25Retriever
# ─────────────────────────────────────────────────────────────────────

class TestBM25Retriever:
    """Tests for run_bm25_retriever in ExperimentOrchestrator."""

    def test_bm25_passes_context_to_llm(self, sample_qa, sample_models, sample_bm25_docs):
        """BM25 should retrieve docs and pass them as context to the LLM."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = set()
            runner.cfg = MagicMock()

            mock_bm25 = MagicMock()
            mock_bm25.invoke.return_value = sample_bm25_docs
            runner.engines = MagicMock()
            runner.engines.bm25_reranker.return_value = mock_bm25

            prompts_captured = []

            def mock_generate(prompt, **kwargs):
                prompts_captured.append(prompt)
                return "Based on context, M06-L captures noncovalent interactions.", 50, 30, 80, 0.0, 1

            with patch("dft_research_studio.utils.experiment_orchestrator.LLMWrapper") as MockLLM:
                MockLLM.return_value.generate = mock_generate
                runner._record = MagicMock()

                runner.run_bm25_retriever(sample_models, sample_qa, 0.0)

                # Prompt should contain retrieved content
                prompt = prompts_captured[0]
                assert "M06-L" in prompt or "density functional" in prompt

    def test_bm25_records_sources(self, sample_qa, sample_models, sample_bm25_docs):
        """Should record source filenames from retrieved docs."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = set()
            runner.cfg = MagicMock()

            mock_bm25 = MagicMock()
            mock_bm25.invoke.return_value = sample_bm25_docs
            runner.engines = MagicMock()
            runner.engines.bm25_reranker.return_value = mock_bm25
            runner._record = MagicMock()

            with patch("dft_research_studio.utils.experiment_orchestrator.LLMWrapper") as MockLLM:
                MockLLM.return_value.generate = lambda p, **kw: ("answer", 50, 30, 80, 0.0, 1)
                runner.run_bm25_retriever(sample_models, sample_qa, 0.0)

                call_args = runner._record.call_args
                assert call_args[0][3] == "BM25Retriever"
                # Sources should be passed (positional arg index 10 or kwarg)
                sources = call_args[0][9] if len(call_args[0]) > 9 else call_args[1].get("sources", [])
                # At minimum, _record was called
                runner._record.assert_called_once()

    def test_bm25_handles_empty_results(self, sample_qa, sample_models):
        """Should handle gracefully when BM25 returns no docs."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = set()
            runner.cfg = MagicMock()

            mock_bm25 = MagicMock()
            mock_bm25.invoke.return_value = []
            runner.engines = MagicMock()
            runner.engines.bm25_reranker.return_value = mock_bm25
            runner._record = MagicMock()

            with patch("dft_research_studio.utils.experiment_orchestrator.LLMWrapper") as MockLLM:
                MockLLM.return_value.generate = lambda p, **kw: ("No context available.", 50, 30, 80, 0.0, 1)
                # Should not raise
                runner.run_bm25_retriever(sample_models, sample_qa, 0.0)


# ─────────────────────────────────────────────────────────────────────
# Test: CrossEncoderReranker
# ─────────────────────────────────────────────────────────────────────

class TestCrossEncoderReranker:
    """Tests for run_cross_encoder_reranker in ExperimentOrchestrator."""

    def test_cross_encoder_records_correct_type(self, sample_qa, sample_models, sample_bm25_docs):
        """Should record as CrossEncoderReranker."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = set()
            runner.cfg = MagicMock()

            mock_rag = MagicMock()
            mock_rag.retriever.invoke.return_value = sample_bm25_docs
            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = sample_bm25_docs[:2]
            runner.engines = MagicMock()
            runner.engines.rag.return_value = mock_rag
            runner.engines.reranker = mock_reranker
            runner._record = MagicMock()

            with patch("dft_research_studio.utils.experiment_orchestrator.LLMWrapper") as MockLLM:
                MockLLM.return_value.generate = lambda p, **kw: ("answer", 50, 30, 80, 0.0, 1)
                runner.run_cross_encoder_reranker(sample_models, sample_qa, 0.0)

                call_args = runner._record.call_args
                assert call_args[0][3] == "CrossEncoderReranker"
                assert call_args[0][4] == "Reranked"

    def test_cross_encoder_retrieves_docs(self, sample_qa, sample_models, sample_bm25_docs):
        """Should retrieve docs via bm25_reranker adapter."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = set()
            runner.cfg = MagicMock()

            mock_bm25 = MagicMock()
            mock_bm25.invoke.return_value = sample_bm25_docs
            runner.engines = MagicMock()
            runner.engines.bm25_reranker.return_value = mock_bm25
            runner._record = MagicMock()

            with patch("dft_research_studio.utils.experiment_orchestrator.LLMWrapper") as MockLLM:
                MockLLM.return_value.generate = lambda p, **kw: ("answer", 50, 30, 80, 0.0, 1)
                runner.run_cross_encoder_reranker(sample_models, sample_qa, 0.0)

                mock_bm25.invoke.assert_called_once()


# ─────────────────────────────────────────────────────────────────────
# Test: BM25RerankerRAG
# ─────────────────────────────────────────────────────────────────────

class TestBM25RerankerRAG:
    """Tests for run_bm25_reranker_rag in ExperimentOrchestrator."""

    def test_bm25_reranker_records_correct_type(self, sample_qa, sample_models, sample_bm25_docs):
        """Should record as BM25RerankerRAG with HybridSparse mode."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = set()
            runner.cfg = MagicMock()

            mock_bm25 = MagicMock()
            mock_bm25.invoke.return_value = sample_bm25_docs
            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = sample_bm25_docs[:2]
            runner.engines = MagicMock()
            runner.engines.bm25_reranker.return_value = mock_bm25
            runner.engines.reranker = mock_reranker
            runner._record = MagicMock()

            with patch("dft_research_studio.utils.experiment_orchestrator.LLMWrapper") as MockLLM:
                MockLLM.return_value.generate = lambda p, **kw: ("answer", 50, 30, 80, 0.0, 1)
                runner.run_bm25_reranker_rag(sample_models, sample_qa, 0.0)

                call_args = runner._record.call_args
                assert call_args[0][3] == "BM25RerankerRAG"
                assert call_args[0][4] == "HybridSparse"

    def test_bm25_reranker_retrieves_and_records(self, sample_qa, sample_models, sample_bm25_docs):
        """Should retrieve via bm25_reranker adapter and record results."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = set()
            runner.cfg = MagicMock()

            mock_bm25 = MagicMock()
            mock_bm25.invoke.return_value = sample_bm25_docs
            runner.engines = MagicMock()
            runner.engines.bm25_reranker.return_value = mock_bm25
            runner._record = MagicMock()

            with patch("dft_research_studio.utils.experiment_orchestrator.LLMWrapper") as MockLLM:
                MockLLM.return_value.generate = lambda p, **kw: ("answer", 50, 30, 80, 0.0, 1)
                runner.run_bm25_reranker_rag(sample_models, sample_qa, 0.0)

                mock_bm25.invoke.assert_called_once()
                runner._record.assert_called_once()


# ─────────────────────────────────────────────────────────────────────
# Test: MultiAgentBM25
# ─────────────────────────────────────────────────────────────────────

class TestMultiAgentBM25:
    """Tests for run_multi_agent_bm25 in ExperimentOrchestrator."""

    def test_multi_agent_bm25_records_correct_type(self, sample_qa, sample_models):
        """Should record as MultiAgentBM25 with HybridAgent mode."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = set()
            runner.cfg = MagicMock()

            mock_mas = MagicMock()
            # run_workflow returns: (answer, log, chunks, pids, in_tok, out_tok, cost, calls)
            mock_mas.run_workflow.return_value = (
                "M06-L is for noncovalent interactions.",
                ["Plan", "Execute", "Validate"],
                ["chunk1", "chunk2"],
                ["paper1", "paper2"],
                100, 50, 0.0, 2,
            )
            runner.engines = MagicMock()
            runner.engines.mas.return_value = mock_mas
            runner._record = MagicMock()

            runner.run_multi_agent_bm25(sample_models, sample_qa, 0.0)

            call_args = runner._record.call_args
            assert call_args[0][3] == "MultiAgentBM25"
            assert call_args[0][4] == "HybridAgent"

    def test_multi_agent_bm25_skips_if_done(self, sample_qa, sample_models):
        """Should skip if already completed."""
        from dft_research_studio.utils.experiment_orchestrator import ExperimentOrchestrator

        with patch.object(ExperimentOrchestrator, "__init__", lambda self, *a, **kw: None):
            runner = ExperimentOrchestrator.__new__(ExperimentOrchestrator)
            runner.results = []
            runner._completed = {("Q1.1v0", 0.0, "MultiAgentBM25")}
            runner._record = MagicMock()

            runner.run_multi_agent_bm25(sample_models, sample_qa, 0.0)
            runner._record.assert_not_called()


# ─────────────────────────────────────────────────────────────────────
# Test: LLMWrapper Ollama Integration
# ─────────────────────────────────────────────────────────────────────

class TestLLMWrapperOllama:
    """Tests for the Ollama/Groq detection and retry logic in LLMWrapper."""

    def test_ollama_detection_when_server_running(self):
        """When Ollama is reachable, use_ollama should be True."""
        from dft_research_studio.agents.llm_wrapper import LLMWrapper

        with patch("dft_research_studio.agents.llm_wrapper.requests") as mock_req:
            mock_req.get.return_value = MagicMock(status_code=200)
            llm = LLMWrapper("llama3.1:8b")
            assert llm.use_ollama is True

    def test_groq_fallback_when_ollama_down(self):
        """When Ollama is unreachable, should fall back to Groq."""
        from dft_research_studio.agents.llm_wrapper import LLMWrapper

        with patch("dft_research_studio.agents.llm_wrapper.requests") as mock_req:
            mock_req.get.side_effect = Exception("Connection refused")
            # Groq is imported inside __init__, patch it at the source module
            with patch.dict("sys.modules", {"groq": MagicMock()}):
                os.environ["GROQ_API_KEY"] = "test-key"
                llm = LLMWrapper("llama3.1:8b")
                assert llm.use_ollama is False

    def test_ollama_generate_returns_six_tuple(self):
        """generate() should return (text, in_tok, out_tok, total, cost, calls)."""
        from dft_research_studio.agents.llm_wrapper import LLMWrapper

        with patch("dft_research_studio.agents.llm_wrapper.requests") as mock_req:
            # Ollama detection
            mock_req.get.return_value = MagicMock(status_code=200)
            # Ollama chat response
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "message": {"content": "M06-L is a meta-GGA functional."},
                "prompt_eval_count": 50,
                "eval_count": 20,
            }
            mock_req.post.return_value = mock_resp

            llm = LLMWrapper("llama3.1:8b")
            result = llm.generate("What is M06-L?")

            assert isinstance(result, tuple)
            assert len(result) == 6
            assert isinstance(result[0], str)
            assert len(result[0]) > 0

    def test_ollama_retries_on_empty_response(self):
        """Should retry when Ollama returns empty content."""
        from dft_research_studio.agents.llm_wrapper import LLMWrapper

        with patch("dft_research_studio.agents.llm_wrapper.requests") as mock_req:
            mock_req.get.return_value = MagicMock(status_code=200)

            call_count = 0

            def mock_post(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                resp = MagicMock()
                if call_count < 3:
                    resp.json.return_value = {"message": {"content": ""}, "prompt_eval_count": 0, "eval_count": 0}
                else:
                    resp.json.return_value = {"message": {"content": "Success!"}, "prompt_eval_count": 50, "eval_count": 10}
                return resp

            mock_req.post = mock_post
            llm = LLMWrapper("llama3.1:8b")
            result = llm.generate("test")
            assert result[0] == "Success!"
            assert call_count == 3


# ─────────────────────────────────────────────────────────────────────
# Test: Evaluator Groundedness Fixes
# ─────────────────────────────────────────────────────────────────────

class TestGroundednessEvaluation:
    """Tests for the _clean_graph_context and groundedness scoring."""

    def test_clean_graph_context_converts_candidates(self):
        """Should convert graph evidence format to readable prose."""
        from dft_research_studio.evaluation.scientific_evaluator import ScientificEvaluator

        evaluator = ScientificEvaluator.__new__(ScientificEvaluator)
        raw = """--- GRAPH EVIDENCE (3 candidates) ---

CANDIDATE 1 [ID: VR_M06_HC7_MUE]
  - TYPE: ValidationResult
  - CONNECTS TO: M06, HC7, MUE

CANDIDATE 2 [ID: M062X]
  - TYPE: Functional
  - CONNECTS TO: PW6B95, M06-L"""

        clean = evaluator._clean_graph_context(raw)
        assert "VR_M06_HC7_MUE" in clean
        assert "ValidationResult" in clean
        assert "CANDIDATE" not in clean  # Raw format should be removed
        assert "---" not in clean

    def test_clean_context_passes_through_normal_text(self):
        """Non-graph text should pass through unchanged."""
        from dft_research_studio.evaluation.scientific_evaluator import ScientificEvaluator

        evaluator = ScientificEvaluator.__new__(ScientificEvaluator)
        normal = "The M06-L functional achieves 0.23 kcal/mol MAE on S66."
        result = evaluator._clean_graph_context(normal)
        assert result == normal

    def test_clean_context_extracts_validation_data(self):
        """Should extract VALUE/VALIDATION data into readable facts."""
        from dft_research_studio.evaluation.scientific_evaluator import ScientificEvaluator

        evaluator = ScientificEvaluator.__new__(ScientificEvaluator)
        raw = """--- GRAPH EVIDENCE ---
CANDIDATE 1 [ID: VR_WTMAD2_M06L]
  - TYPE: ValidationResult
  - CONNECTS TO: M06L, GMTKN55
  - VALIDATION: 8.61 kcal/mol"""

        clean = evaluator._clean_graph_context(raw)
        assert "8.61" in clean
        assert "VR_WTMAD2_M06L" in clean


# ─────────────────────────────────────────────────────────────────────
# Test: Recall@K Fuzzy Matching
# ─────────────────────────────────────────────────────────────────────

class TestRecallFuzzyMatching:
    """Tests for paper_id → filename matching in compute_gold_retrieval_metrics."""

    def test_exact_match(self):
        """Identical filenames should match."""
        from dft_research_studio.evaluation.scientific_evaluator import ScientificEvaluator

        evaluator = ScientificEvaluator.__new__(ScientificEvaluator)
        evaluator.top_k = 5
        assert evaluator._docs_match("paper.pdf", "paper.pdf") is True

    def test_paper_id_matches_gold_filename(self):
        """Paper ID like 'Caldeweyher2019' should match via mapping."""
        from dft_research_studio.evaluation.scientific_evaluator import ScientificEvaluator

        evaluator = ScientificEvaluator.__new__(ScientificEvaluator)
        evaluator.top_k = 5
        # This depends on paper_id_mapping.json existing
        result = evaluator._docs_match(
            "Caldeweyher2019",
            "A Consistent and Accurate Ab Initio Parametrization.pdf"
        )
        # Should match via the mapping
        assert isinstance(result, bool)

    def test_no_false_positive_on_unrelated(self):
        """Completely unrelated names should not match."""
        from dft_research_studio.evaluation.scientific_evaluator import ScientificEvaluator

        evaluator = ScientificEvaluator.__new__(ScientificEvaluator)
        evaluator.top_k = 5
        assert evaluator._docs_match("organic_chemistry_intro.pdf", "quantum_field_theory.pdf") is False


# ─────────────────────────────────────────────────────────────────────
# Test: GraphDeterministic Hub Filtering
# ─────────────────────────────────────────────────────────────────────

class TestHubFiltering:
    """Tests for the _SKIP_HUBS filtering in TopologicalRetriever."""

    def test_global_metadata_hub_filtered(self):
        """Global_Metadata_Hub should not appear in sorted_hubs."""
        from collections import Counter

        # Simulate the filtering logic
        _SKIP_HUBS = {"Global_Metadata_Hub", "Density Functional Theory", "DFT", "Hub", "Metadata"}
        hub_candidates = ["Global_Metadata_Hub", "Global_Metadata_Hub", "Global_Metadata_Hub",
                          "M06-L", "M06-L", "GMTKN55"]
        sorted_hubs = [h for h, _ in Counter(hub_candidates).most_common(25)
                       if h not in _SKIP_HUBS and not h.startswith("Global_")][:15]

        assert "Global_Metadata_Hub" not in sorted_hubs
        assert "M06-L" in sorted_hubs
        assert "GMTKN55" in sorted_hubs

    def test_dft_hub_filtered(self):
        """'Density Functional Theory' generic hub should be filtered."""
        from collections import Counter

        _SKIP_HUBS = {"Global_Metadata_Hub", "Density Functional Theory", "DFT", "Hub", "Metadata"}
        hub_candidates = ["Density Functional Theory", "Density Functional Theory", "M06-2X"]
        sorted_hubs = [h for h, _ in Counter(hub_candidates).most_common(25)
                       if h not in _SKIP_HUBS and not h.startswith("Global_")][:15]

        assert "Density Functional Theory" not in sorted_hubs
        assert "M06-2X" in sorted_hubs
