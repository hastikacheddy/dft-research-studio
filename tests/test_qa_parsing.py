"""
tests/test_qa_parsing.py
--------------------------
Tests for DFTDataManager.get_qa_pairs() and the QA data schema.

These tests load real CSV data written to disk (no mocks) and assert on
the correctness of the regex-based gold-document parser, which is a
critical component for computing Recall@K during evaluation.

A single parsing error would corrupt all retrieval metrics, so these
tests are exhaustive over the parser's edge cases.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from dft_research_studio.config import Config
from dft_research_studio.data import DFTDataManager


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _make_manager(tmp_path, qa_rows: list[dict]) -> DFTDataManager:
    """Write the provided QA rows to CSV and return a DFTDataManager."""
    import pandas as pd

    nodes = pd.DataFrame({
        "node_id": ["PBE0"], "label": ["Functional"],
        "value": [None], "unit": [None], "paper_id": ["p1.pdf"],
    })
    rels = pd.DataFrame({
        "source_id": ["PBE0"], "target_id": ["PBE0"],
        "relationship_type": ["SELF"], "paper_id": ["p1.pdf"],
    })

    d = tmp_path
    d.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(d / "dft_kg_nodes.csv", index=False)
    rels.to_csv(d / "dft_kg_relationships.csv", index=False)
    pd.DataFrame(qa_rows).to_csv(d / "_DFT-QA-120.csv", index=False)

    os.environ.setdefault("GROQ_API_KEY", "test-key")
    cfg = Config(base_dir=str(tmp_path))
    return DFTDataManager(cfg)


# ─────────────────────────────────────────────────────────────────────
# Gold-document string parsing
# ─────────────────────────────────────────────────────────────────────

class TestGoldDocumentParsing:
    """
    The gold-doc column uses comma-separated PDF filenames.
    The parser must handle:
      - Single document
      - Multiple documents separated by comma-space
      - Trailing commas or whitespace
      - Empty / NaN cell
    """

    def test_single_gold_doc_parsed_correctly(self, tmp_path):
        dm = _make_manager(tmp_path, [{
            "Question number": "Q1.1v0",
            "Question": "What is PBE0?",
            "Ground truth": "A functional.",
            "Gold Standard Document": "paper1.pdf",
        }])
        pair = dm.get_qa_pairs()[0]
        assert pair["gold_docs"] == ["paper1.pdf"]

    def test_two_gold_docs_split_correctly(self, tmp_path):
        dm = _make_manager(tmp_path, [{
            "Question number": "Q1.2v0",
            "Question": "Compare PBE0 and B3LYP.",
            "Ground truth": "Both are hybrid functionals.",
            "Gold Standard Document": "paper1.pdf, paper2.pdf",
        }])
        pair = dm.get_qa_pairs()[0]
        assert set(pair["gold_docs"]) == {"paper1.pdf", "paper2.pdf"}

    def test_three_gold_docs_split_correctly(self, tmp_path):
        dm = _make_manager(tmp_path, [{
            "Question number": "Q1.3v0",
            "Question": "Overview.",
            "Ground truth": "Overview text.",
            "Gold Standard Document": "a.pdf, b.pdf, c.pdf",
        }])
        pair = dm.get_qa_pairs()[0]
        assert len(pair["gold_docs"]) == 3
        assert "a.pdf" in pair["gold_docs"]

    def test_all_gold_docs_end_with_dot_pdf(self, tmp_path):
        dm = _make_manager(tmp_path, [{
            "Question number": "Q1.4v0",
            "Question": "Multi-doc question.",
            "Ground truth": "GT.",
            "Gold Standard Document": "x.pdf, y.pdf",
        }])
        for doc in dm.get_qa_pairs()[0]["gold_docs"]:
            assert doc.lower().endswith(".pdf")

    def test_empty_gold_doc_cell_returns_empty_list(self, tmp_path):
        dm = _make_manager(tmp_path, [{
            "Question number": "Q1.5v0",
            "Question": "No gold doc.",
            "Ground truth": "GT.",
            "Gold Standard Document": "",
        }])
        assert dm.get_qa_pairs()[0]["gold_docs"] == []

    def test_gold_docs_with_internal_spaces_preserved(self, tmp_path):
        dm = _make_manager(tmp_path, [{
            "Question number": "Q2.1v0",
            "Question": "Spaced names.",
            "Ground truth": "GT.",
            "Gold Standard Document": "paper one.pdf, paper two.pdf",
        }])
        docs = dm.get_qa_pairs()[0]["gold_docs"]
        assert any("paper one.pdf" in d for d in docs)


# ─────────────────────────────────────────────────────────────────────
# Question number parsing
# ─────────────────────────────────────────────────────────────────────

class TestQuestionNumberParsing:
    """
    The question number field encodes both the original question ID (Q1.1)
    and the variation type (v0, v1, v2).  The parser must extract both.
    """

    def test_original_question_identifier_extracted(self, tmp_path):
        dm = _make_manager(tmp_path, [{
            "Question number": "Q2.5v1",
            "Question": "Q?",
            "Ground truth": "GT.",
            "Gold Standard Document": "p.pdf",
        }])
        pair = dm.get_qa_pairs()[0]
        assert pair["original_question_identifier"] == "Q2.5"

    def test_variation_type_extracted(self, tmp_path):
        dm = _make_manager(tmp_path, [{
            "Question number": "Q3.7v2",
            "Question": "Q?",
            "Ground truth": "GT.",
            "Gold Standard Document": "p.pdf",
        }])
        pair = dm.get_qa_pairs()[0]
        assert pair["variation_type"] == "v2"

    def test_variation_v0_extracted(self, tmp_path):
        dm = _make_manager(tmp_path, [{
            "Question number": "Q1.1v0",
            "Question": "Q?",
            "Ground truth": "GT.",
            "Gold Standard Document": "p.pdf",
        }])
        assert dm.get_qa_pairs()[0]["variation_type"] == "v0"

    def test_all_required_keys_present(self, tmp_path):
        dm = _make_manager(tmp_path, [{
            "Question number": "Q1.1v0",
            "Question": "What is PBE0?",
            "Ground truth": "A functional.",
            "Gold Standard Document": "p1.pdf",
        }])
        pair = dm.get_qa_pairs()[0]
        required = {"id", "original_question_identifier", "variation_type",
                    "question", "ground_truth", "gold_docs"}
        assert required <= set(pair.keys())

    def test_question_text_preserved_verbatim(self, tmp_path):
        q = "What is the MAE of PBE0 on the S66 benchmark in kcal/mol?"
        dm = _make_manager(tmp_path, [{
            "Question number": "Q1.1v0",
            "Question": q,
            "Ground truth": "0.23",
            "Gold Standard Document": "p1.pdf",
        }])
        assert dm.get_qa_pairs()[0]["question"] == q

    def test_ground_truth_preserved_verbatim(self, tmp_path):
        gt = "The PBE0 functional achieves a MAE of 0.23 kcal/mol on S66."
        dm = _make_manager(tmp_path, [{
            "Question number": "Q1.1v0",
            "Question": "What is the MAE?",
            "Ground truth": gt,
            "Gold Standard Document": "p1.pdf",
        }])
        assert dm.get_qa_pairs()[0]["ground_truth"] == gt

    def test_multiple_qa_pairs_all_parsed(self, tmp_path):
        rows = [
            {"Question number": f"Q1.{i}v0", "Question": f"Q{i}?",
             "Ground truth": f"GT{i}.", "Gold Standard Document": f"p{i}.pdf"}
            for i in range(1, 6)
        ]
        dm = _make_manager(tmp_path, rows)
        pairs = dm.get_qa_pairs()
        assert len(pairs) == 5
        for i, pair in enumerate(pairs, start=1):
            assert pair["question"] == f"Q{i}?"
