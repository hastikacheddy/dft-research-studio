"""
tests/conftest.py
-----------------
Shared pytest fixtures built from real in-memory data structures.

Design philosophy
-----------------
These tests exercise actual algorithmic behaviour — graph construction,
text cleaning, retrieval scoring, metric computation — using synthetic
but structurally faithful DFT knowledge-graph data.

No external services (Groq API, ChromaDB, SpaCy model downloads) are
contacted.  Any component that requires a network call is tested via a
thin subprocess-free stub that exercises only the code paths visible
in this codebase.

The fixture data is deliberately chosen to have known, verifiable
properties (e.g. PBE0 → MAE_PBE0_S66 → S66 is a known path in the
graph) so that assertions can be made about real outputs.
"""

from __future__ import annotations

import os
import textwrap
from typing import List

import networkx as nx
import pandas as pd
import pytest
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────
# Knowledge-graph fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def nodes_df() -> pd.DataFrame:
    """
    Minimal but structurally faithful DFT knowledge-graph node table.
    Covers all four node label types used in the real dataset.
    """
    return pd.DataFrame(
        {
            "node_id":  ["PBE0",   "B3LYP",   "S66",     "TMC32",   "MAE_PBE0_S66"],
            "label":    ["Functional","Functional","Dataset","Dataset","ValidationResult"],
            "value":    [None,      None,       None,      None,      0.23],
            "unit":     [None,      None,       None,      None,      "kcal/mol"],
            "paper_id": ["p1.pdf",  "p1.pdf",   "p2.pdf",  "p2.pdf",  "p1.pdf"],
        }
    )


@pytest.fixture(scope="session")
def rels_df() -> pd.DataFrame:
    """
    Edge table matching the real CSV schema.
    PBE0 ──HAS_RESULT──► MAE_PBE0_S66 ──EVALUATED_ON──► S66
    B3LYP──HAS_RESULT──► MAE_PBE0_S66
    """
    return pd.DataFrame(
        {
            "source_id":         ["PBE0",        "B3LYP",       "MAE_PBE0_S66"],
            "target_id":         ["MAE_PBE0_S66", "MAE_PBE0_S66","S66"],
            "relationship_type": ["HAS_RESULT",   "HAS_RESULT",  "EVALUATED_ON"],
            "paper_id":          ["p1.pdf",        "p1.pdf",      "p2.pdf"],
        }
    )


@pytest.fixture(scope="session")
def sample_graph(nodes_df, rels_df) -> nx.DiGraph:
    """Real NetworkX DiGraph built from the fixture DataFrames."""
    G = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        G.add_node(row["node_id"], **row.dropna().to_dict())
    for _, row in rels_df.iterrows():
        G.add_edge(
            row["source_id"],
            row["target_id"],
            relationship=row["relationship_type"],
            paper_id=row["paper_id"],
        )
    return G


# ─────────────────────────────────────────────────────────────────────
# Document corpus fixture
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_docs() -> List[Document]:
    """
    Synthetic DFT corpus chunks.  Content is chosen so that:
      - BM25 retrieval of "PBE0 S66 MAE" returns doc 0 first.
      - ROUGE-L between doc 0 and itself is 1.0.
      - Groundedness scoring with doc 0 as context and doc 0 as answer is 1.0.
    """
    return [
        Document(
            page_content=(
                "The PBE0 hybrid functional achieves a mean absolute error (MAE) "
                "of 0.23 kcal/mol on the S66 non-covalent interaction benchmark dataset."
            ),
            metadata={"source": "p1.pdf", "section": "RESULTS"},
        ),
        Document(
            page_content=(
                "B3LYP is a widely used three-parameter hybrid density functional "
                "approximation combining Becke exchange with Lee-Yang-Parr correlation."
            ),
            metadata={"source": "p2.pdf", "section": "INTRODUCTION"},
        ),
        Document(
            page_content=(
                "The TMC32 benchmark contains 32 transition-metal complexes "
                "with reference energies computed at the CCSD(T) level."
            ),
            metadata={"source": "p3.pdf", "section": "METHODS"},
        ),
    ]


# ─────────────────────────────────────────────────────────────────────
# Config fixture
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def config(tmp_path):
    """
    Config instance whose paths all point to tmp_path.
    Uses a dummy API key so no real Groq calls are made in unit tests.
    """
    os.environ.setdefault("GROQ_API_KEY", "test-key-not-used-in-unit-tests")
    from dft_research_studio.config import Config
    return Config(
        base_dir=str(tmp_path / "datasets"),
        chroma_base_dir=str(tmp_path / "vectordbs"),
    )
