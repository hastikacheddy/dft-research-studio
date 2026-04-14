"""
data/data_manager.py
--------------------
Orchestrates data loading: knowledge-graph CSVs, PDF corpus, and QA pairs.
"""

from __future__ import annotations

import glob
import os
import random
import re
from typing import Dict, List

import networkx as nx
import pandas as pd
from langchain_core.documents import Document

from ..config import Config
from .pdf_processor import ScientificPDFProcessor


class DFTDataManager:
    """Central data layer.  One instance per experiment run."""

    def __init__(self, config: Config) -> None:
        self.config = config

        self.nodes_df: pd.DataFrame = pd.read_csv(config.nodes_path)
        self.rels_df: pd.DataFrame = pd.read_csv(config.rels_path)
        self.qa_df: pd.DataFrame = pd.read_csv(config.qa_path)
        self.qa_df.columns = self.qa_df.columns.str.strip()

        self.graph: nx.DiGraph = self._build_graph()

    # ------------------------------------------------------------------ #
    # Knowledge graph                                                      #
    # ------------------------------------------------------------------ #

    def _build_graph(self) -> nx.DiGraph:
        """Build a directed NetworkX graph from the CSV schema."""
        G = nx.DiGraph()

        for _, row in self.nodes_df.iterrows():
            attrs = row.dropna().to_dict()
            if "node_id" in attrs:
                G.add_node(row["node_id"], **attrs)

        for _, row in self.rels_df.iterrows():
            if pd.notna(row["source_id"]) and pd.notna(row["target_id"]):
                G.add_edge(
                    row["source_id"],
                    row["target_id"],
                    relationship=row["relationship_type"],
                    paper_id=row.get("paper_id", "Unknown"),
                )

        return G

    def remove_disconnected_nodes(self) -> int:
        """Remove isolated nodes (degree == 0) and return count removed."""
        isolated = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        for n in isolated:
            self.graph.remove_node(n)
        return len(isolated)

    # ------------------------------------------------------------------ #
    # PDF corpus                                                           #
    # ------------------------------------------------------------------ #

    def load_pdfs(self, distractor_ratio: float = 0.0) -> List[Document]:
        """
        Load the relevant corpus plus a controlled fraction of distractors.

        Parameters
        ----------
        distractor_ratio : float
            Fraction of distractor files relative to the relevant set.
            0.0 → no distractors, 1.0 → equal number, 3.0 → 3× as many.
        """
        relevant = glob.glob(os.path.join(self.config.paper_dir, "*.pdf"))
        distractors = glob.glob(os.path.join(self.config.distractor_dir, "*.pdf"))

        n_distractors = int(len(relevant) * distractor_ratio) if distractor_ratio > 0 else 0
        if n_distractors > 0 and distractors:
            random.seed(self.config.random_seed)
            selected_distractors = random.sample(
                distractors, min(n_distractors, len(distractors))
            )
        else:
            selected_distractors = []

        files = relevant + selected_distractors
        print(
            f"[DataManager] Processing {len(files)} PDFs "
            f"({len(relevant)} relevant, {len(selected_distractors)} distractors) …"
        )

        docs: List[Document] = []
        for pdf_path in files:
            sections = ScientificPDFProcessor.extract_sections(pdf_path)
            for section_name, text in sections.items():
                if len(text) < 50:
                    continue
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "section": section_name,
                        },
                    )
                )

        return docs

    # ------------------------------------------------------------------ #
    # QA pairs                                                             #
    # ------------------------------------------------------------------ #

    def get_qa_pairs(self) -> List[Dict]:
        """Parse the QA CSV into a list of structured dicts."""
        pairs: List[Dict] = []

        for _, row in self.qa_df.iterrows():
            gold_str = str(row.get("Gold Standard Document", "")).strip()
            q_num = str(row.get("Question number", "")).strip()

            m_orig = re.match(r"(Q\d+\.\d+)", q_num)
            m_var = re.search(r"(v\d+)", q_num)

            gold_docs: List[str] = []
            if gold_str and gold_str.lower() not in ('nan', 'none', ''):
                raw_parts = [
                    p.strip() + ".pdf"
                    for p in gold_str.split(".pdf")
                    if p.strip()
                ]
                for part in raw_parts:
                    clean = re.sub(r"^[\s,]+", "", part)
                    if clean.lower().endswith(".pdf"):
                        gold_docs.append(clean)

            pairs.append(
                {
                    "id": q_num,
                    "original_question_identifier": m_orig.group(1) if m_orig else q_num,
                    "variation_type": m_var.group(1) if m_var else "original",
                    "question": row["Question"],
                    "ground_truth": row["Ground truth"],
                    "gold_docs": gold_docs,
                }
            )

        return pairs
