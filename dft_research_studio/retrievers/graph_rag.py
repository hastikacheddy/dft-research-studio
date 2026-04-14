"""
retrievers/graph_rag.py
-----------------------
Star-topology knowledge-graph retriever with fuzzy entity resolution.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

import pandas as pd
import spacy
from thefuzz import fuzz, process

from ..config import Config


class GraphRAG:
    """
    Retrieves graph evidence via fuzzy NLP entity matching and
    star-topology intersection on the knowledge graph.
    """

    def __init__(self, nodes_df: pd.DataFrame, rels_df: pd.DataFrame) -> None:
        self.nodes_df = nodes_df
        self.rels_df = rels_df
        self.node_map: Dict = nodes_df.set_index("node_id").to_dict("index")

        print("[GraphRAG] Loading SpaCy model …")
        model_name = Config().spacy_model
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)

    # ------------------------------------------------------------------ #
    # Entity resolution                                                    #
    # ------------------------------------------------------------------ #

    def find_closest_nodes(self, term: str, top_k: int = 3) -> List[str]:
        """Fuzzy-match *term* against all node IDs (≥ 60 % similarity)."""
        all_ids = [str(x) for x in self.nodes_df["node_id"].unique()]
        matches = process.extract(
            term, all_ids, limit=top_k, scorer=fuzz.token_sort_ratio
        )
        return [m[0] for m in matches if m[1] > 60]

    # ------------------------------------------------------------------ #
    # Star-context retrieval                                               #
    # ------------------------------------------------------------------ #

    def get_star_context(
        self, user_query: str
    ) -> Tuple[str, List[str], List[str]]:
        """
        Returns
        -------
        (context_buffer, paper_ids, debug_log)
        """
        context_parts: List[str] = []
        paper_ids: Set[str] = set()
        debug_log: List[str] = []

        # A. Entity extraction
        doc = self.nlp(user_query)
        entities = list(
            {e.text for e in doc.ents}
            | {t.text for t in doc if t.pos_ in ("PROPN", "NOUN") and len(t.text) > 2}
        )
        debug_log.append(f"Extracted entities: {entities}")

        # B. Map to graph nodes
        found: Dict[str, List[str]] = {}
        for term in entities:
            hits = self.find_closest_nodes(term)
            if hits:
                found[term] = hits
                debug_log.append(f"  '{term}' → {hits}")

        # C. Gather candidates via edges
        candidates: Set[str] = set()
        for node_ids in found.values():
            for nid in node_ids:
                incoming = self.rels_df[self.rels_df["target_id"] == nid]
                outgoing = self.rels_df[self.rels_df["source_id"] == nid]
                candidates.update(incoming["source_id"].tolist())
                candidates.update(outgoing["target_id"].tolist())
                paper_ids.update(incoming["paper_id"].dropna().tolist())
                paper_ids.update(outgoing["paper_id"].dropna().tolist())

        if not candidates:
            return "No direct graph matches found.", list(paper_ids), debug_log

        # D. Build context buffer
        context_parts.append(
            f"--- GRAPH EVIDENCE ({len(candidates)} candidates) ---"
        )
        for count, res_id in enumerate(candidates):
            if count >= 15 or res_id not in self.node_map:
                continue
            details = self.node_map[res_id]
            connected = self.rels_df[
                (self.rels_df["source_id"] == res_id)
                | (self.rels_df["target_id"] == res_id)
            ]
            paper_ids.update(connected["paper_id"].dropna().tolist())
            conns = list(
                set(connected["target_id"].tolist() + connected["source_id"].tolist())
                - {res_id}
            )
            val = details.get("value", "N/A")
            unit = details.get("unit", "")
            node_type = details.get("label", "Unknown")

            lines = [
                f"CANDIDATE {count + 1} [ID: {res_id}]",
                f"  - TYPE: {node_type}",
                f"  - CONNECTS TO: {', '.join(str(c) for c in conns[:8])}",
            ]
            if str(val) not in ("nan", "N/A"):
                lines.append(f"  - VALUE: {val} {unit}")
            context_parts.append("\n".join(lines))

        debug_log.append(f"Formatted {min(len(candidates), 15)} candidates.")
        return "\n\n".join(context_parts), list(paper_ids), debug_log

    # ------------------------------------------------------------------ #
    # Prompt generation                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_paranoid_prompt(user_query: str, context: str) -> str:
        return f"""
ROLE: You are a precise quantum chemistry assistant answering questions from a knowledge graph.

TASK: Answer the USER QUERY in 2-3 clear sentences using the graph evidence below.

STRICT RULES:
1. Give a direct, concise answer — do NOT list graph connections or traversal paths.
2. If a VALUE is present in the evidence, state it with its unit.
3. Do NOT write "Candidate X connects to Candidate Y" — that is graph structure, not an answer.
4. If the exact answer is not in the evidence, say so in one sentence.
5. Never produce bullet points about graph topology.

GRAPH EVIDENCE:
{context}

USER QUERY: {user_query}

ANSWER (2-3 sentences, direct and factual):"""
