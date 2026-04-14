"""
retrievers/topological_retriever.py
------------------------------------
Deterministic NetworkX-based topological retriever (star search).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Set, Tuple

import networkx as nx
import pandas as pd
import spacy
from thefuzz import fuzz, process, utils as fuzz_utils

from ..config import Config


class TopologicalRetriever:
    """
    Deterministic graph retrieval using NetworkX.
    Multi-tier entity resolution: exact → substring → fuzzy.
    """

    def __init__(self, nodes_df: pd.DataFrame, rels_df: pd.DataFrame) -> None:
        self.nodes_df = nodes_df
        self.rels_df = rels_df

        self.G: nx.DiGraph = nx.DiGraph()
        for _, row in nodes_df.iterrows():
            self.G.add_node(str(row["node_id"]), **row.dropna().to_dict())
        for _, row in rels_df.iterrows():
            if pd.notna(row["source_id"]) and pd.notna(row["target_id"]):
                self.G.add_edge(
                    str(row["source_id"]),
                    str(row["target_id"]),
                    relationship=row.get("relationship_type", "related"),
                )

        print(
            f"[TopologicalRetriever] Graph: "
            f"{self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges."
        )

        self.node_lookup: Dict[str, str] = {
            str(nid).lower(): str(nid)
            for nid in nodes_df["node_id"].unique()
        }
        self.all_node_ids: List[str] = list(self.node_lookup.values())

        # Lazy-load SpaCy so the model isn't re-downloaded on each call
        self._nlp = None

    # ------------------------------------------------------------------ #

    @property
    def nlp(self) -> spacy.Language:
        if self._nlp is None:
            model = Config().spacy_model
            try:
                self._nlp = spacy.load(model)
            except OSError:
                from spacy.cli import download
                download(model)
                self._nlp = spacy.load(model)
        return self._nlp

    # ------------------------------------------------------------------ #
    # Entity resolution                                                    #
    # ------------------------------------------------------------------ #

    def find_nodes_robust(self, term: str, top_k: int = 5) -> List[str]:
        """Exact → Substring → Fuzzy (≥ 80 %) match."""
        term_lower = term.lower()
        candidates: Set[str] = set()

        # Exact
        if term_lower in self.node_lookup:
            candidates.add(self.node_lookup[term_lower])

        # Substring
        for nid in self.all_node_ids:
            if term_lower in nid.lower():
                candidates.add(nid)

        # Fuzzy fallback
        if len(candidates) < top_k:
            matches = process.extract(
                term_lower,
                self.all_node_ids,
                limit=top_k,
                scorer=fuzz.token_sort_ratio,
                processor=fuzz_utils._default_process,
            )
            candidates.update(m[0] for m in matches if m[1] > 80)

        return list(candidates)[:top_k]

    # ------------------------------------------------------------------ #
    # Star retrieval                                                        #
    # ------------------------------------------------------------------ #

    def get_topological_context(
        self, user_query: str
    ) -> Tuple[str, List[str], List[str]]:
        """
        Returns
        -------
        (context_buffer, debug_log, paper_ids)
        """
        debug_log: List[str] = []
        paper_ids: Set[str] = set()

        # A. NLP entity extraction
        doc = self.nlp(user_query)
        terms = list(
            {
                t.text
                for t in doc
                if t.pos_ in ("PROPN", "NOUN") or t.shape_.isupper()
            }
        )
        debug_log.append(f"Extracted terms: {terms}")

        # B. Map terms → nodes
        term_map: Dict[str, List[str]] = {}
        for t in terms:
            nodes = self.find_nodes_robust(t)
            if nodes:
                term_map[t] = nodes
                debug_log.append(f"  '{t}' → {nodes}")

        if not term_map:
            return "No graph entities found.", debug_log, []

        # C. 1-hop star search
        all_entity_nodes = [n for ns in term_map.values() for n in ns]
        hub_candidates: List[str] = []

        for entity in all_entity_nodes:
            if entity not in self.G:
                continue
            neighbors = list(self.G.successors(entity)) + list(self.G.predecessors(entity))
            hub_candidates.extend(neighbors)

            for _, _, data in self.G.edges(entity, data=True):
                if "paper_id" in data and pd.notna(data["paper_id"]):
                    paper_ids.add(data["paper_id"])
            for _, _, data in self.G.in_edges(entity, data=True):
                if "paper_id" in data and pd.notna(data["paper_id"]):
                    paper_ids.add(data["paper_id"])

        sorted_hubs = [h for h, _ in Counter(hub_candidates).most_common(15)]

        # D. Build context
        lines = [f"--- GRAPH EVIDENCE ({len(sorted_hubs)} hubs found) ---"]
        for i, hub_id in enumerate(sorted_hubs):
            attrs = self.G.nodes.get(hub_id, {})
            neighbors = (
                list(self.G.successors(hub_id)) + list(self.G.predecessors(hub_id))
            )
            short_neighbors = [n for n in neighbors if len(n) < 50]
            val = attrs.get("value", "N/A")
            node_type = attrs.get("label", "Node")

            chunk = [
                f"CANDIDATE {i + 1} [ID: {hub_id}]",
                f"  - TYPE: {node_type}",
                f"  - CONNECTS TO: {', '.join(short_neighbors[:8])}",
            ]
            if str(val) not in ("nan", "N/A"):
                chunk.append(f"  - VALUE: {val} {attrs.get('unit', '')}")
            lines.append("\n".join(chunk))

            for _, _, data in self.G.edges(hub_id, data=True):
                if "paper_id" in data and pd.notna(data["paper_id"]):
                    paper_ids.add(data["paper_id"])
            for _, _, data in self.G.in_edges(hub_id, data=True):
                if "paper_id" in data and pd.notna(data["paper_id"]):
                    paper_ids.add(data["paper_id"])

        return "\n\n".join(lines), debug_log, list(paper_ids)

    # ------------------------------------------------------------------ #
    # Prompt generation                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_deterministic_prompt(user_query: str, context: str) -> str:
        return f"""
ROLE: You are a highly precise Quantum Chemistry Fact Extractor.
TASK: Answer the user query using ONLY the provided Graph Evidence.

CRITICAL PROTOCOL:
1. The evidence presents CANDIDATE hubs with TYPE, VALUE (if any), and CONNECTIONS.
2. Identify the candidate(s) most relevant to the USER QUERY.
3. Synthesize a concise answer directly from the accepted candidate facts.
4. If the information is not present, state: "The requested information cannot
   be found in the provided knowledge graph evidence."
5. Do NOT infer, speculate, or use external knowledge.

INPUT CONTEXT:
{context}

USER QUERY:
{user_query}

FINAL ANSWER:
"""
