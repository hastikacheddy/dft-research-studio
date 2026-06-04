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
        """Match *term* against node IDs — exact first, then fuzzy (≥ 70%)."""
        all_ids = [str(x) for x in self.nodes_df["node_id"].unique()]
        # 1. Exact match (case-insensitive)
        exact = [nid for nid in all_ids if nid.upper() == term.upper()]
        if exact:
            return exact[:top_k]
        # 2. Substring match (M06-L in "VR_M06-L_S66")
        substring = [nid for nid in all_ids if term.upper() in nid.upper() 
                      and len(term) > 2]
        if substring:
            # Sort by length (shorter = more specific match)
            substring.sort(key=len)
            return substring[:top_k]
        # 3. Fuzzy match as fallback
        matches = process.extract(
            term, all_ids, limit=top_k, scorer=fuzz.token_sort_ratio
        )
        return [m[0] for m in matches if m[1] > 70]

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

        # A. Entity extraction — DFT-specific patterns first, then SpaCy
        import re as _re
        # DFT functional/benchmark patterns (preserves hyphens and version numbers)
        dft_pattern = (
            r"\b(M06-?L|M06-?2X|M06-?HF|M06|M05-?2X|M05|M08-?HX|M08-?SO|M11-?L|M11|"
            r"MN15-?L|MN15|MN12-?L|MN12|B3LYP|PBE0|PBE|TPSS|SCAN|R2SCAN|HSE06|"
            r"CAM-B3LYP|[wω]B97X?-?[VD234]*|B2PLYP|DSD-PBEP86|BLYP|BP86|PW91|B97|"
            r"PWPB95|XYG3|SVWN|LDA|revTPSS|"
            r"S22|S66|GMTKN55|GMTKN30|BH76|W4-?11|NCIE|TMC32|DBH24|"
            r"HTBH|NHTBH|DARC|BSR36|ISO34|DC13|IDISP|PCONF21|"
            r"D3\(?BJ\)?|D4|DFT-D[34]|MBD|VV10|NL|"
            r"def2-[A-Za-z]+|aug-cc-pV[DTQ5]Z|cc-pV[DTQ5]Z)\b"
        )
        dft_entities = set(_re.findall(dft_pattern, user_query, _re.IGNORECASE))
        # SpaCy for remaining entities
        doc = self.nlp(user_query)
        spacy_entities = (
            {e.text for e in doc.ents}
            | {t.text for t in doc if t.pos_ in ("PROPN", "NOUN") and len(t.text) > 2}
        )
        # DFT-specific takes priority
        entities = list(dft_entities | spacy_entities)
        debug_log.append(f"Extracted entities: {entities}")

        # B. Map to graph nodes
        found: Dict[str, List[str]] = {}
        for term in entities:
            hits = self.find_closest_nodes(term)
            if hits:
                found[term] = hits
                debug_log.append(f"  '{term}' → {hits}")

        # C. Gather candidates via edges (with hub filtering)
        _SKIP_HUBS = {"Global_Metadata_Hub", "Density Functional Theory", "DFT",
                       "Hub", "Metadata", "GMTKN55", "GMTKN30"}
        candidates: Set[str] = set()
        # Add the matched nodes themselves first
        for node_ids in found.values():
            for nid in node_ids:
                candidates.add(nid)
                incoming = self.rels_df[self.rels_df["target_id"] == nid]
                outgoing = self.rels_df[self.rels_df["source_id"] == nid]
                # Only add neighbors that aren't massive hubs
                for neighbor in incoming["source_id"].tolist() + outgoing["target_id"].tolist():
                    if neighbor not in _SKIP_HUBS:
                        candidates.add(neighbor)
                paper_ids.update(incoming["paper_id"].dropna().tolist())
                paper_ids.update(outgoing["paper_id"].dropna().tolist())
        # Cap at 50 most relevant candidates
        if len(candidates) > 50:
            # Prioritize: matched nodes first, then VR nodes, then others
            matched = set()
            for node_ids in found.values():
                matched.update(node_ids)
            vr_nodes = {c for c in candidates if c.startswith("VR_")}
            others = candidates - matched - vr_nodes
            candidates = matched | set(list(vr_nodes)[:30]) | set(list(others)[:20-len(matched)])

        if not candidates:
            return "No direct graph matches found.", list(paper_ids), debug_log

        # D. Build context buffer
        context_parts.append(
            f"--- GRAPH EVIDENCE ({len(candidates)} candidates) ---"
        )
        for count, res_id in enumerate(candidates):
            if count >= 10 or res_id not in self.node_map:
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

            # Get relationship types for richer context
            rel_details = []
            for _, rel in connected.iterrows():
                if rel["source_id"] == res_id:
                    rel_details.append(f"{rel['relationship_type']} → {rel['target_id']}")
                else:
                    rel_details.append(f"{rel['source_id']} → {rel['relationship_type']} → {res_id}")
            # Collect VALUES from connected VR_ nodes (the key enrichment)
            vr_facts = []
            for conn_id in conns[:10]:
                if str(conn_id).startswith("VR_") and conn_id in self.node_map:
                    vr_detail = self.node_map[conn_id]
                    vr_val = vr_detail.get("value", "N/A")
                    vr_unit = vr_detail.get("unit", "")
                    if str(vr_val) not in ("nan", "N/A", "None", ""):
                        # Parse VR node ID for benchmark and metric info
                        vr_parts = str(conn_id).replace("VR_", "").split("_")
                        vr_facts.append(f"{conn_id} = {vr_val} {vr_unit}")
            # Also get rung/class/family for Functional nodes
            rung = details.get("rung", "")
            cls = details.get("class", "")
            family = details.get("family", "")
            classification = ""
            if str(rung) not in ("nan", "", "None", "Unknown"):
                classification = f"Rung {rung} ({cls})"
                if str(family) not in ("nan", "", "None", "Unknown"):
                    classification += f", {family} family"
            lines = [
                f"CANDIDATE {count + 1} [ID: {res_id}]",
                f"  - TYPE: {node_type}",
            ]
            if classification:
                lines.append(f"  - CLASSIFICATION: {classification}")
            lines.append(f"  - RELATIONSHIPS: {'; '.join(rel_details[:6])}")
            if vr_facts:
                lines.append(f"  - QUANTITATIVE DATA: {'; '.join(vr_facts[:5])}")
            lines.append(f"  - ALSO CONNECTS TO: {', '.join(str(c) for c in conns[:5])}")
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
        return f"""ROLE: You are a precise quantum chemistry assistant. You MUST answer ONLY from the evidence provided below.
TASK: Answer the USER QUERY using ONLY the graph evidence below. Do NOT use your own knowledge.
STRICT RULES:
1. Answer the question using the GRAPH EVIDENCE below. Interpret relationship names as facts (e.g., "RESULT_FOR_METHOD → M06-L" means the result was measured for M06-L; "RECOMMENDED_FOR → Noncovalent_Interactions" means it is recommended for noncovalent interactions).
2. Cite supporting evidence with [CANDIDATE X] references where possible.
3. If a VALUE appears, cite it exactly with its unit.
4. You MAY compare entities by examining their different connections and values in the evidence.
5. Base your answer primarily on the evidence. If you must add brief context to make the answer coherent, clearly distinguish it from evidence-based claims.
6. If the evidence is genuinely irrelevant, say so — but first check if relationships or values can answer the question indirectly.
7. Give a direct, informative answer in 2-5 sentences.
GRAPH EVIDENCE:
{context}
USER QUERY: {user_query}
ANSWER (cite evidence node IDs, do NOT use external knowledge):"""
