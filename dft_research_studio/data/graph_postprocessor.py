"""
data/graph_postprocessor.py
------------------------------
Post-processing pipeline for the Auto-KGR knowledge graph.
Implements the logic from notebooks 03 (normalization) and 04 (topological enhancement).

Pipeline:
    1. Deduplication (notebook 03)
    2. Node attribute completion — rung/class/family (notebook 04, 1a)
    3. Relationship type refinement (notebook 04, 1b)
    4. Entity resolution — fuzzy dedup (notebook 04, 3)
    5. Link prediction — Jaccard proxy (notebook 04, 2)

Usage:
    from dft_research_studio.data.graph_postprocessor import GraphPostProcessor
    gpp = GraphPostProcessor(nodes_df, rels_df)
    nodes_clean, rels_clean = gpp.run_full_pipeline()
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class GraphPostProcessor:
    """
    Post-processing pipeline for KG nodes and relationships.
    Mirrors the logic from Archive_Extraction_Logic notebooks 03 and 04.
    """

    def __init__(self, nodes_df: pd.DataFrame, rels_df: pd.DataFrame):
        self.nodes = nodes_df.copy()
        self.rels = rels_df.copy()
        self._stats: Dict[str, int] = {}

    # ─────────────────────────────────────────────────────────────
    # Step 1: Deduplication (Notebook 03)
    # ─────────────────────────────────────────────────────────────
    def deduplicate(self) -> None:
        """Remove exact duplicate rows and duplicate node_ids."""
        orig_nodes = len(self.nodes)
        orig_rels = len(self.rels)

        # Deduplicate relationships (exact triple match)
        self.rels = self.rels.drop_duplicates()

        # Deduplicate nodes (keep first occurrence per node_id)
        self.nodes = self.nodes.drop_duplicates(subset=["node_id"], keep="first")

        self._stats["dedup_nodes_removed"] = orig_nodes - len(self.nodes)
        self._stats["dedup_rels_removed"] = orig_rels - len(self.rels)
        logger.info(
            "[PostProcess] Deduplication: removed %d duplicate nodes, %d duplicate rels.",
            self._stats["dedup_nodes_removed"],
            self._stats["dedup_rels_removed"],
        )

    # ─────────────────────────────────────────────────────────────
    # Step 2: Node Attribute Completion (Notebook 04, 1a)
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _determine_metadata(node_id: str) -> Tuple[str, str, str]:
        """
        Dictionary-based heuristic injection for rung/class/family.
        Maps known substrings in node_id to Jacob's Ladder classification.
        """
        nid = str(node_id).upper()

        rung = "Unknown"
        cls = "Unknown"
        family = "Unknown"

        # Jacob's Ladder — Rung classification
        if any(x in nid for x in ["LDA", "SVWN"]):
            rung, cls = "1", "LDA"
        elif any(x in nid for x in ["PBE", "BLYP", "BP86", "B97", "PW91"]) and "0" not in nid and "3" not in nid:
            rung, cls = "2", "GGA"
        elif any(x in nid for x in ["TPSS", "M06-L", "M06L", "MN12-L", "MN15-L", "SCAN", "R2SCAN", "VSXC", "M11-L"]):
            rung, cls = "3", "meta-GGA"
        elif any(x in nid for x in ["B3LYP", "PBE0", "M06-2X", "M062X", "M06-HF", "M08", "M11",
                                      "MN15", "CAM-B3LYP", "WB97", "ΩB97", "HSE", "LC-"]):
            rung, cls = "4", "Hybrid"
        elif any(x in nid for x in ["B2PLYP", "DSD-", "PWPB95", "B2GP", "XYG3"]):
            rung, cls = "5", "Double-Hybrid"

        # Family classification
        if any(x in nid for x in ["M06", "M05", "M08", "M11", "MN12", "MN15"]):
            family = "Minnesota"
        elif any(x in nid for x in ["WB97", "ΩB97", "CAM-"]):
            family = "Range-Separated"
        elif any(x in nid for x in ["D3", "D4", "DFT-D"]):
            family = "Dispersion-Corrected"
        elif any(x in nid for x in ["SCAN", "R2SCAN"]):
            family = "SCAN"
        elif any(x in nid for x in ["B3LYP", "BLYP", "B97"]):
            family = "Becke"
        elif any(x in nid for x in ["PBE", "RPBE", "REVPBE"]):
            family = "PBE"

        return rung, cls, family

    def complete_node_attributes(self) -> None:
        """Fill missing rung, class, family attributes for Functional/Method nodes."""
        filled = 0
        for idx, row in self.nodes.iterrows():
            if row.get("label") not in ("Functional", "Method"):
                continue
            # Only fill if missing
            if pd.notna(row.get("rung")) and row.get("rung") != "Unknown":
                continue
            rung, cls, family = self._determine_metadata(row["node_id"])
            if rung != "Unknown":
                self.nodes.at[idx, "rung"] = rung
                self.nodes.at[idx, "class"] = cls
                self.nodes.at[idx, "family"] = family
                filled += 1

        self._stats["attributes_filled"] = filled
        logger.info("[PostProcess] Node attribute completion: filled %d nodes.", filled)

    # ─────────────────────────────────────────────────────────────
    # Step 3: Relationship Type Refinement (Notebook 04, 1b)
    # ─────────────────────────────────────────────────────────────
    def refine_relationships(self) -> None:
        """Replace generic relationship types with semantically specific ones."""
        refined = 0

        # Basis set relations
        mask_basis = self.rels["target_id"].str.contains(
            "TZVP|QZVP|AUG-CC|DEF2|6-31|CC-PV", case=False, na=False
        )
        if mask_basis.any():
            self.rels.loc[mask_basis, "relationship_type"] = "USES_BASIS_SET"
            refined += mask_basis.sum()

        # Dispersion correction relations
        mask_disp = self.rels["target_id"].str.contains("D3|D4|MBD|DFT-D", case=False, na=False) & \
                    self.rels["relationship_type"].isin(["REFINES", "CITES", "APPLIES"])
        if mask_disp.any():
            self.rels.loc[mask_disp, "relationship_type"] = "REFINES_DISPERSION_CORRECTION"
            refined += mask_disp.sum()

        # XC Kernel relations (functional refinement)
        mask_xc = self.rels["relationship_type"].isin(["REFINES", "EXTENDS"]) & \
                  ~mask_disp & \
                  self.rels["target_id"].str.contains(
                      "B3LYP|PBE|M06|M05|SCAN|TPSS|B97|BLYP|HSE", case=False, na=False
                  )
        if mask_xc.any():
            self.rels.loc[mask_xc, "relationship_type"] = "REFINES_XC_KERNEL"
            refined += mask_xc.sum()

        # Benchmark evaluation relations
        mask_bench = self.rels["target_id"].str.contains(
            "GMTKN|S22|S66|NCIE|BH76|DBH|W4|G2|MGAE|HTBH|NHTBH", case=False, na=False
        ) & self.rels["relationship_type"].isin(["CITES", "EVALUATES", "TESTED_ON"])
        if mask_bench.any():
            self.rels.loc[mask_bench, "relationship_type"] = "VALIDATED_ON"
            refined += mask_bench.sum()

        # Failure mode relations
        mask_fail = self.rels["source_id"].str.contains("FAIL|ERROR|POOR", case=False, na=False) | \
                    self.rels["target_id"].str.contains("FAIL|ERROR|POOR", case=False, na=False)
        if mask_fail.any():
            self.rels.loc[mask_fail, "relationship_type"] = "FAILS_ON"
            refined += mask_fail.sum()

        self._stats["rels_refined"] = refined
        logger.info("[PostProcess] Relationship refinement: refined %d edges.", refined)

    # ─────────────────────────────────────────────────────────────
    # Step 4: Entity Resolution (Notebook 04, 3)
    # ─────────────────────────────────────────────────────────────
    def resolve_entities(self, threshold: float = 0.9) -> None:
        """
        Fuzzy deduplication of node_ids using Ratcliff/Obershelp similarity.
        Merges near-duplicate identifiers (e.g., B3LYP vs B3-LYP).
        """
        ids = sorted(self.nodes["node_id"].astype(str).tolist())
        merge_map: Dict[str, str] = {}

        for i in range(len(ids) - 1):
            curr = ids[i]
            next_id = ids[i + 1]

            sim = SequenceMatcher(None, curr, next_id).ratio()

            if sim > threshold and abs(len(curr) - len(next_id)) <= 3:
                # Keep the shorter/canonical form
                canonical = curr if len(curr) <= len(next_id) else next_id
                duplicate = next_id if canonical == curr else curr
                merge_map[duplicate] = canonical

        if not merge_map:
            self._stats["entities_resolved"] = 0
            logger.info("[PostProcess] Entity resolution: no duplicates found.")
            return

        # Apply merge map to nodes
        self.nodes["node_id"] = self.nodes["node_id"].replace(merge_map)
        self.nodes = self.nodes.drop_duplicates(subset=["node_id"], keep="first")

        # Apply merge map to relationships
        self.rels["source_id"] = self.rels["source_id"].replace(merge_map)
        self.rels["target_id"] = self.rels["target_id"].replace(merge_map)
        self.rels = self.rels.drop_duplicates()

        self._stats["entities_resolved"] = len(merge_map)
        logger.info(
            "[PostProcess] Entity resolution: merged %d duplicate entities. Examples: %s",
            len(merge_map),
            list(merge_map.items())[:5],
        )

    # ─────────────────────────────────────────────────────────────
    # Step 5: Link Prediction — Jaccard Proxy (Notebook 04, 2)
    # ─────────────────────────────────────────────────────────────
    def predict_links(self, jaccard_threshold: float = 0.3, max_predictions: int = 50) -> None:
        """
        Generate plausible unobserved relationships using Jaccard similarity.
        If two benchmarks share >30% of their evaluated functionals, link them.
        """
        # Build adjacency
        neighbors: Dict[str, Set[str]] = {}
        for _, row in self.rels.iterrows():
            s, t = str(row["source_id"]), str(row["target_id"])
            neighbors.setdefault(s, set()).add(t)
            neighbors.setdefault(t, set()).add(s)

        # Find benchmark/dataset nodes
        benchmark_ids = set(
            self.nodes[self.nodes["label"].isin(["Benchmark", "BenchmarkSet", "Dataset"])]["node_id"].tolist()
        )

        existing_edges = set(
            zip(self.rels["source_id"].astype(str), self.rels["target_id"].astype(str))
        )

        predicted = []
        benchmark_list = sorted(benchmark_ids & set(neighbors.keys()))

        for i, b1 in enumerate(benchmark_list):
            for b2 in benchmark_list[i + 1:]:
                if (b1, b2) in existing_edges or (b2, b1) in existing_edges:
                    continue
                n1, n2 = neighbors.get(b1, set()), neighbors.get(b2, set())
                if not n1 or not n2:
                    continue
                jaccard = len(n1 & n2) / len(n1 | n2)
                if jaccard >= jaccard_threshold:
                    predicted.append({
                        "paper_id": "AutoKGR_LinkPred",
                        "source_id": b1,
                        "target_id": b2,
                        "relationship_type": "SHARES_EVALUATION_CONTEXT",
                        "condition": f"jaccard={jaccard:.2f}",
                    })
                    if len(predicted) >= max_predictions:
                        break
            if len(predicted) >= max_predictions:
                break

        if predicted:
            self.rels = pd.concat([self.rels, pd.DataFrame(predicted)], ignore_index=True)

        self._stats["links_predicted"] = len(predicted)
        logger.info("[PostProcess] Link prediction: added %d predicted edges.", len(predicted))

    # ─────────────────────────────────────────────────────────────
    # Step 6: Ghost node repair
    # ─────────────────────────────────────────────────────────────
    def repair_ghost_nodes(self) -> None:
        """Remove relationships pointing to non-existent nodes."""
        valid_ids = set(self.nodes["node_id"].tolist())
        before = len(self.rels)

        mask = self.rels["source_id"].isin(valid_ids) & self.rels["target_id"].isin(valid_ids)
        self.rels = self.rels[mask]

        removed = before - len(self.rels)
        self._stats["ghost_rels_removed"] = removed
        if removed > 0:
            logger.info("[PostProcess] Ghost repair: removed %d orphan relationships.", removed)

    # ─────────────────────────────────────────────────────────────
    # Full pipeline
    # ─────────────────────────────────────────────────────────────
    def run_full_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete post-processing pipeline:
        1. Deduplication
        2. Node attribute completion
        3. Relationship refinement
        4. Entity resolution
        5. Link prediction
        6. Ghost node repair

        Returns (nodes_df, rels_df)
        """
        logger.info("[PostProcess] Starting full pipeline on %d nodes, %d rels.",
                    len(self.nodes), len(self.rels))

        self.deduplicate()
        self.complete_node_attributes()
        self.refine_relationships()
        self.resolve_entities()
        self.predict_links()
        self.repair_ghost_nodes()

        logger.info(
            "[PostProcess] Pipeline complete. Final: %d nodes, %d rels. Stats: %s",
            len(self.nodes), len(self.rels), self._stats,
        )
        return self.nodes, self.rels

    def get_summary(self) -> str:
        """Human-readable summary of post-processing."""
        lines = [
            "=" * 56,
            "📊 GRAPH POST-PROCESSING SUMMARY",
            "=" * 56,
            f"  Duplicate nodes removed:    {self._stats.get('dedup_nodes_removed', 0)}",
            f"  Duplicate rels removed:     {self._stats.get('dedup_rels_removed', 0)}",
            f"  Node attributes filled:     {self._stats.get('attributes_filled', 0)}",
            f"  Relationships refined:      {self._stats.get('rels_refined', 0)}",
            f"  Entities resolved (merged): {self._stats.get('entities_resolved', 0)}",
            f"  Links predicted (Jaccard):  {self._stats.get('links_predicted', 0)}",
            f"  Ghost rels removed:         {self._stats.get('ghost_rels_removed', 0)}",
            f"  Final nodes:                {len(self.nodes)}",
            f"  Final relationships:        {len(self.rels)}",
            "=" * 56,
        ]
        return "\n".join(lines)
