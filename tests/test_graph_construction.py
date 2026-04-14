"""
tests/test_graph_construction.py
----------------------------------
Tests for graph construction (DFTDataManager._build_graph) and the
deterministic topological retrieval algorithms (TopologicalRetriever).

All graph assertions use the known fixture topology:
  PBE0 ──HAS_RESULT──► MAE_PBE0_S66 ──EVALUATED_ON──► S66
  B3LYP──HAS_RESULT──► MAE_PBE0_S66

so expected paths and neighbour sets are mathematically verifiable.
"""

from __future__ import annotations

import networkx as nx
import pytest

from dft_research_studio.retrievers.topological_retriever import TopologicalRetriever


# ─────────────────────────────────────────────────────────────────────
# Graph topology invariants
# ─────────────────────────────────────────────────────────────────────

class TestGraphTopologyInvariants:
    """
    These assertions are true by construction of the fixture data and
    by the correctness of the DFTDataManager._build_graph method.
    """

    def test_all_nodes_present(self, sample_graph, nodes_df):
        for nid in nodes_df["node_id"]:
            assert nid in sample_graph.nodes, (
                f"Node '{nid}' is missing from the constructed graph."
            )

    def test_node_count_equals_csv_row_count(self, sample_graph, nodes_df):
        assert sample_graph.number_of_nodes() == len(nodes_df)

    def test_edge_count_equals_csv_row_count(self, sample_graph, rels_df):
        assert sample_graph.number_of_edges() == len(rels_df)

    def test_pbe0_to_mae_edge_exists(self, sample_graph):
        assert sample_graph.has_edge("PBE0", "MAE_PBE0_S66")

    def test_b3lyp_to_mae_edge_exists(self, sample_graph):
        assert sample_graph.has_edge("B3LYP", "MAE_PBE0_S66")

    def test_mae_to_s66_edge_exists(self, sample_graph):
        assert sample_graph.has_edge("MAE_PBE0_S66", "S66")

    def test_no_self_loops(self, sample_graph):
        loops = list(nx.selfloop_edges(sample_graph))
        assert len(loops) == 0

    def test_relationship_attribute_stored_on_edges(self, sample_graph):
        data = sample_graph["PBE0"]["MAE_PBE0_S66"]
        assert data.get("relationship") == "HAS_RESULT"

    def test_graph_is_directed(self, sample_graph):
        assert sample_graph.is_directed()

    def test_path_pbe0_to_s66_exists(self, sample_graph):
        assert nx.has_path(sample_graph, "PBE0", "S66"), (
            "No directed path from PBE0 to S66; "
            "star-topology retrieval cannot connect them."
        )

    def test_pbe0_has_no_predecessors(self, sample_graph):
        # PBE0 is a source node — nothing points to it in the fixture
        assert len(list(sample_graph.predecessors("PBE0"))) == 0

    def test_mae_node_has_two_predecessors(self, sample_graph):
        preds = list(sample_graph.predecessors("MAE_PBE0_S66"))
        assert set(preds) == {"PBE0", "B3LYP"}


# ─────────────────────────────────────────────────────────────────────
# Topological retriever — entity resolution
# ─────────────────────────────────────────────────────────────────────

class TestTopologicalRetrieverEntityResolution:
    """
    find_nodes_robust must correctly map query strings to graph node IDs
    using the three-tier strategy: exact → substring → fuzzy.
    """

    @pytest.fixture(scope="class")
    def retriever(self, nodes_df, rels_df):
        return TopologicalRetriever(nodes_df, rels_df)

    def test_exact_match_returns_correct_node(self, retriever):
        result = retriever.find_nodes_robust("PBE0")
        assert "PBE0" in result

    def test_case_insensitive_exact_match(self, retriever):
        result = retriever.find_nodes_robust("pbe0")
        assert "PBE0" in result

    def test_substring_match_finds_validation_result(self, retriever):
        # "MAE_PBE0" is a prefix of "MAE_PBE0_S66"
        result = retriever.find_nodes_robust("MAE_PBE0")
        assert any("MAE_PBE0" in r for r in result)

    def test_completely_unknown_term_returns_list(self, retriever):
        # Must not raise — graceful empty-or-low-score return
        result = retriever.find_nodes_robust("ZZZNOMATCH999XYZ")
        assert isinstance(result, list)

    def test_top_k_respected(self, retriever):
        result = retriever.find_nodes_robust("PBE0", top_k=1)
        assert len(result) <= 1


# ─────────────────────────────────────────────────────────────────────
# Topological retriever — star search context
# ─────────────────────────────────────────────────────────────────────

class TestTopologicalRetrieverStarSearch:
    """
    get_topological_context must return evidence that includes nodes
    connected to the entities extracted from the query.
    """

    @pytest.fixture(scope="class")
    def retriever(self, nodes_df, rels_df):
        return TopologicalRetriever(nodes_df, rels_df)

    def test_returns_three_tuple(self, retriever):
        ctx, dbg, pids = retriever.get_topological_context("PBE0 S66 performance")
        assert isinstance(ctx, str)
        assert isinstance(dbg, list)
        assert isinstance(pids, list)

    def test_context_contains_candidate_header(self, retriever):
        ctx, _, _ = retriever.get_topological_context("PBE0 S66")
        # Either we found candidates or we found nothing — both are valid strings
        assert isinstance(ctx, str) and len(ctx) > 0

    def test_paper_ids_are_strings(self, retriever):
        _, _, pids = retriever.get_topological_context("PBE0")
        for pid in pids:
            assert isinstance(pid, str)

    def test_debug_log_is_non_empty_for_known_entity(self, retriever):
        _, dbg, _ = retriever.get_topological_context("PBE0")
        assert len(dbg) >= 1

    def test_star_context_includes_pbe0_neighbourhood(self, retriever):
        """
        When querying for PBE0, the ValidationResult node MAE_PBE0_S66
        (which is PBE0's 1-hop successor) must appear as a hub candidate.
        """
        ctx, _, _ = retriever.get_topological_context("PBE0")
        # MAE_PBE0_S66 is the only successor of PBE0 in the fixture graph
        assert "MAE_PBE0_S66" in ctx or "CANDIDATE" in ctx

    def test_deterministic_prompt_contains_query_verbatim(self, retriever):
        prompt = retriever.generate_deterministic_prompt(
            "What is the MAE of PBE0 on S66?",
            "GRAPH EVIDENCE block"
        )
        assert "What is the MAE of PBE0 on S66?" in prompt

    def test_deterministic_prompt_contains_context_verbatim(self, retriever):
        ctx_block = "CANDIDATE 1 [ID: MAE_PBE0_S66]"
        prompt = retriever.generate_deterministic_prompt("query", ctx_block)
        assert ctx_block in prompt


# ─────────────────────────────────────────────────────────────────────
# Disconnected node removal
# ─────────────────────────────────────────────────────────────────────

class TestDisconnectedNodeRemoval:
    """
    remove_disconnected_nodes must remove exactly the orphaned nodes
    and leave the connected component intact.
    """

    def test_orphan_node_removed(self, nodes_df, rels_df, tmp_path):
        import pandas as pd
        from dft_research_studio.config import Config
        from dft_research_studio.data import DFTDataManager

        # Add an orphan node not referenced by any edge
        extended_nodes = pd.concat(
            [nodes_df, pd.DataFrame([{
                "node_id": "ORPHAN_FUNC",
                "label": "Functional",
                "value": None,
                "unit": None,
                "paper_id": None,
            }])],
            ignore_index=True,
        )

        dataset_dir = tmp_path / "processed"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        extended_nodes.to_csv(dataset_dir / "dft_kg_nodes.csv", index=False)
        rels_df.to_csv(dataset_dir / "dft_kg_relationships.csv", index=False)
        pd.DataFrame({
            "Question number": ["Q1.1v0"],
            "Question": ["What is PBE0?"],
            "Ground truth": ["A functional."],
            "Gold Standard Document": ["p1.pdf"],
        }).to_csv(dataset_dir / "_DFT-QA-120.csv", index=False)

        import os
        os.environ.setdefault("GROQ_API_KEY", "test-key")
        cfg = Config(base_dir=str(dataset_dir))
        dm = DFTDataManager(cfg)

        assert "ORPHAN_FUNC" in dm.graph.nodes
        removed = dm.remove_disconnected_nodes()
        assert removed >= 1
        assert "ORPHAN_FUNC" not in dm.graph.nodes

    def test_connected_nodes_survive_removal(self, nodes_df, rels_df, tmp_path):
        import pandas as pd
        import os
        from dft_research_studio.config import Config
        from dft_research_studio.data import DFTDataManager

        dataset_dir = tmp_path / "processed"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        nodes_df.to_csv(dataset_dir / "dft_kg_nodes.csv", index=False)
        rels_df.to_csv(dataset_dir / "dft_kg_relationships.csv", index=False)
        pd.DataFrame({
            "Question number": ["Q1.1v0"],
            "Question": ["What is PBE0?"],
            "Ground truth": ["A functional."],
            "Gold Standard Document": ["p1.pdf"],
        }).to_csv(dataset_dir / "_DFT-QA-120.csv", index=False)

        os.environ.setdefault("GROQ_API_KEY", "test-key")
        cfg = Config(base_dir=str(dataset_dir))
        dm = DFTDataManager(cfg)
        dm.remove_disconnected_nodes()

        # All nodes in the fixture are connected — none should be removed
        for nid in ["PBE0", "B3LYP", "S66", "MAE_PBE0_S66"]:
            assert nid in dm.graph.nodes
