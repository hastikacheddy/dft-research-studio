"""
visualization/graph_visualizer.py
-----------------------------------
Static SVG subgraph rendering (Matplotlib) and interactive HTML graph (Pyvis).
"""

from __future__ import annotations

import io
import random
from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pyvis.network import Network


class GraphVisualizer:
    """Renders static SVG subgraphs and interactive Pyvis HTML graphs."""

    _COLOR_MAP = {
        "Functional": "#FF7F0E",
        "ValidationResult": "#1F77B4",
        "BasisSet": "#2CA02C",
        "Property": "#D62728",
        "Molecule": "#9467BD",
        "Method": "#8C564B",
        "Dataset": "#E377C2",
        "Unit": "#7F7F7F",
        "Other": "#BCBD22",
    }

    def __init__(
        self,
        nx_graph: nx.DiGraph,
        nodes_df: pd.DataFrame,
        rels_df: pd.DataFrame,
    ) -> None:
        self.nx_graph = nx_graph
        self.nodes_df = nodes_df
        self.rels_df = rels_df

    # ------------------------------------------------------------------ #
    # Static SVG                                                           #
    # ------------------------------------------------------------------ #

    def generate_svg(
        self, highlighted_node_ids: Optional[List[str]] = None
    ) -> str:
        """Return an SVG string of the subgraph centred on highlighted nodes."""
        highlighted = set(highlighted_node_ids or [])

        # Build subgraph around highlighted nodes + their 1-hop neighbours
        nodes_to_show: set = set(highlighted)
        for nid in highlighted:
            if nid in self.nx_graph:
                nodes_to_show.update(self.nx_graph.neighbors(nid))
                nodes_to_show.update(self.nx_graph.predecessors(nid))

        if not nodes_to_show:
            if self.nx_graph.number_of_nodes() == 0:
                return "<svg width='200' height='60'><text x='10' y='30'>Empty Graph</text></svg>"
            random.seed(42)
            nodes_to_show = set(
                random.sample(
                    list(self.nx_graph.nodes()),
                    min(20, self.nx_graph.number_of_nodes()),
                )
            )

        sub = self.nx_graph.subgraph(nodes_to_show)
        pos = nx.spring_layout(sub, seed=42, k=0.8)

        node_colors = [
            "red" if nid in highlighted else "skyblue" for nid in sub.nodes()
        ]

        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw_networkx_nodes(sub, pos, node_color=node_colors, node_size=1000, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(sub, pos, width=1.0, alpha=0.5, edge_color="gray", ax=ax)
        nx.draw_networkx_labels(sub, pos, font_size=8, font_color="black", ax=ax)
        ax.axis("off")

        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    # ------------------------------------------------------------------ #
    # Interactive Pyvis                                                    #
    # ------------------------------------------------------------------ #

    def generate_interactive_html(
        self, output_filename: str = "interactive_graph.html"
    ) -> None:
        """Render a force-directed interactive HTML graph with Pyvis."""
        print(
            f"[GraphVisualizer] Building interactive graph "
            f"({self.nx_graph.number_of_nodes()} nodes, "
            f"{self.nx_graph.number_of_edges()} edges) …"
        )
        net = Network(
            notebook=True,
            directed=True,
            cdn_resources="remote",
            height="750px",
            width="100%",
        )

        for node_id, attrs in self.nx_graph.nodes(data=True):
            label = str(attrs.get("node_id", node_id))
            title = f"ID: {node_id}\nType: {attrs.get('label', 'Unknown')}\n"
            for k, v in attrs.items():
                if k not in ("node_id", "label") and pd.notna(v):
                    title += f"{k.replace('_', ' ').title()}: {v}\n"
            node_type = attrs.get("label", "Other")
            color = self._COLOR_MAP.get(node_type, self._COLOR_MAP["Other"])
            size = 15 if node_type == "ValidationResult" else 10
            net.add_node(
                n_id=node_id,
                label=label,
                title=title,
                group=node_type,
                color=color,
                size=size,
            )

        for src, tgt, attrs in self.nx_graph.edges(data=True):
            rel = attrs.get("relationship", "CONNECTS_TO")
            net.add_edge(
                src, tgt,
                title=f"Relationship: {rel}\nPaper: {attrs.get('paper_id', 'N/A')}",
                label=rel,
                length=200,
            )

        net.set_options("""
        var options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -200,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08,
                    "damping": 0.4,
                    "avoidOverlap": 1
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
            }
        }
        """)
        net.show(output_filename)
        print(f"[GraphVisualizer] Saved interactive graph → '{output_filename}'.")
