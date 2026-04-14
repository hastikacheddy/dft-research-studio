"""
visualization/trace_visualizer.py
-----------------------------------
Per-question topological trace SVG rendering and HTML card grid.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from IPython.display import HTML, display


class TraceVisualizer:
    """Generates SVG topological traces from experiment results."""

    def __init__(
        self,
        results_path: str = "debug_results_raw.json",
        metrics_path: str = "final_experiment_metrics.csv",
    ) -> None:
        self.results: List[Dict] = []
        if os.path.exists(results_path):
            with open(results_path) as f:
                self.results = json.load(f)
        else:
            print(f"[TraceVisualizer] Warning: {results_path} not found.")

        self.metrics = (
            pd.read_csv(metrics_path)
            if os.path.exists(metrics_path)
            else pd.DataFrame()
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_str(obj: Any) -> str:
        if isinstance(obj, dict):
            return obj.get("final_answer") or obj.get("answer") or str(obj)
        if isinstance(obj, (list, tuple)):
            return " ".join(str(x) for x in obj if x is not None)
        return str(obj) if obj is not None else ""

    def generate_trace_report(
        self, question_id: str
    ) -> Tuple[Optional[str], str]:
        """Return (svg_string_or_None, trace_html_note)."""
        entry = None
        for exp in ("MultiAgent", "GraphRAG", "GraphDeterministic"):
            for r in self.results:
                if (
                    r.get("question_id") == question_id
                    and exp in r.get("experiment_type", "")
                ):
                    entry = r
                    break
            if entry:
                break

        if not entry:
            return None, f"No entry found for {question_id}."

        # Extract context
        if "MultiAgent" in entry.get("experiment_type", ""):
            chunks = entry.get("retrieved_text_chunks", [])
            context = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
        else:
            context = str(entry.get("context_used", ""))

        # Parse accepted node IDs from answer
        answer_str = self._to_str(entry.get("generated_answer", ""))
        id_re = re.compile(r"ID:\s*([\w\d_-]+)")
        accepted: Set[str] = {m.group(1).strip() for m in id_re.finditer(answer_str)}

        if accepted:
            trace_note = f"<b>Path:</b> {' &rarr; '.join(sorted(accepted))}"
        else:
            trace_note = "<b>Path:</b> No accepted nodes parsed from answer."

        svg = self._render_svg(context, question_id, accepted)
        return svg, trace_note

    def _render_svg(
        self,
        context: str,
        qid: str,
        accepted: Set[str],
    ) -> Optional[str]:
        G = nx.DiGraph()
        G.add_node("ROOT", color="#E74C3C", label="Query\nContext")

        pattern = re.compile(
            r"CANDIDATE \d+ \[ID: (.*?)\]\n\s+- TYPE: (.*?)\n\s+- CONNECTS TO: (.*?)(?:\n|$)"
        )
        for i, m in enumerate(pattern.finditer(context)):
            if i >= 15:
                break
            hub_id, node_type, conns_str = (
                m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            )
            is_accepted = any(
                hub_id.lower() == a.lower() for a in accepted
            )
            color = "#F1C40F" if is_accepted else "#3498DB"
            G.add_node(hub_id, color=color, label=textwrap.fill(hub_id, 15))
            G.add_edge("ROOT", hub_id)
            for conn in [c.strip() for c in conns_str.split(",") if c.strip()]:
                if not G.has_node(conn):
                    G.add_node(conn, color="#2ECC71", label=textwrap.fill(conn, 15))
                G.add_edge(hub_id, conn)

        if G.number_of_nodes() <= 1:
            return None

        pos = nx.spring_layout(G, k=3.5, iterations=300, seed=42)
        node_colors = [G.nodes[n].get("color", "#95A5A6") for n in G.nodes]
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes}

        edge_colors, widths = [], []
        for u, v in G.edges():
            gold = any(u.lower() == a.lower() or v.lower() == a.lower() for a in accepted)
            edge_colors.append("#F1C40F" if gold else "#BDC3C7")
            widths.append(2.0 if gold else 0.5)

        fig, ax = plt.subplots(figsize=(6, 5))
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color=node_colors,
                               edgecolors="#333", linewidths=0.5, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=widths,
                               arrowsize=10, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6,
                                font_weight="bold", ax=ax)
        ax.set_title(f"QID: {qid}", fontsize=9, fontweight="bold")
        ax.axis("off")

        tmp = f"_trace_{qid}.svg"
        fig.savefig(tmp, format="svg", bbox_inches="tight")
        plt.close(fig)
        with open(tmp) as f:
            svg_content = f.read()
        os.remove(tmp)
        return svg_content

    # ------------------------------------------------------------------ #

    def render_grid(self, question_ids: List[str]) -> None:
        """Display a 2-column HTML card grid for multiple question traces."""
        cards: List[str] = []
        for qid in question_ids:
            svg, note = self.generate_trace_report(qid)
            body = svg if svg else "<p>No graph data available.</p>"
            cards.append(
                f"""
                <div style="border:1px solid #ddd;padding:10px;border-radius:8px;
                            background:#fff;box-shadow:2px 2px 5px rgba(0,0,0,.1);">
                  <h4 style="margin:0 0 5px 0;">{qid}</h4>
                  <div style="font-size:11px;color:#444;margin-bottom:10px;
                              font-family:monospace;">{note}</div>
                  <div>{body}</div>
                </div>
                """
            )
        grid = (
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;">'
            + "".join(cards)
            + "</div>"
        )
        display(HTML(grid))
