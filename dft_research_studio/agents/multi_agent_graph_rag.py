"""
agents/multi_agent_graph_rag.py
--------------------------------
Strategist → Graph retrieval → Critic → optional Vector fallback workflow.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .llm_wrapper import LLMWrapper


class MultiAgentGraphRAG:
    """
    Two-agent pipeline:
      1. Strategist decomposes the query into chemical search terms.
      2. GraphRAG retrieves star-context for each term.
      3. Critic scores context sufficiency (0 or 1).
      4. If critic rejects, optionally falls back to the vector retriever.
    """

    def __init__(
        self,
        graph_engine: Any,           # GraphRAG instance
        vector_retriever: Any = None, # LangChain retriever
        llm_client: LLMWrapper | None = None,
    ) -> None:
        self.graph = graph_engine
        self.vector_retriever = vector_retriever
        self.llm = llm_client

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json_list(text: str) -> List[str]:
        """Extract a JSON string list from LLM output."""
        try:
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return [
                    str(item.get("entity", item) if isinstance(item, dict) else item)
                    for item in data
                ]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------ #
    # Agents                                                               #
    # ------------------------------------------------------------------ #

    def agent_strategist(self, question: str) -> List[str]:
        prompt = (
            f'TASK: Research planner for DFT.\n'
            f'QUERY: "{question}"\n'
            f'ACTION: Identify 3-4 specific chemical entities (functionals, datasets).\n'
            f'Output ONLY a JSON list of strings, e.g. ["PBE0", "S66 dataset"]'
        )
        response, *_ = self.llm.generate(prompt)
        plan = self._parse_json_list(response)
        if not plan:
            return [str(question)]
        return [str(p) for p in plan[:4]]

    def agent_critic(self, question: str, context: str) -> int:
        prompt = (
            f"TASK: Rate context sufficiency.\n"
            f"Question: {question}\n"
            f"Context snippet: {context[:1000]}\n"
            f"Output strictly one number: 1 (sufficient) or 0 (insufficient)."
        )
        response, *_ = self.llm.generate(prompt)
        return 1 if "1" in str(response) else 0

    # ------------------------------------------------------------------ #
    # Main workflow                                                        #
    # ------------------------------------------------------------------ #

    def run_workflow(
        self, question: str
    ) -> Tuple[str, Dict, List[str], List[str], int, int, float, int]:
        """
        Returns
        -------
        (answer, log, text_chunks, paper_ids, input_tokens,
         output_tokens, cost_usd, api_calls)
        """
        log: Dict = {"steps": []}
        text_chunks: List[str] = []
        paper_ids: Set[str] = set()

        search_terms = self.agent_strategist(question)
        log["steps"].append(f"Strategist plan: {search_terms}")

        graph_parts: List[str] = []
        for term in search_terms:
            ctx_str, term_ids, _ = self.graph.get_star_context(str(term))
            if ctx_str and ctx_str != "No direct graph matches found.":
                graph_parts.append(ctx_str)
                paper_ids.update(term_ids)

        final_context = "\n\n---\n\n".join(graph_parts) if graph_parts else "No graph context found."
        final_context = final_context[:2500]

        score = self.agent_critic(question, final_context)
        log["steps"].append(f"Critic score: {score}")

        if score == 0 and self.vector_retriever is not None:
            log["steps"].append("Critic rejected graph → activating vector fallback.")
            vector_docs = self.vector_retriever.invoke(question)
            vector_text = "\n".join(d.page_content for d in vector_docs)
            final_context = (final_context + "\n\n--- FALLBACK VECTOR ---\n" + vector_text)[:4000]
            text_chunks.append(vector_text)

        final_prompt = self.graph.generate_paranoid_prompt(question, final_context)
        answer, it, ot, tt, cost, calls = self.llm.generate(final_prompt)

        return answer, log, text_chunks, list(paper_ids), it, ot, cost, calls
