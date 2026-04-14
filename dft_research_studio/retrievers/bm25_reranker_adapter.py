"""
retrievers/bm25_reranker_adapter.py
------------------------------------
Two-stage hybrid: BM25 broad recall → cross-encoder precision refinement.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from ..config import Config
from .bm25_retriever import BM25Retriever
from .reranker import Reranker


class BM25RerankerAdapter:
    """
    Wraps BM25Retriever + Reranker into a single `.invoke()` interface
    compatible with the ExperimentOrchestrator.
    """

    def __init__(
        self,
        documents: List[Document],
        reranker: Reranker,
        top_k: int | None = None,
    ) -> None:
        cfg = Config()
        self.top_k = top_k if top_k is not None else cfg.top_k_retrieval
        self.bm25 = BM25Retriever(documents)
        self.reranker = reranker
        print(f"[BM25Reranker] Adapter ready for {len(documents)} documents.")

    def invoke(self, query: str) -> List[Document]:
        # Fetch 2× candidates so the reranker has more to work with
        candidates = self.bm25.invoke(query, k=self.top_k * 2)
        if not candidates:
            return []
        reranked = self.reranker.invoke(query, candidates)
        return reranked[: self.top_k]
