"""
retrievers/reranker.py
----------------------
Cross-encoder neural reranker.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from ..config import Config


class Reranker:
    """Re-scores a candidate list using a cross-encoder model."""

    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or Config().reranker_model
        self.model = CrossEncoder(model_name)
        print(f"[Reranker] Initialised with '{model_name}'.")

    def invoke(self, query: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]
