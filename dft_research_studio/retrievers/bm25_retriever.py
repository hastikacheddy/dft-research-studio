"""
retrievers/bm25_retriever.py
----------------------------
Lexical BM25 retrieval engine.
"""

from __future__ import annotations

from typing import List

import nltk
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ..config import Config


class BM25Retriever:
    """Probabilistic BM25 retriever over a pre-loaded document corpus."""

    def __init__(self, documents: List[Document]) -> None:
        self.documents = documents

        tokenized: List[List[str]] = []
        for doc in documents:
            if isinstance(doc.page_content, str):
                tokenized.append(nltk.word_tokenize(doc.page_content.lower()))
            else:
                print(
                    f"[BM25] Non-string page_content for doc: {doc.metadata}"
                )
                tokenized.append([])

        if not tokenized:
            self.bm25 = None
            return
        self.bm25 = BM25Okapi(tokenized)
        print(f"[BM25] Initialised with {len(documents)} documents.")

    def invoke(
        self,
        query: str,
        k: int | None = None,
    ) -> List[Document]:
        if not self.documents or self.bm25 is None:
            return []
        cfg_k = Config().top_k_retrieval
        k = k if k is not None else cfg_k
        tokens = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_indices]
