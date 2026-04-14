"""
retrievers/standard_rag.py
--------------------------
Dense vector retrieval using ChromaDB + HuggingFace embeddings.
"""

from __future__ import annotations

import os
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import Config


class StandardRAG:
    """
    Builds (or reloads from disk) a ChromaDB vector store and exposes a
    LangChain-compatible retriever.
    """

    def __init__(
        self,
        documents: List[Document],
        persist_directory: str,
        config: Config | None = None,
    ) -> None:
        cfg = config or Config()
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name=cfg.embedding_model
        )

        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            print(f"[StandardRAG] Loading ChromaDB from '{persist_directory}' …")
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name="dft_experiment",
            )
        else:
            print(f"[StandardRAG] Building ChromaDB at '{persist_directory}' …")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            )
            chunks = splitter.split_documents(documents)
            print(f"[StandardRAG] Indexing {len(chunks)} chunks …")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name="dft_experiment",
                persist_directory=persist_directory,
            )
            self.vector_store.persist()

        self._verify_store(documents)
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": cfg.top_k_retrieval}
        )

    # ------------------------------------------------------------------ #

    def _verify_store(self, source_docs: List[Document]) -> None:
        try:
            count = self.vector_store._collection.count()
            print(f"[StandardRAG] Vector store has {count} documents.")
            if count == 0 and source_docs:
                print("[StandardRAG] WARNING: store is empty despite input documents.")
            if count > 0:
                result = self.vector_store.similarity_search(
                    "What is PBE0 functional?", k=1
                )
                if result:
                    print(
                        f"[StandardRAG] Test retrieval OK — "
                        f"snippet: {result[0].page_content[:80]} …"
                    )
        except Exception as exc:
            print(f"[StandardRAG] Verification error: {exc}")
