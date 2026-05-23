from __future__ import annotations
import logging
from typing import Dict, Optional
from ..config import Config
from ..data import DFTDataManager
from ..retrievers import BM25RerankerAdapter, GraphRAG, Reranker, StandardRAG, TopologicalRetriever
from ..agents import MultiAgentGraphRAG, LLMWrapper

logger = logging.getLogger(__name__)


class EngineRegistry:
    """Single responsibility: owns and initialises all retrieval engines."""

    def __init__(self, data_manager: DFTDataManager, config: Config) -> None:
        self.dm  = data_manager
        self.cfg = config
        self._rag:           Dict[float, StandardRAG]          = {}
        self._graph:         Dict[float, GraphRAG]             = {}
        self._topo:          Dict[float, TopologicalRetriever] = {}
        self._mas:           Dict[float, MultiAgentGraphRAG]   = {}
        self._bm25_reranker: Dict[float, BM25RerankerAdapter]  = {}
        self._reranker:      Optional[Reranker]                = None
        self._default_llm:   Optional[LLMWrapper]              = None

    @property
    def default_llm(self) -> LLMWrapper:
        if self._default_llm is None:
            self._default_llm = LLMWrapper(self.cfg.models_to_test[0], config=self.cfg)
        return self._default_llm

    def build(self, ratio: float) -> None:
        if ratio in self._rag:
            return
        logger.info("Building engines for ratio=%.2f ...", ratio)
        persist_dir = f"{self.cfg.chroma_persist_dir_base}_{str(ratio).replace('.', '_')}"
        docs = self.dm.load_pdfs(distractor_ratio=ratio)
        self._rag[ratio]   = StandardRAG(docs, persist_dir, config=self.cfg)
        self._graph[ratio] = GraphRAG(self.dm.nodes_df, self.dm.rels_df)
        self._topo[ratio]  = TopologicalRetriever(self.dm.nodes_df, self.dm.rels_df)
        if self._reranker is None:
            self._reranker = Reranker()
        # If no PDFs found, extract docs from ChromaDB for BM25
        bm25_docs = docs
        if not bm25_docs:
            try:
                chroma_data = self._rag[ratio].vector_store.get()
                if chroma_data and chroma_data.get("documents"):
                    from langchain_core.documents import Document as LCDoc
                    bm25_docs = [
                        LCDoc(page_content=text, metadata={"source": meta.get("source","")})
                        for text, meta in zip(chroma_data["documents"], chroma_data.get("metadatas", [{}]*len(chroma_data["documents"])))
                        if text and len(text) > 50
                    ]
                    logger.info("BM25: loaded %d docs from ChromaDB (no PDFs available)", len(bm25_docs))
            except Exception as exc:
                logger.warning("BM25: could not extract docs from ChromaDB: %s", exc)
        self._bm25_reranker[ratio] = BM25RerankerAdapter(
            documents=bm25_docs, reranker=self._reranker, top_k=self.cfg.top_k_retrieval,
        )
        self._mas[ratio] = MultiAgentGraphRAG(
            graph_engine=self._graph[ratio],
            vector_retriever=self._rag[ratio].retriever,
            llm_client=self.default_llm,
        )
        logger.debug("Engines ready for ratio=%.2f", ratio)

    def build_all(self) -> None:
        for ratio in self.cfg.distractor_ratios:
            self.build(ratio)

    def rag(self, ratio: float)           -> StandardRAG:          return self._rag[ratio]
    def graph(self, ratio: float)         -> GraphRAG:             return self._graph[ratio]
    def topo(self, ratio: float)          -> TopologicalRetriever: return self._topo[ratio]
    def mas(self, ratio: float)           -> MultiAgentGraphRAG:   return self._mas[ratio]
    def bm25_reranker(self, ratio: float) -> BM25RerankerAdapter:  return self._bm25_reranker[ratio]
