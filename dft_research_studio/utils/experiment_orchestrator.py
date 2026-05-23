from __future__ import annotations
import json, logging, os, time
from typing import Any, Dict, List, Optional, Set, Tuple
from ..agents import LLMWrapper
from ..config import Config
from ..data import DFTDataManager
from ..evaluation.schemas import ExperimentResult
from .engine_registry import EngineRegistry

logger = logging.getLogger(__name__)

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


class ExperimentOrchestrator:
    def __init__(
        self,
        data_manager: DFTDataManager,
        config: Config | None = None,
        checkpoint_file: str = "checkpoint_results.jsonl",
        wandb_project: str | None = None,
    ) -> None:
        self.cfg  = config or Config()
        self.dm   = data_manager
        self.qa_pairs = self.dm.get_qa_pairs()
        self.checkpoint_file = checkpoint_file
        self.results: List[Dict] = []
        self._completed: Set[Tuple[str, float, str]] = set()
        self._load_checkpoint()

        self.engines = EngineRegistry(data_manager, self.cfg)
        logger.info("Initialising engines for %d ratios ...", len(self.cfg.distractor_ratios))
        self.engines.build_all()
        logger.info("All engines ready. %d results in checkpoint.", len(self.results))

        self._wandb_run = None
        if wandb_project and _WANDB:
            try:
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    config={"distractor_ratios": self.cfg.distractor_ratios,
                            "models": self.cfg.models_to_test,
                            "top_k": self.cfg.top_k_retrieval},
                    resume="allow",
                )
                logger.info("W&B run: %s", self._wandb_run.url)
            except Exception as exc:
                logger.warning("W&B init failed: %s", exc)

    def _load_checkpoint(self) -> None:
        if not os.path.exists(self.checkpoint_file):
            return
        loaded = 0
        with open(self.checkpoint_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self.results.append(entry)
                    self._completed.add((
                        entry["question_id"],
                        float(entry["distractor_ratio"]),
                        entry["experiment_type"],
                    ))
                    loaded += 1
                except Exception as exc:
                    logger.warning("Bad checkpoint line: %s", exc)
        logger.info("Checkpoint: %d results loaded.", loaded)

    def _is_done(self, qa_id, ratio, exp_type):
        return (qa_id, ratio, exp_type) in self._completed

    def _append_checkpoint(self, entry):
        with open(self.checkpoint_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _record(self, model, qa, ratio, exp_type, mode, answer, metrics,
                context="", chunks=None, sources=None, latency_ms=None):
        entry: Dict[str, Any] = {
            "question_id": qa["id"], "question": qa["question"],
            "ground_truth": qa["ground_truth"], "gold_docs": qa.get("gold_docs", []),
            "model": model, "judge_model": os.getenv("JUDGE_MODEL", "unknown"),
            "distractor_ratio": ratio, "experiment_type": exp_type, "mode": mode,
            "generated_answer": answer, "context_used": context,
            "retrieved_text_chunks": chunks or [],
            "retrieved_source_filenames": sources or [],
            "metrics": metrics, "latency_ms": latency_ms,
        }
        try:
            ExperimentResult(**entry)
        except Exception as exc:
            logger.warning("Schema validation warning: %s", exc)
        self.results.append(entry)
        self._completed.add((qa["id"], ratio, exp_type))
        self._append_checkpoint(entry)
        if self._wandb_run and metrics:
            try:
                self._wandb_run.log({
                    f"{exp_type}/{k}": v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float))
                })
            except Exception:
                pass
        logger.debug("Recorded: %s | ratio=%.1f | type=%s", qa["id"], ratio, exp_type)

    def run_baseline(self, models, qa, ratio):
        for m in models:
            if self._is_done(qa["id"], ratio, "Baseline"):
                return
            t0 = time.perf_counter()
            llm = LLMWrapper(m, config=self.cfg)
            ans, *_ = llm.generate(f"Question: {qa['question']}\nAnswer concisely:")
            self._record(m, qa, ratio, "Baseline", "Zero-Shot", ans, {},
                         latency_ms=round((time.perf_counter()-t0)*1000, 2))

    def run_rag(self, models, qa, ratio):
        if self._is_done(qa["id"], ratio, "StandardRAG"):
            return
        t0   = time.perf_counter()
        docs = self.engines.rag(ratio).retriever.invoke(qa["question"])
        context = "\n".join(d.page_content for d in docs)
        sources = [d.metadata.get("source", "") for d in docs]
        for m in models:
            llm = LLMWrapper(m, config=self.cfg)
            ans, *_ = llm.generate(f"Context: {context}\nQ: {qa['question']}")
            self._record(m, qa, ratio, "StandardRAG", "Dense", ans, {},
                         context, [context], sources,
                         round((time.perf_counter()-t0)*1000, 2))

    def run_graph_rag(self, models, qa, ratio):
        if self._is_done(qa["id"], ratio, "GraphRAG"):
            return
        t0     = time.perf_counter()
        engine = self.engines.graph(ratio)
        ctx, pids, _ = engine.get_star_context(qa["question"])
        prompt = engine.generate_paranoid_prompt(qa["question"], ctx)
        for m in models:
            llm = LLMWrapper(m, config=self.cfg)
            ans, *_ = llm.generate(prompt)
            self._record(m, qa, ratio, "GraphRAG", "Topological", ans, {},
                         ctx, [ctx], pids,
                         round((time.perf_counter()-t0)*1000, 2))

    def run_graph_deterministic(self, models, qa, ratio):
        if self._is_done(qa["id"], ratio, "GraphDeterministic"):
            return
        t0     = time.perf_counter()
        engine = self.engines.topo(ratio)
        ctx, _, pids = engine.get_topological_context(qa["question"])
        prompt = engine.generate_deterministic_prompt(qa["question"], ctx)
        for m in models:
            llm = LLMWrapper(m, config=self.cfg)
            ans, *_ = llm.generate(prompt)
            self._record(m, qa, ratio, "GraphDeterministic", "Topological", ans, {},
                         ctx, [ctx], pids,
                         round((time.perf_counter()-t0)*1000, 2))

    def run_multi_agent_system(self, models, qa, ratio):
        if self._is_done(qa["id"], ratio, "MultiAgent"):
            return
        t0  = time.perf_counter()
        mas = self.engines.mas(ratio)
        ans, log, chunks, pids, it, ot, cost, calls = mas.run_workflow(qa["question"])
        self._record(models[0], qa, ratio, "MultiAgent", "Hybrid", ans, {},
                     str(log), chunks, pids,
                     round((time.perf_counter()-t0)*1000, 2))

    def run_template_prompting(self, models, qa, ratio):
        for m in models:
            if self._is_done(qa["id"], ratio, "TemplatePrompting"):
                return
            t0 = time.perf_counter()
            llm = LLMWrapper(m, config=self.cfg)
            prompt = (
                "You are an expert computational chemist specialising in DFT. "
                "Answer precisely using benchmark evidence where possible.\n\n"
                f"Question: {qa['question']}"
            )
            ans, *_ = llm.generate(prompt)
            self._record(m, qa, ratio, "TemplatePrompting", "Template", ans, {},
                         latency_ms=round((time.perf_counter()-t0)*1000, 2))

    def run_cot_prompting(self, models, qa, ratio):
        for m in models:
            if self._is_done(qa["id"], ratio, "CoTPrompting"):
                return
            t0 = time.perf_counter()
            llm = LLMWrapper(m, config=self.cfg)
            prompt = (
                "You are an expert computational chemist. "
                "Think step by step before answering.\n\n"
                f"Question: {qa['question']}\n\n"
                "Let's think step by step:"
            )
            ans, *_ = llm.generate(prompt)
            self._record(m, qa, ratio, "CoTPrompting", "Chain-of-Thought", ans, {},
                         latency_ms=round((time.perf_counter()-t0)*1000, 2))

    def run_bm25_retriever(self, models, qa, ratio):
        if self._is_done(qa["id"], ratio, "BM25Retriever"):
            return
        t0 = time.perf_counter()
        try:
            bm25 = self.engines.bm25_reranker(ratio)
            docs = bm25.invoke(qa["question"])
            context = "\n".join(d.page_content for d in docs)[:2000]
            sources = [d.metadata.get("source", "") for d in docs]
            for m in models:
                llm = LLMWrapper(m, config=self.cfg)
                ans, *_ = llm.generate(f"Context: {context}\nQ: {qa['question']}")
                self._record(m, qa, ratio, "BM25Retriever", "Sparse", ans, {},
                             context, [context], sources,
                             round((time.perf_counter()-t0)*1000, 2))
        except Exception as exc:
            logger.warning("BM25Retriever error: %s", exc)

    def run_cross_encoder_reranker(self, models, qa, ratio):
        if self._is_done(qa["id"], ratio, "CrossEncoderReranker"):
            return
        t0 = time.perf_counter()
        try:
            bm25 = self.engines.bm25_reranker(ratio)
            docs = bm25.invoke(qa["question"])
            context = "\n".join(d.page_content for d in docs)[:2000]
            sources = [d.metadata.get("source", "") for d in docs]
            for m in models:
                llm = LLMWrapper(m, config=self.cfg)
                ans, *_ = llm.generate(f"Context (reranked): {context}\nQ: {qa['question']}")
                self._record(m, qa, ratio, "CrossEncoderReranker", "Reranked", ans, {},
                             context, [context], sources,
                             round((time.perf_counter()-t0)*1000, 2))
        except Exception as exc:
            logger.warning("CrossEncoderReranker error: %s", exc)

    def run_bm25_reranker_rag(self, models, qa, ratio):
        if self._is_done(qa["id"], ratio, "BM25RerankerRAG"):
            return
        t0 = time.perf_counter()
        try:
            bm25 = self.engines.bm25_reranker(ratio)
            docs = bm25.invoke(qa["question"])
            context = "\n".join(d.page_content for d in docs)[:2000]
            sources = [d.metadata.get("source", "") for d in docs]
            for m in models:
                llm = LLMWrapper(m, config=self.cfg)
                ans, *_ = llm.generate(f"Context: {context}\nQ: {qa['question']}")
                self._record(m, qa, ratio, "BM25RerankerRAG", "HybridSparse", ans, {},
                             context, [context], sources,
                             round((time.perf_counter()-t0)*1000, 2))
        except Exception as exc:
            logger.warning("BM25RerankerRAG error: %s", exc)

    def run_multi_agent_bm25(self, models, qa, ratio):
        if self._is_done(qa["id"], ratio, "MultiAgentBM25"):
            return
        t0  = time.perf_counter()
        mas = self.engines.mas(ratio)
        ans, log, chunks, pids, it, ot, cost, calls = mas.run_workflow(qa["question"])
        self._record(models[0], qa, ratio, "MultiAgentBM25", "HybridAgent", ans, {},
                     str(log), chunks, pids,
                     round((time.perf_counter()-t0)*1000, 2))

    def save_results(self, filename="experiment_results_raw.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info("Results saved → '%s' (%d entries).", filename, len(self.results))

    def finish(self):
        if self._wandb_run:
            self._wandb_run.finish()

    def get_chatbot_answer(self, question, experiment_type, distractor_ratio):
        ratio = distractor_ratio
        llm   = self.engines.default_llm

        # ── Part 1: Pure LLM Baselines ──────────────────────────────────
        if experiment_type == "Zero-Shot Prompting":
            ans, *_ = llm.generate(question)
            return ans, "Zero-Shot | No retrieval", ""

        if experiment_type == "Template Prompting":
            prompt = (
                "You are an expert computational chemist specialising in DFT. "
                "Answer precisely using benchmark evidence where possible.\n\n"
                f"Question: {question}"
            )
            ans, *_ = llm.generate(prompt)
            return ans, "Template Prompting | Domain persona", ""

        if experiment_type == "Chain-of-Thought (CoT) Prompting":
            prompt = (
                "You are an expert computational chemist. "
                "Think step by step before answering.\n\n"
                f"Question: {question}\n\n"
                "Let's think step by step:"
            )
            ans, *_ = llm.generate(prompt)
            return ans, "CoT Prompting | Step-by-step reasoning", ""

        # ── Part 2: Standard IR Baselines ───────────────────────────────
        if experiment_type == "BM25 Retriever":
            try:
                bm25  = self.engines.bm25_reranker(ratio)
                docs  = bm25.invoke(question)
                context = "\n".join(d.page_content for d in docs)[:2000]
                ans, *_ = llm.generate(f"Context: {context}\nQ: {question}")
                return ans, f"BM25 Sparse | {len(docs)} docs", context
            except Exception as exc:
                return f"BM25 error: {exc}", "", ""

        if experiment_type == "Cross-Encoder Reranker":
            try:
                bm25  = self.engines.bm25_reranker(ratio)
                docs  = bm25.invoke(question)
                context = "\n".join(d.page_content for d in docs)[:2000]
                ans, *_ = llm.generate(f"Context (reranked): {context}\nQ: {question}")
                return ans, f"Cross-Encoder | {len(docs)} docs reranked", context
            except Exception as exc:
                return f"Reranker error: {exc}", "", ""

        if "BM25 + Reranker" in experiment_type and "Multi-Agent" not in experiment_type:
            try:
                bm25  = self.engines.bm25_reranker(ratio)
                docs  = bm25.invoke(question)
                context = "\n".join(d.page_content for d in docs)[:2000]
                ans, *_ = llm.generate(f"Context: {context}\nQ: {question}")
                return ans, f"BM25+Reranker Hybrid | {len(docs)} docs", context
            except Exception as exc:
                return f"BM25+Reranker error: {exc}", "", ""

        if "Standard RAG" in experiment_type and "Multi-Agent" not in experiment_type:
            docs    = self.engines.rag(ratio).retriever.invoke(question)
            context = "\n".join(d.page_content for d in docs)[:2000]
            ans, *_ = llm.generate(f"Context: {context}\nQ: {question}")
            return ans, f"Standard RAG | {len(docs)} chunks", context

        # ── Part 3: Graph-Based Architectures ───────────────────────────
        if experiment_type == "GraphRAG":
            ctx, pids, _ = self.engines.graph(ratio).get_star_context(question)
            prompt = self.engines.graph(ratio).generate_paranoid_prompt(question, ctx)
            ans, *_ = llm.generate(prompt)
            return ans, "GraphRAG | Star-topology traversal", ctx

        if experiment_type == "Graph Deterministic":
            ctx, _, pids = self.engines.topo(ratio).get_topological_context(question)
            prompt = self.engines.topo(ratio).generate_deterministic_prompt(question, ctx)
            ans, *_ = llm.generate(prompt)
            return ans, "Graph Deterministic | Hub-based 1-hop", ctx

        if "Multi-Agent" in experiment_type:
            ans, log, chunks, pids, it, ot, cost, calls = self.engines.mas(ratio).run_workflow(question)
            return ans, f"Multi-Agent | Cost: ${cost:.4f}", "\n".join(log["steps"])

        return "Mode not implemented", "", ""
