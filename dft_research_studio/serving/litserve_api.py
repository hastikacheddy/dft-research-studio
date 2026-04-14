"""
serving/litserve_api.py
------------------------
Production inference backend built on LitServe.

Architecture
------------
  Gradio UI (app.py)
       │  HTTP POST /predict
       ▼
  LitServe  ←  this file  (port 8000)
       │
       ▼
  ExperimentOrchestrator  →  LLMWrapper (Groq)

Each request is a JSON payload:
  {
    "question":         str,
    "experiment_type":  str,   # "GraphRAG" | "Standard RAG" | ...
    "distractor_ratio": float  # 0.0 | 0.5 | 1.0 | 2.0 | 3.0
  }

Each response:
  {
    "answer":       str,
    "metrics":      str,   # human-readable telemetry line
    "trace":        str,   # graph context / agent log
    "latency_ms":   float
  }

Run locally:
    python serving/litserve_api.py

Run in Docker (see Dockerfile.serve):
    docker build -f Dockerfile.serve -t dft-serve .
    docker run --env-file .env -p 8000:8000 dft-serve
"""

from __future__ import annotations

import os
import time
from typing import Any

import litserve as ls

from dft_research_studio.config import Config
from dft_research_studio.data import DFTDataManager
from dft_research_studio.utils import ExperimentOrchestrator


class DFTInferenceAPI(ls.LitAPI):
    """
    LitAPI implementation.
    setup()   → called once per worker at startup — loads all engines.
    decode_request() → validates + normalises the incoming JSON.
    predict()        → routes to the correct retrieval architecture.
    encode_response() → wraps the result in the response schema.
    """

    def setup(self, device: str) -> None:
        """Load config, data, and all retrieval engines (once per worker)."""
        self.config = Config()
        self.dm = DFTDataManager(self.config)
        self.dm.remove_disconnected_nodes()
        self.runner = ExperimentOrchestrator(self.dm, self.config)
        print(f"[LitServe] DFTInferenceAPI ready on device={device}.")

    # ---------------------------------------------------------------- #

    def decode_request(self, request: dict, **kwargs) -> dict:
        """Validate and coerce request fields."""
        question = str(request.get("question", "")).strip()
        if not question:
            raise ValueError("'question' field is required and must be non-empty.")

        exp_type = str(
            request.get("experiment_type", "GraphRAG")
        )
        valid_types = {
            "Standard RAG",
            "GraphRAG",
            "Graph Deterministic",
            "Multi-Agent System (Standard RAG Fallback)",
        }
        if exp_type not in valid_types:
            raise ValueError(
                f"'experiment_type' must be one of {sorted(valid_types)}. "
                f"Got: '{exp_type}'"
            )

        raw_ratio = request.get("distractor_ratio", 0.0)
        try:
            ratio = float(raw_ratio)
        except (TypeError, ValueError):
            raise ValueError(f"'distractor_ratio' must be a float. Got: {raw_ratio!r}")

        if ratio not in self.config.distractor_ratios:
            raise ValueError(
                f"'distractor_ratio' must be one of {self.config.distractor_ratios}. "
                f"Got: {ratio}"
            )

        return {
            "question": question,
            "experiment_type": exp_type,
            "distractor_ratio": ratio,
        }

    # ---------------------------------------------------------------- #

    def predict(self, payload: dict, **kwargs) -> dict:
        """Route to the ExperimentOrchestrator and capture latency."""
        t0 = time.perf_counter()

        answer, metrics, trace = self.runner.get_chatbot_answer(
            question=payload["question"],
            experiment_type=payload["experiment_type"],
            distractor_ratio=payload["distractor_ratio"],
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "answer": answer,
            "metrics": metrics,
            "trace": trace,
            "latency_ms": round(latency_ms, 2),
        }

    # ---------------------------------------------------------------- #

    def encode_response(self, output: dict, **kwargs) -> dict:
        """Pass the prediction dict through as-is (already JSON-serialisable)."""
        return output


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    workers = int(os.getenv("LITSERVE_WORKERS", "1"))
    port = int(os.getenv("LITSERVE_PORT", "8000"))

    api = DFTInferenceAPI()
    server = ls.LitServer(
        api,
        accelerator="auto",          # GPU if available, else CPU
        workers_per_device=workers,
        timeout=120,                  # seconds — Groq inference can be slow
        max_batch_size=1,             # one query at a time (Groq rate limits)
    )
    server.run(port=port)
