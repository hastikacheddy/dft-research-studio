"""
config/settings.py
------------------
Centralised experiment configuration.

On Lightning.ai, mount your dataset drive under /teamspace/studios/this_studio/
or point the paths to your Lightning Data Studio mounted volume.
All secrets (API keys) must be set as environment variables or via
Lightning.ai's built-in Secrets manager — never hard-coded here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # FILE PATHS                                                           #
    # Adjust base_dir for Lightning.ai (e.g., /teamspace/uploads/Dataset) #
    # ------------------------------------------------------------------ #
    base_dir: str = os.getenv(
        "DFT_BASE_DIR",
        "/teamspace/uploads/Dissertation/Datasets",
    )

    @property
    def nodes_path(self) -> str:
        return os.path.join(self.base_dir, "dft_kg_nodes.csv")

    @property
    def rels_path(self) -> str:
        return os.path.join(self.base_dir, "dft_kg_relationships.csv")

    @property
    def qa_path(self) -> str:
        return os.path.join(self.base_dir, "_DFT-QA-120.csv")

    @property
    def paper_dir(self) -> str:
        return os.path.join(os.path.dirname(self.base_dir), "raw/DFT-Corpus-25")

    @property
    def distractor_dir(self) -> str:
        return os.path.join(os.path.dirname(self.base_dir), "raw/Distractors")

    # ------------------------------------------------------------------ #
    # CHROMADB PERSISTENCE                                                 #
    # ------------------------------------------------------------------ #
    chroma_base_dir: str = os.getenv(
        "DFT_CHROMA_DIR",
        "/teamspace/uploads/Dissertation/VectorDBs",
    )

    @property
    def chroma_persist_dir_clean(self) -> str:
        return os.path.join(self.chroma_base_dir, "chroma_dft_clean")

    @property
    def chroma_persist_dir_noisy(self) -> str:
        return os.path.join(self.chroma_base_dir, "chroma_dft_noisy")

    @property
    def chroma_persist_dir_base(self) -> str:
        return os.path.join(self.chroma_base_dir, "chroma_dft_ratio")

    # ------------------------------------------------------------------ #
    # MODEL SETTINGS                                                       #
    # ------------------------------------------------------------------ #
    models_to_test: List[str] = field(
        default_factory=lambda: ["llama3.1:8b"]
    )
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    spacy_model: str = "en_core_web_sm"

    # ------------------------------------------------------------------ #
    # RETRIEVAL SETTINGS                                                   #
    # ------------------------------------------------------------------ #
    top_k_retrieval: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 100

    # ------------------------------------------------------------------ #
    # EXPERIMENT PARAMETERS                                                #
    # ------------------------------------------------------------------ #
    random_seed: int = 42
    distractor_ratios: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 3.0]
    )

    # Question level index ranges (0-indexed into the full 120-question set)
    question_levels: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "L1": (0, 30),
            "L2": (30, 60),
            "L3": (60, 90),
            "L4": (90, 120),
        }
    )

    # ------------------------------------------------------------------ #
    # LICENSING                                                            #
    # ------------------------------------------------------------------ #
    data_license: str = "CC BY-NC 4.0"
    model_licenses: Dict[str, str] = field(
        default_factory=lambda: {
            "llama3.1:8b": "Llama 3.1 Community License (Meta)",
            "sentence-transformers/all-MiniLM-L6-v2": "Apache-2.0",
        }
    )

    # ------------------------------------------------------------------ #
    # GROQ API KEY (read from environment — never hard-code)              #
    # ------------------------------------------------------------------ #
    @property
    def groq_api_key(self) -> str:
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it via Lightning.ai Secrets or `export GROQ_API_KEY=...`"
            )
        return key
