"""
evaluation/reproducibility_tracker.py
---------------------------------------
Metadata capture, determinism enforcement, data fingerprinting,
and compute-cost estimation for full experiment auditability.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import psutil
import torch

from ..config import Config


class ReproducibilityTracker:
    """
    Records hardware, library versions, config snapshot, data hashes,
    and runtime stats into a JSON reproducibility log.
    """

    def __init__(
        self,
        config: Config,
        input_files: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self.start_time = time.time()

        self.metadata: Dict = {
            "timestamp_start": datetime.now().isoformat(),
            "python_version": sys.version,
            "os_platform": platform.platform(),
            "hardware": self._hardware_info(),
            "libraries": self._library_versions(),
            "config_snapshot": self._config_dict(),
            "data_hashes": self._hash_files(input_files or []),
        }

        self.set_seeds(config.random_seed)

    # ------------------------------------------------------------------ #

    def _hardware_info(self) -> Dict:
        info: Dict = {
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(
                psutil.virtual_memory().total / (1024 ** 3), 2
            ),
            "gpu_name": "None",
            "gpu_count": 0,
            "cuda_version": "N/A",
        }
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
        elif torch.backends.mps.is_available():
            info["gpu_name"] = "Apple Silicon (MPS)"
        return info

    def _library_versions(self) -> Dict:
        def _ver(obj=None, pkg: str | None = None) -> str:
            try:
                if pkg:
                    return importlib.metadata.version(pkg)
                if obj and hasattr(obj, "__version__"):
                    return obj.__version__
            except Exception:
                pass
            return "N/A"

        import chromadb
        import networkx as nx
        import pandas as pd
        import spacy

        return {
            "numpy": _ver(np),
            "torch": _ver(torch),
            "chromadb": _ver(chromadb),
            "networkx": _ver(nx),
            "pandas": _ver(pd),
            "spacy": _ver(spacy),
            "scikit-learn": _ver(pkg="scikit-learn"),
            "langchain-community": _ver(pkg="langchain-community"),
            "langchain-text-splitters": _ver(pkg="langchain-text-splitters"),
            "rouge-score": _ver(pkg="rouge-score"),
            "thefuzz": _ver(pkg="thefuzz"),
            "groq": _ver(pkg="groq"),
            "pymupdf": _ver(pkg="pymupdf"),
            "pyvis": _ver(pkg="pyvis"),
            "python": "{}.{}.{}".format(*sys.version_info[:3]),
            "platform": platform.platform(),
        }

    def _config_dict(self) -> Dict:
        import dataclasses
        return dataclasses.asdict(self.config)

    def _hash_files(self, paths: List[str]) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        for fp in paths:
            if not os.path.exists(fp):
                hashes[fp] = "MISSING"
                continue
            try:
                h = hashlib.md5()
                with open(fp, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                hashes[fp] = h.hexdigest()
            except Exception as exc:
                hashes[fp] = f"Error: {exc}"
        return hashes

    # ------------------------------------------------------------------ #

    def set_seeds(self, seed: int = 42) -> None:
        """Enforce determinism across Python, NumPy, and PyTorch."""
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"[Tracker] Determinism enforced — seed={seed}.")

    def _estimate_compute_cost(self) -> None:
        duration_h = self.metadata.get("total_runtime_seconds", 0) / 3600
        gpu = self.metadata["hardware"].get("gpu_name", "")
        gpu_kw = 0.0
        if "T4" in gpu:
            gpu_kw = 0.07
        elif "A100" in gpu:
            gpu_kw = 0.40
        elif "V100" in gpu:
            gpu_kw = 0.25
        total_kwh = (gpu_kw + 0.15) * duration_h
        self.metadata["estimated_kwh"] = round(total_kwh, 4)
        self.metadata["estimated_co2_grams"] = round(total_kwh * 475, 2)

    def finalize(self) -> Dict:
        self.metadata["timestamp_end"] = datetime.now().isoformat()
        self.metadata["total_runtime_seconds"] = round(
            time.time() - self.start_time, 2
        )
        self._estimate_compute_cost()
        return self.metadata

    def save_reproducibility_log(
        self, output_path: str = "reproducibility_log.json"
    ) -> None:
        report = self.finalize()
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)
        print(f"[Tracker] Reproducibility log saved to '{output_path}'.")
