# DFT Research Studio

**Evaluating Topological vs. Dense Retrieval in Quantum Chemistry**

A production-grade research pipeline comparing GraphRAG, Standard RAG, BM25+Reranker, and Multi-Agent architectures on a 19,000-node DFT knowledge graph, evaluated on a 120-question benchmark under controlled distractor noise (0%–300%).

[![CI](https://github.com/YOUR_USERNAME/dft-research-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/dft-research-studio/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/YOUR_USERNAME/dft-research-studio/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/dft-research-studio)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture overview

```
                     ┌──────────────────────┐
                     │     Gradio UI         │  app.py  (port 7860)
                     └──────────┬────────────┘
                                │  HTTP POST /predict
                                │  X-API-Key: <token>
                                ▼
                     ┌──────────────────────┐
                     │  LitServe Backend     │  serving/litserve_api.py  (port 8000)
                     │  (auth + validation)  │
                     └──────────┬────────────┘
                                │
          ┌─────────────────────┼──────────────────────┐
          ▼                     ▼                       ▼
     StandardRAG            GraphRAG              MultiAgent
     (ChromaDB)         (star-topology)      (Strategist+Critic)
       BM25+Reranker     TopologicalRAG        Vector fallback
          │                     │                       │
          └─────────────────────┴───────────────────────┘
                                │
                     ┌──────────▼────────────┐
                     │    LLMWrapper          │  → Groq API
                     │    (retry + telemetry) │    llama-3.1-8b-instant
                     └───────────────────────┘
```

### Key production features

| Feature | Implementation |
|---|---|
| **Crash-safe checkpointing** | Results appended to `.jsonl` after every QA pair; resume with `--resume` |
| **Exponential-backoff retry** | 5 retries on Groq rate limits / timeouts with jitter |
| **Structured logging** | Console + rotating `.log` + `.jsonl` files; grep/jq-friendly |
| **API-key authentication** | `X-API-Key` header on LitServe; rejected with HTTP 403 |
| **Environment isolation** | All secrets via `.env` / env vars — nothing hard-coded |
| **Reproducibility tracking** | MD5 fingerprints of all input CSVs + full library version snapshot |
| **W&B integration** | Optional experiment tracking with `--wandb <project>` |
| **CI pipeline** | GitHub Actions: tests on Python 3.11/3.12, ruff lint, Docker build |

---

## Project structure

```
dft_research_studio/
├── config/
│   └── settings.py                  # Env-var driven Config dataclass
├── data/
│   ├── pdf_processor.py             # PyMuPDF section extractor + text cleaner
│   └── data_manager.py              # Graph + corpus + QA orchestration
├── retrievers/
│   ├── bm25_retriever.py            # BM25Okapi lexical retrieval
│   ├── reranker.py                  # Cross-encoder neural reranker
│   ├── bm25_reranker_adapter.py     # Two-stage hybrid pipeline
│   ├── standard_rag.py              # Dense ChromaDB retrieval
│   ├── graph_rag.py                 # Fuzzy star-topology GraphRAG
│   └── topological_retriever.py     # Deterministic NetworkX retriever
├── agents/
│   ├── llm_wrapper.py               # Groq SDK + retry + token telemetry
│   └── multi_agent_graph_rag.py     # Strategist / Critic workflow
├── evaluation/
│   ├── scientific_evaluator.py      # LLM-as-Judge + ROUGE + Recall@K + MRR
│   ├── advanced_metrics.py          # Bootstrap CIs, GFG, plot grids
│   └── reproducibility_tracker.py   # MD5 fingerprinting + env snapshot
├── visualization/
│   ├── graph_visualizer.py          # SVG subgraphs + Pyvis HTML
│   ├── heatmap.py                   # Stratified metric heatmaps
│   ├── fidelity_gap.py              # Generative Fidelity Gap heatmap
│   ├── radar.py                     # Architecture capability radar
│   ├── noise_sensitivity.py         # Noise-collapse line charts
│   ├── significance.py              # Welch T-test p-value matrix
│   └── trace_visualizer.py          # Per-question topological trace SVG
├── serving/
│   └── litserve_api.py              # LitServe backend (auth + logging)
├── utils/
│   ├── experiment_orchestrator.py   # Engine lifecycle + checkpointing + W&B
│   └── logging_config.py            # Centralised structured logging setup
├── tests/
│   ├── conftest.py
│   ├── test_config.py               # Path derivation, question-level partitioning
│   ├── test_pdf_processor.py        # Regex section splitter, text cleaning
│   ├── test_graph_construction.py   # Graph topology invariants, star search
│   ├── test_retrieval_algorithms.py # BM25 ordering, MRR/Recall arithmetic, ROUGE
│   ├── test_statistical_analysis.py # Bootstrap CI, GFG formula, MD5 fingerprinting
│   └── test_qa_parsing.py           # Gold-doc parser edge cases
├── .github/
│   └── workflows/ci.yml             # GitHub Actions CI
├── app.py                           # Gradio frontend
├── main.py                          # CLI experiment runner (checkpointing + W&B)
├── Dockerfile.serve                 # LitServe backend image
├── Dockerfile.app                   # Gradio frontend image
├── docker-compose.yml               # Orchestrates both services
├── pyproject.toml                   # Project metadata + tool config (ruff, pytest)
├── requirements.txt                 # Minimum versions
├── requirements.lock                # Exact pins (generated, see below)
├── .env.example                     # Secret and path template
├── .gitignore
└── Makefile
```

---

## Quick start — Lightning.ai

### 1. Find your files

```bash
find /teamspace -name "dft_kg_nodes.csv" 2>/dev/null
```

Note the base path — this is your `DFT_BASE_DIR`.

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set:
#   GROQ_API_KEY=sk-...
#   DFT_BASE_DIR=/teamspace/uploads/Dissertation/Datasets
#   DFT_CHROMA_DIR=/teamspace/uploads/Dissertation/VectorDBs
#   SERVE_API_KEY=any-random-secret-string
```

### 3. Install

```bash
make install
```

### 4. Pin exact dependencies (do this once, commit the result)

```bash
make pin-deps
git add requirements.lock
```

### 5. Verify paths

```bash
python - <<'EOF'
from dotenv import load_dotenv; load_dotenv()
from dft_research_studio.config import Config
import os
cfg = Config()
print("Nodes exists:", os.path.exists(cfg.nodes_path))
print("QA exists:   ", os.path.exists(cfg.qa_path))
print("PDFs found:  ", len([f for f in os.listdir(cfg.paper_dir) if f.endswith('.pdf')]))
EOF
```

All three should print `True / True / 25`.

### 6. Run the tests

```bash
make test
```

136 unit tests, no API key or datasets needed.

### 7. Debug matrix (sanity check)

```bash
make debug
```

Runs 1 QA pair through all 5 architectures × 5 distractor ratios. Takes ~5 minutes.

### 8. Full experiment

```bash
make run

# With W&B tracking:
make run-tracked
```

If the run is interrupted for any reason:

```bash
make resume
```

The orchestrator reads the existing checkpoint and skips already-completed runs.

---

## Docker deployment

### Setup

```bash
cp .env.example .env
# Edit .env — set GROQ_API_KEY, SERVE_API_KEY, and volume paths
```

### Start both services

```bash
make docker-up
```

| Service | Port | Role |
|---|---|---|
| `dft-serve` | 8000 | LitServe inference backend |
| `dft-app` | 7860 | Gradio frontend |

The frontend waits for the backend's health check before accepting requests.

### LitServe API reference

```
POST http://localhost:8000/predict
X-API-Key: <your SERVE_API_KEY>
Content-Type: application/json

{
  "question":         "What is the MAE of PBE0 on the S66 dataset?",
  "experiment_type":  "GraphRAG",
  "distractor_ratio": 0.0
}
```

**Valid `experiment_type`:** `"Standard RAG"`, `"GraphRAG"`, `"Graph Deterministic"`, `"Multi-Agent System (Standard RAG Fallback)"`

**Valid `distractor_ratio`:** `0.0`, `0.5`, `1.0`, `2.0`, `3.0`

**Response:**

```json
{
  "answer":      "PBE0 achieves a MAE of 0.23 kcal/mol on S66.",
  "metrics":     "Graph Trace Active",
  "trace":       "GRAPH EVIDENCE (3 candidates)\n...",
  "latency_ms":  842.3
}
```

---

## GitHub setup (version control)

```bash
git init
git add .
git commit -m "Initial commit: DFT Research Studio v1.0"
git remote add origin https://github.com/YOUR_USERNAME/dft-research-studio.git
git push -u origin main
```

The CI pipeline runs automatically on every push. Replace `YOUR_USERNAME` in the badge URLs at the top of this file once the repo is public.

---

## W&B experiment tracking (free tier)

```bash
pip install wandb
wandb login                    # paste your API key from wandb.ai
python main.py --wandb dft-research-studio
```

Every result is logged to your W&B dashboard with metrics by architecture and distractor ratio. The free tier covers unlimited projects with 100 GB storage.

---

## Tests

```bash
make test          # all unit tests (no API key, no datasets)
make test-fast     # stop on first failure
make coverage      # coverage report + HTML, min 70% required
```

### Test philosophy

Each module tests a specific mathematical property with a known expected value:

| Module | Verified property |
|---|---|
| `test_config.py` | Question levels partition to exactly 120 with zero gaps or overlaps |
| `test_pdf_processor.py` | Section regex matches ACS/RSC header formats; cleaning is idempotent |
| `test_graph_construction.py` | `PBE0 → MAE_PBE0_S66 → S66` path exists; star search returns correct 1-hop neighbourhood |
| `test_retrieval_algorithms.py` | BM25 ranks the PBE0/S66 document first for a PBE0/S66 query; MRR at rank-2 = exactly 0.5 |
| `test_statistical_analysis.py` | Bootstrap CI satisfies `lower ≤ mean ≤ upper`; GFG = 0 for ceiling system; GFG = 0.375 for StandardRAG (2.5/4.0 ceiling); MD5 detects 1-byte change |
| `test_qa_parsing.py` | Gold-doc parser correctly splits comma-separated filenames including edge cases |

Integration tests (require datasets + `GROQ_API_KEY`) are marked `@pytest.mark.integration` and skipped by default.

---

## Logging

Logs are written to `logs/` on every run:

```
logs/
├── full_20240315_142300.log     # human-readable
└── full_20240315_142300.jsonl   # JSON lines for grep/jq analysis
```

Filter by component:
```bash
grep "agents.llm_wrapper" logs/full_*.log      # all LLM calls
grep '"level":"ERROR"' logs/full_*.jsonl | jq  # all errors as JSON
```

Set verbosity:
```bash
export LOG_LEVEL=DEBUG   # DEBUG, INFO, WARNING, ERROR
```

---

## Reproducibility

Every run produces a `*_reproducibility_log.json` with:
- Library versions (numpy, torch, chromadb, spacy …)
- Hardware snapshot (CPU, RAM, GPU, CUDA version)
- MD5 fingerprints of all three input CSVs
- Total runtime, energy estimate (kWh), CO₂ estimate (g)
- Full Config snapshot and random seed

---

## Makefile reference

```bash
make install          # pip install + SpaCy + NLTK
make pin-deps         # generate requirements.lock
make test             # unit tests (no LLM, no datasets)
make test-fast        # stop on first failure
make coverage         # coverage report, min 70%
make lint             # ruff check
make format           # ruff format + autofix
make serve-api        # start LitServe backend (port 8000)
make serve-app        # start Gradio frontend (port 7860)
make serve-standalone # Gradio with embedded engines
make docker-build     # build both Docker images
make docker-up        # docker compose up --build
make docker-down      # docker compose down
make debug            # 1 QA pair smoke test
make run              # full experiment
make resume           # resume from checkpoint
make run-tracked      # full experiment with W&B
make setup-wandb      # install wandb + login
```

---

## Licensing

| Component | License |
|---|---|
| Dataset (`dft_kg_nodes.csv`, `dft_kg_relationships.csv`, `_DFT-QA-120.csv`) | CC BY-NC 4.0 |
| `llama-3.1-8b-instant` | Llama 3.1 Community License (Meta) |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 |
| This codebase | MIT |
