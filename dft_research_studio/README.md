<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/Gradio-4.36-orange?logo=gradio" />
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-green?logo=ollama" />
  <img src="https://img.shields.io/badge/KG-19%2C726%20nodes-purple" />
  <img src="https://img.shields.io/badge/Experiments-6%2C600%20runs-red" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

<h1 align="center">🧪 Auto-KGR Research Studio</h1>

<p align="center">
  <strong>An Autonomous Knowledge-Graph-Driven Reasoning Framework for Scientific Meta-Engineering in Quantum Chemistry</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Hastika06/dft-research-studio">🤗 Live Demo</a> •
  <a href="#-quick-start">🚀 Quick Start</a> •
  <a href="#-architecture">🏗️ Architecture</a> •
  <a href="#-results">📊 Results</a> •
  <a href="#-citation">📝 Citation</a>
</p>

---

## 📖 Overview

**Auto-KGR** is a neuro-symbolic framework that integrates Large Language Models with structured Knowledge Graphs to enable autonomous, multi-hop reasoning for DFT (Density Functional Theory) functional selection in quantum chemistry.

The system compares **11 retrieval architectures** across **3 paradigms** (Pure LLM, Standard IR, Graph-Based), evaluated on a **120-question stratified challenge set** under **5 controlled distractor noise levels**, totalling **6,600 experiment runs**.

### Key Contributions

- **19,726-node Knowledge Graph** constructed from 25 DFT benchmark papers with 97,924 relationships
- **Multi-Agent Dialectic Debate System (MAMEF)** with Advisor, Safety Officer, and Meta-Reasoner agents
- **Groundedness-Correctness Tradeoff Discovery**: Graph architectures achieve **94% groundedness** (vs 0% for baselines) while maintaining verifiable scientific reasoning
- **Dual-Judge Evaluation** with inter-rater agreement **r = 0.856**
- **Q-120 Stratified Challenge Set** with 4 complexity tiers and 3 paraphrase variations per question

---

## 🏗️ Architecture

### 11 Retrieval Architectures (3 Paradigms)

| # | Architecture | Paradigm | Mechanism |
|---|---|---|---|
| 1 | Zero-Shot Prompting | Pure LLM | No retrieval — parametric memory only |
| 2 | Template Prompting | Pure LLM | Domain persona + formatting rules |
| 3 | Chain-of-Thought (CoT) | Pure LLM | Step-by-step reasoning trace |
| 4 | BM25 Retriever | Standard IR | Okapi BM25 keyword matching |
| 5 | Cross-Encoder Reranker | Standard IR | Deep semantic query-document pair scoring |
| 6 | BM25 + Reranker RAG | Standard IR | BM25 recall → Cross-Encoder precision |
| 7 | Standard Vector RAG | Standard IR | HuggingFace embeddings → ChromaDB |
| 8 | GraphRAG (Star Context) | Graph-Based | NER → fuzzy node match → ego-graph traversal |
| 9 | Topological Retriever | Graph-Based | Hub-based 1-hop ranked by connection density |
| 10 | Multi-Agent (GraphRAG) | Graph-Based | Strategist + Critic + graph validation |
| 11 | Multi-Agent (BM25) | Graph-Based | Strategist + Critic + BM25 fallback |

### Multi-Agent Dialectic Debate (MAMEF)

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐
│   ADVISOR    │────▶│  SAFETY OFFICER  │────▶│ META-REASONER │
│ llama3.1:8b  │     │   llama3.1:8b    │     │ qwen2.5:14b   │
│ Vector Store │     │ Knowledge Graph  │     │ 5-Step CoT    │
└─────────────┘     └──────────────────┘     └───────────────┘
       │                     │                       │
   Proposes            Validates against        Evaluates criteria
   functional +        KG failure modes         quality & concludes
   basis set           (FAILS_* edges)          with certificate
```

### Knowledge Graph Schema

```
Functional ──[VALIDATED_ON]──▶ BenchmarkSet
     │                              │
     ├──[FAILS_ON]──▶ FailureMode   ├──[CONTAINS]──▶ Chemical_System
     │                              │
     └──[APPLIES_CORRECTION]──▶ DispersionCorrection
                                    │
Paper ──[CITES]──▶ Functional      ValidationResult ──[REPORTS_RESULT]──▶ Functional
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/) (for local LLM inference)
- ~25 GB disk space (models + VectorDBs)
- GPU with ≥16 GB VRAM (recommended: NVIDIA L4/T4)

### 1. Clone the Repository

```bash
git clone https://github.com/hastikacheddy/dft-research-studio.git
cd dft-research-studio
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab stopwords wordnet
```

### 3. Set Up Ollama (Local LLM)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull required models
ollama serve &
ollama pull llama3.1:8b        # Main LLM (answer generation)
ollama pull qwen2.5:14b        # Primary Judge
ollama pull gemma2:9b           # Secondary Judge
```

### 4. Configure Environment

Create a `.env` file in the project root:

```env
# ── LLM Configuration ──
# If using Ollama (recommended): models are detected automatically
# If using Groq API instead: uncomment and set your key
# GROQ_API_KEY=gsk_your_key_here

# ── Judge Models ──
JUDGE_MODEL=qwen2.5:14b
SECONDARY_JUDGE_MODEL=gemma2:9b

# ── Optional ──
HF_TOKEN=hf_your_token_here        # For HuggingFace model downloads
WANDB_API_KEY=your_wandb_key       # For experiment tracking
```

### 5. Run the Gradio App (Interactive Demo)

```bash
python app.py --standalone
```

Open `http://localhost:7860` in your browser.

---

## 🧪 Running Experiments

### Debug Run (1 question × 11 architectures × 5 ratios = 55 runs)

```bash
export $(grep -v '^#' .env | xargs)
python main.py --debug
```

### Full Ablation Study (120 questions × 11 architectures × 5 ratios = 6,600 runs)

```bash
export $(grep -v '^#' .env | xargs)
mkdir -p results/full
nohup python main.py --wandb dft-research-studio > results/full/full_run.log 2>&1 &
```

**Estimated runtime:** ~15 hours on NVIDIA L4 (24GB)

### Resume from Checkpoint

If the experiment is interrupted, it automatically resumes:

```bash
python main.py --resume
```

### Custom Distractor Ratios

```bash
python main.py --ratios 0.0 1.0 3.0
```

---

## 📊 Results

### Generate All Charts ( visualizations)

```bash
python -m dft_research_studio.visualization.generate_all_charts \
    --input results/full/full_experiment_metrics.csv \
    --output results/full/charts
```

### Generate All Tables (15 CSV tables for dissertation)

```bash
python -m dft_research_studio.visualization.generate_all_tables \
    --input results/full/full_experiment_metrics.csv \
    --output results/full/tables
```

### Key Findings

| Metric | Best Architecture | Score |
|---|---|---|
| Correctness (1-5) | StandardRAG | 2.86 ± 0.96 |
| Groundedness (0-1) | **GraphRAG** | **0.94** |
| Recall@K | CrossEncoderReranker | 0.75 |
| Inter-rater Agreement | — | r = 0.856 |

**The Correctness–Groundedness Tradeoff:** Graph architectures sacrifice fluency for verifiability — achieving 94% groundedness (vs 0% for baselines) by constraining answers to knowledge graph evidence, critical for safety-sensitive scientific applications.

---

## 📁 Project Structure

```
dft-research-studio/
├── app.py                          # Gradio interactive demo (3 tabs)
├── main.py                         # Experiment runner (debug/full)
├── setup_ollama.sh                 # Ollama installation script
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project metadata + build config
├── pytest.ini                      # Pytest configuration
├── Makefile                        # Common dev commands
├── Dockerfile                      # HuggingFace Spaces deployment
├── Dockerfile.serve                # LitServe production deployment
├── docker-compose.yml              # Multi-container orchestration
├── demo_cache.json                 # Pre-computed answers for HF Spaces cache mode
├── README.md                       # This file
├── README_HF.md                    # HuggingFace Spaces README
├── .env                            # Configuration (create from template above)
├── .dockerignore                   # Docker build exclusions
├── .gitignore                      # Git exclusions
│
├── dft_research_studio/
│   ├── __init__.py
│   ├── README.md                   # Package-level documentation
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── llm_wrapper.py          # Ollama/Groq unified LLM interface
│   │   ├── debate_orchestrator.py  # Multi-agent dialectic debate (Auto-KGR Lab)
│   │   ├── multi_agent_graph_rag.py# Strategist + Critic retrieval pipeline
│   │   ├── meta_reasoning_engine.py# Meta-reasoner (5-step CoT)
│   │   ├── kg_query_engine.py      # KG query helper for agents
│   │   └── tab3_ingestion.py       # ArXiv ingestion tab logic
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py             # Centralised experiment configuration
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_manager.py         # KG + dataset orchestrator
│   │   ├── pdf_processor.py        # Scientific PDF extraction
│   │   ├── arxiv_ingester.py       # ArXiv auto-ingestion pipeline
│   │   ├── graph_postprocessor.py  # KG normalization & cleanup
│   │   └── processed/
│   │       ├── dft_kg_nodes.csv         # 19,726 KG nodes
│   │       ├── dft_kg_relationships.csv # 100,442 relationships
│   │       ├── _DFT-QA-120.csv          # Q-120 challenge set
│   │       └── paper_id_mapping.json    # Paper ID → filename mapping
│   │
│   ├── retrievers/
│   │   ├── __init__.py
│   │   ├── standard_rag.py         # ChromaDB vector retrieval
│   │   ├── graph_rag.py            # Star-topology graph traversal
│   │   ├── topological_retriever.py# GraphDeterministic (NetworkX traversal)
│   │   ├── bm25_retriever.py       # Okapi BM25 sparse retrieval
│   │   ├── reranker.py             # Cross-encoder neural reranker
│   │   └── bm25_reranker_adapter.py# BM25 + Reranker hybrid
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── scientific_evaluator.py # LLM-as-Judge + ROUGE + IR metrics
│   │   ├── advanced_metrics.py     # Bootstrap CI, Cohen's d, GFG
│   │   ├── reproducibility_tracker.py# Hardware specs, seeds, hashes
│   │   └── schemas.py              # Pydantic schemas for evaluation outputs
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   └── litserve_api.py         # LitServe production API endpoint
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── generate_all_charts.py  #  charts (50+ PNGs)
│   │   ├── generate_all_tables.py  # CSV summary tables (16 tables)
│   │   ├── heatmap.py              # Multi-subplot heatmaps
│   │   ├── radar.py                # Architecture capability radar
│   │   ├── noise_sensitivity.py    # Robustness profile curves
│   │   ├── fidelity_gap.py         # Generative Fidelity Gap heatmap
│   │   ├── compute_graph_lift.py   # Graph Lift (ΔG%) computation
│   │   ├── significance.py         # Welch's t-test p-value matrix
│   │   ├── graph_visualizer.py     # KG topology rendering
│   │   └── trace_visualizer.py     # Retrieval trace visualization
│   │
│   └── utils/
│       ├── __init__.py
│       ├── experiment_orchestrator.py# 11 run_* methods + checkpointing
│       ├── engine_registry.py      # Model/retriever factory
│       └── logging_config.py       # Structured logging
│
├── results/
│   ├── debug/                      # Debug experiment outputs
│   │   ├── app_hf.py
│   │   ├── debug_checkpoint.jsonl
│   │   ├── debug_experiment_metrics.csv
│   │   ├── debug_results_raw.json
│   │   ├── debug_reproducibility_log.json
│   │   ├── final_results_with_ci.csv
│   │   ├── table3_generative_gap.csv
│   │   └── metrics_bar_grid.png / .pdf
│   │
│   └── full/                       # Full ablation study (6,600 runs)
│       ├── full_results_raw.json       # Raw answers
│       ├── full_experiment_metrics.csv # Evaluated metrics
│       ├── final_results_with_ci.csv   # Bootstrap confidence intervals
│       ├── table3_generative_gap.csv   # Generative Fidelity Gap
│       ├── charts/                     # 50+ PNG visualizations
│       └── tables/                     # 16 CSV tables (table_5_1 through table_6_10 + graph_lift)
│
├── tests/                          # Pytest suite
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures
│   ├── test_config.py
│   ├── test_graph_construction.py
│   ├── test_new_architectures.py
│   ├── test_pdf_processor.py
│   ├── test_qa_parsing.py
│   ├── test_retrieval_algorithms.py
│   └── test_statistical_analysis.py
│
├── notebooks/                      # Research & KG construction notebooks
│   ├── 00_Auto_KGR_R&D.ipynb       # Main research notebook
│   └── Archive_Extraction_Logic/
│       ├── 01_Distractor_corpus_acquisition.ipynb
│       ├── 02_entity_relationship_extraction.ipynb
│       ├── 03_graph_normalization_cleaning.ipynb
│       └── 04_topological_enhancement.ipynb
│
└── VectorDBs/                      # ChromaDB stores (per distractor ratio)
    ├── chroma_dft_ratio_0_0/           # 2,330 chunks (clean corpus)
    ├── chroma_dft_ratio_0_5/           # 3,097 chunks (+50% distractors)
    ├── chroma_dft_ratio_1_0/           # 4,025 chunks (+100% distractors)
    ├── chroma_dft_ratio_2_0/           # 5,679 chunks (+200% distractors)
    └── chroma_dft_ratio_3_0/           # 7,828 chunks (+300% distractors)
```

---

## 🐳 Docker / HuggingFace Spaces

### Live Demo

The app is deployed on HuggingFace Spaces in **cache-only mode** (no GPU required):

🔗 **[https://huggingface.co/spaces/Hastika06/dft-research-studio](https://huggingface.co/spaces/Hastika06/dft-research-studio)**

### Build Docker Locally

```bash
docker build -t auto-kgr .
docker run -p 7860:7860 --env-file .env auto-kgr
```

### Deploy to HuggingFace Spaces

1. Set secrets in HF Space settings:
   - `GROQ_API_KEY` (optional, for API mode)
   - `DFT_BASE_DIR` → `/app/dft_research_studio/data/processed`

2. Push to HF:
```bash
git remote add hf https://huggingface.co/spaces/Hastika06/dft-research-studio
git push hf main --force
```

---

## 🔧 Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | No* | — | Groq API key (only if not using Ollama) |
| `JUDGE_MODEL` | Yes | `qwen2.5:14b` | Primary LLM-as-Judge model |
| `SECONDARY_JUDGE_MODEL` | Yes | `gemma2:9b` | Secondary judge for inter-rater reliability |
| `HF_TOKEN` | No | — | HuggingFace token for model downloads |
| `WANDB_API_KEY` | No | — | Weights & Biases experiment tracking |

*\*Ollama is auto-detected. If Ollama is running, API keys are not needed.*

### Models Used

| Role | Model | Size | Purpose |
|---|---|---|---|
| Generator | `llama3.1:8b` | 4.9 GB | Answer generation (all 11 architectures) |
| Primary Judge | `qwen2.5:14b` | 9.0 GB | Correctness, Relevance, Groundedness scoring |
| Secondary Judge | `gemma2:9b` | 5.4 GB | Inter-rater reliability validation |
| Embeddings | `all-MiniLM-L6-v2` | 80 MB | Document chunk embeddings (ChromaDB) |
| Reranker | `ms-marco-MiniLM-L-6-v2` | 80 MB | Cross-encoder document reranking |
| NER | `en_core_web_sm` | 12 MB | Entity extraction for GraphRAG |

---

## 📈 Evaluation Metrics

| Metric | Scale | Method | Description |
|---|---|---|---|
| Correctness | 1-5 (Likert) | LLM-as-Judge | Answer accuracy vs ground truth |
| Correctness (2nd) | 1-5 (Likert) | LLM-as-Judge | Inter-rater reliability check |
| Relevance | 1-5 (Likert) | LLM-as-Judge | How well the answer addresses the question |
| Groundedness | 0-1 (Binary) | LLM-as-Judge | Whether claims are supported by retrieved context |
| ROUGE-1 | 0-1 (F1) | Automated | Unigram overlap with ground truth |
| ROUGE-L | 0-1 (F1) | Automated | Longest common subsequence overlap |
| Recall@K | 0-1 | Automated | Fraction of gold documents retrieved |
| Precision@K | 0-1 | Automated | Fraction of retrieved docs that are relevant |
| MRR | 0-1 | Automated | Mean reciprocal rank of first relevant document |
| Cohen's d | Real | Computed | Effect size vs baseline (pooled std) |
| GFG | % | Computed | Generative Fidelity Gap vs StandardRAG |

---

## 🧑‍🎓 Author

**Hastika Cheddy**

---

## 📝 Citation

```bibtex
@thesis{cheddy2026autokgr,
  title     = {Auto-KGR: An Autonomous Knowledge-Graph-Driven Reasoning Framework 
               for Scientific Meta-Engineering in Quantum Chemistry},
  author    = {Cheddy, Hastika},
  year      = {2026}
}
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ for reproducible scientific AI research
</p>
