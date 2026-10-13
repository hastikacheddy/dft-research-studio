# Data and Software Availability

## Software

- **Code:** https://github.com/hastikacheddy/dft-research-studio (version 1.0.0), MIT licence, see [`LICENSE`](LICENSE).
- **Python:** 3.12 (3.12.11 for the recorded run).
- **Libraries:** minimum versions in [`requirements.txt`](requirements.txt); exact versions recorded by the experiment's reproducibility tracker in [`requirements-recorded.txt`](requirements-recorded.txt). Install with `pip install -r requirements.txt -c requirements-recorded.txt`.
- **Third-party models:**

| Role | Identifier | Location |
|---|---|---|
| Generator | `llama3.1:8b` | https://ollama.com/library/llama3.1 |
| Primary judge | `qwen2.5:14b` | https://ollama.com/library/qwen2.5 |
| Secondary judge | `gemma2:9b` | https://ollama.com/library/gemma2 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2 |
| NER | `en_core_web_sm` (spaCy) | https://spacy.io/models/en |

## Data in this repository

All files are machine-readable (CSV or JSON).

| File | Content |
|---|---|
| `dft_research_studio/data/processed/dft_kg_nodes.csv` | Knowledge-graph nodes (19,726) |
| `dft_research_studio/data/processed/dft_kg_relationships.csv` | Knowledge-graph relationships (100,442) |
| `dft_research_studio/data/processed/_DFT-QA-120.csv` | Q-120 challenge set (120 questions, ground truths, gold documents) |
| `dft_research_studio/data/processed/paper_id_mapping.json` | `paper_id` key to paper title (fragment) |
| `dft_research_studio/data/processed/corpus_manifest.csv` | The 25 source papers |
| `results/full/full_results_raw.json` | Raw generated answers, all 6,600 runs |
| `results/full/full_experiment_metrics.csv` | Judge scores and retrieval metrics, one row per run |
| `results/full/final_results_with_ci.csv`, `table3_generative_gap.csv` | Aggregates with bootstrap confidence intervals |
| `results/full/tables/`, `results/full/charts/` | Tables and figures |
| `results/debug/` | Single-question debug run, including `debug_reproducibility_log.json` (configuration and library versions of that run) |

## Third-party data not redistributed

- **Source papers (25):** listed in [`corpus_manifest.csv`](dft_research_studio/data/processed/corpus_manifest.csv). The PDFs are third-party publications and are not included.
- **Distractor papers:** downloaded from arXiv with the `arxiv` Python client (`notebooks/Archive_Extraction_Logic/01_Distrator_corpus_acquisition.ipynb`) using the query `abs:"density functional theory" AND abs:"benchmark"`, sorted by submission date (newest first); the first 75 results were saved as `Distractor_001.pdf` to `Distractor_075.pdf`.
- **Vector stores:** built automatically from the PDFs when the target directory is missing or empty (`dft_research_studio/retrievers/standard_rag.py`), one store per distractor ratio.

## Parameters

Defined in [`dft_research_studio/config/settings.py`](dft_research_studio/config/settings.py) and written to the run's reproducibility log (`config_snapshot`, see `results/debug/debug_reproducibility_log.json`):

| Parameter | Value |
|---|---|
| Retrieval top-k | 5 |
| Chunk size / overlap | 1000 / 100 |
| Distractor ratios | 0.0, 0.5, 1.0, 2.0, 3.0 |
| Random seed | 42 |
| Question tiers (index ranges in Q-120) | L1 0-30, L2 30-60, L3 60-90, L4 90-120 |
| Generation temperature | 0.0 |
| Judge options | temperature 0, `num_predict` 256 |

## Reproducing the results

```bash
git clone https://github.com/hastikacheddy/dft-research-studio.git
cd dft-research-studio
pip install -r requirements.txt -c requirements-recorded.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab stopwords wordnet
# pull the three Ollama models listed above (see README)

# point the code at the shipped data (defaults to a private cloud path)
export DFT_BASE_DIR=$PWD/dft_research_studio/data/processed
export DFT_CHROMA_DIR=$PWD/VectorDBs
```

Place the source PDFs in `dft_research_studio/data/raw/DFT-Corpus-25/` and the distractor PDFs in `dft_research_studio/data/raw/Distractors/` (the code looks for `raw/` next to `processed/`), then:

```bash
python main.py --debug                                   # one question, all architectures, all ratios
python main.py --wandb dft-research-studio               # full run (120 questions x 11 architectures x 5 ratios)
python main.py --resume                                  # resume from checkpoint

python -m dft_research_studio.visualization.generate_all_tables --input results/full/full_experiment_metrics.csv --output results/full/tables
python -m dft_research_studio.visualization.generate_all_charts --input results/full/full_experiment_metrics.csv --output results/full/charts
```
