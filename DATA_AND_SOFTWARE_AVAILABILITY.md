# Data and Software Availability

This file documents what is and is not shared with this repository, and how to
reproduce the reported results. It follows the JCIM *Data and Software
Availability* requirements. Items marked **⚠ TO COMPLETE** need information that
only the author holds.

## 1. Software

| Item | Value |
|---|---|
| Repository | https://github.com/hastikacheddy/dft-research-studio |
| Version | 1.0.0 (`pyproject.toml`) |
| Licence (code) | MIT, see [`LICENSE`](LICENSE) |
| Archived release / DOI | **⚠ TO COMPLETE** — create a GitHub release and archive it on Zenodo, then add the DOI here and in `CITATION.cff` |
| Language | Python 3.12 (study run on 3.12.11) |
| Library versions | [`requirements-recorded.txt`](requirements-recorded.txt) (exact pins recorded by the run) plus [`requirements.txt`](requirements.txt) (minimum versions) |
| Commercial software | None |

**Pinned versions are partial.** `requirements-recorded.txt` holds the 14 library
versions that `reproducibility_tracker.py` wrote to
`results/debug/debug_reproducibility_log.json` (debug run, 2026-05-20, CPU-only
Linux). The tracker did not record the other packages in `requirements.txt`
(for example `nltk`, `sentence-transformers`, `rank-bm25`, `scipy`, `matplotlib`,
`langchain` and `gradio`), and no reproducibility log for the full 6,600-run
experiment is committed. The pins resolve together on Python 3.12 but have not been checked
against the full run's environment.

### Models and external services

| Role | Identifier | Where to get it | Version pinned? |
|---|---|---|---|
| Generator | `llama3.1:8b` | [Ollama](https://ollama.com/library/llama3.1) | Tag only. **⚠ TO COMPLETE:** run `ollama list` and add the model digest (ID) |
| Primary judge | `qwen2.5:14b` | [Ollama](https://ollama.com/library/qwen2.5) | Tag only. **⚠ TO COMPLETE:** digest |
| Secondary judge | `gemma2:9b` | [Ollama](https://ollama.com/library/gemma2) | Tag only. **⚠ TO COMPLETE:** digest |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Hugging Face | Name only |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face | Name only |
| NER | `en_core_web_sm` (spaCy) | spaCy | Version not recorded. **⚠ TO COMPLETE:** `python -c "import spacy; print(spacy.load('en_core_web_sm').meta['version'])"` |
| Ollama server | **⚠ TO COMPLETE:** version (`ollama --version`) | https://ollama.com | — |

**⚠ TO COMPLETE — inference backend of the full run.** `agents/llm_wrapper.py`
can run the same model names through Groq instead of Ollama, mapping
`llama3.1:8b` → `llama-3.1-8b-instant` and both `qwen2.5:14b` and `gemma2:9b` →
`llama-3.3-70b-versatile`. `results/full/full_experiment_metrics.csv` records the
Ollama names (`llama3.1:8b` generator and `qwen2.5:14b` judge on all 6,600 rows),
while the committed debug log lists `llama-3.1-8b-instant`. State in the
manuscript which backend produced the reported results, and if Groq was used,
give the access dates.

## 2. Data shared in this repository

All files are plain CSV or JSON. Paths are relative to the repository root.

| File | Content | Size |
|---|---|---|
| `dft_research_studio/data/processed/dft_kg_nodes.csv` | Knowledge-graph nodes (`node_id,label,rung,class,family,unit,value,source`) | 19,726 nodes |
| `dft_research_studio/data/processed/dft_kg_relationships.csv` | Knowledge-graph edges (`paper_id,source_id,target_id,relationship_type,condition`) | 100,442 edges |
| `dft_research_studio/data/processed/_DFT-QA-120.csv` | Q-120 challenge set (`Question number,Question,Ground truth,Gold Standard Document`) | 120 questions |
| `dft_research_studio/data/processed/paper_id_mapping.json` | `paper_id` key → title fragment | 38 keys, 19 distinct titles |
| `dft_research_studio/data/processed/corpus_manifest.csv` | **New.** The 25 papers in the selection figure (Ancestors, Core Lineage, Competitors, Collaborators, Hubs / Databases), with whether each is in the graph, its `paper_id` keys and the Q-120 filenames that cite it | 25 papers (23 in the graph) |
| `results/full/full_results_raw.json` | Raw generated answers, full experiment | 6,600 runs |
| `results/full/full_experiment_metrics.csv` | Judge scores and retrieval metrics per run | 6,600 rows |
| `results/full/final_results_with_ci.csv`, `table3_generative_gap.csv` | Aggregates with bootstrap confidence intervals | — |
| `results/full/tables/*.csv` | Tables 5.1–6.10 and Graph Lift | 16 files |
| `results/full/charts/` | Figures (PNG, one SVG) | — |
| `results/debug/` | Single-question debug run, including `debug_reproducibility_log.json` | — |

Known properties of the shared data, stated so that readers are not surprised:

- In `dft_kg_relationships.csv`, 90,457 of the 100,442 rows have an empty
  `paper_id`, so only about 10% of edges are linked to a source paper.
- The Q-120 column `Gold Standard Document` lists file names, separated by
  commas, but many titles themselves contain commas. Split on the `.pdf`
  boundary instead (this is how `corpus_manifest.csv` was built). It contains 22
  distinct name strings, several of which are truncated variants of the same paper.
- **25 papers were selected but the graph has 23 `Paper` nodes.** The two that
  are not fully represented are *Peverati (2012), MN12-L*, which is absent (no
  `Paper` node and no relationship with an MN12 `paper_id`), and *Chai (2008),
  ωB97X-D*, which is partly present (no `Paper` node, but 3 relationships carry
  `paper_id = Chai2008_Disp`, and `paper_id_mapping.json` registers that key under
  the ωB97 paper's title). The study results were produced with the 23-paper
  graph as shipped. `corpus_manifest.csv` lists all 25 with an `in_kg_corpus`
  column, and the manuscript should state the corpus size as 23 papers (or
  explain the two exceptions), not 25.
- Several `paper_id` keys do not name the paper they are attached to, judged
  against the selection figure: `Baskerville2016/2022` is attached to the
  Mardirossian and Head-Gordon (2017) review, `Mardirossian2017` keys to the
  Peverati and Truhlar (2014) review, and `Caldeweyher2019` to the D3 paper
  (Grimme 2010). Only about 10% of edges carry a `paper_id` at all, so the
  effect is limited, but do not use the key names as author-year citations.
  The `notes` column of `corpus_manifest.csv` flags these rows.

## 3. Data not shared, and how to obtain it

| Not shared | Why | How to obtain / rebuild |
|---|---|---|
| The source-paper PDFs (`raw/DFT-Corpus-25`) | Third-party publications; **⚠ TO COMPLETE:** confirm redistribution is not permitted | Titles are in `corpus_manifest.csv`. **⚠ TO COMPLETE:** add the `doi` column (left blank rather than guessed) |
| The 75 distractor PDFs (`raw/Distractors`) | Third-party preprints | See below |
| ChromaDB vector stores (`VectorDBs/`) | Large, and derived from the PDFs | Built automatically when the target directory is missing or empty (`retrievers/standard_rag.py`), one store per distractor ratio |

**Distractor corpus.** It was downloaded with the `arxiv` Python client in
`notebooks/Archive_Extraction_Logic/01_Distrator_corpus_acquisition.ipynb`:

```python
arxiv.Search(
    query='abs:"density functional theory" AND abs:"benchmark"',
    max_results=80,
    sort_by=arxiv.SortCriterion.SubmittedDate,   # newest first
)
```

The first 75 results were saved as `Distractor_001.pdf` … `Distractor_075.pdf`.
Because the search is sorted by submission date, re-running it today returns a
different set, and the notebook did not store the arXiv identifiers.
`data/data_manager.py` uses every `*.pdf` found in the distractor folder, so the
number actually used is the number of files that were in it.
**⚠ TO COMPLETE:** confirm that number, recover the identifiers (arXiv-generated
PDFs normally print `arXiv:YYMM.NNNNN` in the first-page margin) from the original
`distractors.zip`, and add them to this repository as `distractor_manifest.csv`.

## 4. Reproducing the results

```bash
git clone https://github.com/hastikacheddy/dft-research-studio.git
cd dft-research-studio
pip install -r requirements.txt -c requirements-recorded.txt   # Python 3.12
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab stopwords wordnet
# Ollama: pull llama3.1:8b, qwen2.5:14b, gemma2:9b (see README)
```

**Paths.** `config/settings.py` defaults to the author's private Lightning.ai
mount (`/teamspace/uploads/Dissertation/…`), which will not exist on another
machine. Point it at the shipped data by adding to `.env`:

```env
DFT_BASE_DIR=/absolute/path/to/dft-research-studio/dft_research_studio/data/processed
DFT_CHROMA_DIR=/absolute/path/to/dft-research-studio/VectorDBs
```

`settings.py` then looks for the PDFs in `<DFT_BASE_DIR>/../raw/DFT-Corpus-25`
and `<DFT_BASE_DIR>/../raw/Distractors`, i.e.
`dft_research_studio/data/raw/DFT-Corpus-25` and `…/raw/Distractors` (this
directory is git-ignored). Then run `python main.py` as described in the README.

Without the PDFs, the **analysis** can still be reproduced from the shipped
results:

```bash
python -m dft_research_studio.visualization.generate_all_tables \
    --input results/full/full_experiment_metrics.csv --output results/full/tables
python -m dft_research_studio.visualization.generate_all_charts \
    --input results/full/full_experiment_metrics.csv --output results/full/charts
```

Parameters (chunk size 1000, overlap 100, top-k 5, distractor ratios
0 / 0.5 / 1 / 2 / 3, seed 42, temperature 0 for generation and judging) are in
`dft_research_studio/config/settings.py` and are written to the
reproducibility log of every run. The bootstrap confidence intervals in
`advanced_metrics.py` and `generate_all_tables.py` use a fixed seed (42);
`generate_all_charts.py` (`07_bootstrap_ci.png`) resamples with NumPy's global
generator and is not seeded when run on its own.

## 5. Licences

| Component | Licence |
|---|---|
| Code | MIT ([`LICENSE`](LICENSE)) |
| Knowledge graph, Q-120 and result files | **⚠ DECISION NEEDED.** `config/settings.py` declares `data_license = "CC BY-NC 4.0"`, which conflicts with MIT for the repository and restricts commercial use, whereas JCIM asks for data to be available without restrictions. Choose one licence (for example CC BY 4.0 or CC0) and state it here and in the manuscript. The graph is derived from third-party papers, so confirm that you are entitled to licence it |
| Llama 3.1 | Llama 3.1 Community License (Meta) |
| Qwen2.5-14B, Gemma 2 9B | Per their model cards. **⚠ TO COMPLETE** |
| `all-MiniLM-L6-v2` | Apache-2.0 |

## 6. Draft text for the manuscript's *Data and Software Availability* section

> The code, knowledge graph (19,726 nodes, 100,442 relationships), the 120-question
> challenge set (Q-120), the raw model outputs and the evaluated metrics for all
> 6,600 experimental runs are available under [LICENCE] at
> https://github.com/hastikacheddy/dft-research-studio (version 1.0.0; archived at
> [ZENODO DOI]). Library versions are given in `requirements-recorded.txt`; the
> study used Python 3.12.11. Language models were run through [Ollama X.Y.Z /
> Groq, accessed DATE]: `llama3.1:8b` (generator), `qwen2.5:14b` (primary judge)
> and `gemma2:9b` (secondary judge). Parameters are listed in
> `dft_research_studio/config/settings.py`. The full texts of the 23
> source publications in the knowledge graph are not redistributed because they
> are third-party copyrighted works; the 25 selected papers, with their titles
> and identifiers, are listed in
> `dft_research_studio/data/processed/corpus_manifest.csv`. The [N] distractor
> preprints were retrieved from arXiv with the query
> `abs:"density functional theory" AND abs:"benchmark"` (newest first) and are
> listed in `distractor_manifest.csv`. All figures and tables can be regenerated
> from the shipped results with the commands in the repository README.
