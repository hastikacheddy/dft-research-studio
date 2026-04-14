---
title: DFT Research Studio
emoji: 🧪
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.36.0
app_file: app_hf.py
pinned: true
license: mit
---

# DFT Research Studio

Auto-KGR: Knowledge-Graph-Grounded Retrieval for Quantum Chemistry

## Architecture
- 19,726-node DFT knowledge graph
- Dialectic Multi-Agent Debate (Advisor + Safety Officer)
- Four retrieval architectures: GraphRAG, Graph Deterministic, Standard RAG, Multi-Agent
- Outputs: PSI4 / ORCA / Gaussian 16 input files

## Setup
Set these secrets in the Space settings:
- `LITSERVE_URL` — your LitServe endpoint on Lightning.ai
- `SERVE_API_KEY` — your API key
