"""
Tab 3 — Auto-KGR Ingestion Lab
Shows the full auto-ingestion pipeline live:
1. Search ArXiv for new DFT papers
2. Download + extract text
3. LLM extracts nodes/relationships
4. Merge into KG (augmented files only)
5. Show what was added
"""

import gradio as gr
import os
import time
from typing import Generator

_ingester = None
_kg       = None
_llm      = None

def init_ingestion_engines(cfg, nodes_df, rels_df, llm):
    global _ingester, _kg, _llm
    from dft_research_studio.data.arxiv_ingester import ArXivIngester
    from dft_research_studio.agents.kg_query_engine import KGQueryEngine
    _llm = llm
    _kg  = KGQueryEngine(nodes_df, rels_df)
    _ingester = ArXivIngester(
        nodes_path=cfg.nodes_path,
        rels_path=cfg.rels_path,
        llm_client=llm,
        max_papers=3,
    )


def run_ingestion(keyword, max_papers):
    if _ingester is None:
        yield "⚠️ Run with --standalone first.", "", "", ""
        return

    log  = []
    rows = []

    log.append("=" * 56)
    log.append("🔄 AUTO-KGR PAPER INGESTION PIPELINE")
    log.append("=" * 56)
    log.append(f"  Keyword:    {keyword}")
    log.append(f"  Max papers: {max_papers}")
    log.append(f"  KG before:  {len(_ingester.nodes_df)} nodes, {len(_ingester.rels_df)} rels")
    log.append("")
    yield "\n".join(log), "", _make_kg_stats(), ""

    # Step 1: Search ArXiv
    log.append("📡 Step 1: Searching ArXiv...")
    yield "\n".join(log), "", _make_kg_stats(), ""
    _ingester.max_papers = int(max_papers)
    papers = _ingester.search_arxiv(keyword, max_results=int(max_papers)+2)
    log.append(f"   Found {len(papers)} new papers.")
    for p in papers:
        log.append(f"   ✓ {p.title[:55]}")
        log.append(f"     ArXiv: {p.arxiv_id} | {p.published}")
    log.append("")
    yield "\n".join(log), "", _make_kg_stats(), ""

    if not papers:
        log.append("⚠️ No new papers found. Try a different keyword.")
        yield "\n".join(log), "", _make_kg_stats(), ""
        return

    # Step 2-4: Process each paper
    for i, paper in enumerate(papers[:int(max_papers)]):
        log.append(f"📄 Paper {i+1}/{min(len(papers), int(max_papers))}: {paper.title[:50]}...")
        yield "\n".join(log), "", _make_kg_stats(), ""

        # Download + extract text
        log.append("   ⬇️  Downloading PDF...")
        yield "\n".join(log), "", _make_kg_stats(), ""
        text = _ingester._get_text(paper)
        word_count = len(text.split())
        log.append(f"   📝 Extracted {word_count} words from paper.")
        yield "\n".join(log), "", _make_kg_stats(), ""

        # LLM extraction
        log.append("   🤖 LLM extracting nodes & relationships...")
        yield "\n".join(log), "", _make_kg_stats(), ""
        nodes, rels = _ingester._extract(paper, text)
        log.append(f"   🔵 Extracted {len(nodes)} nodes, {len(rels)} relationships")
        for n in nodes[:5]:
            log.append(f"      Node: {n.get('node_id','?')} [{n.get('label','?')}]"
                       + (f" = {n.get('value')} {n.get('unit','')}" if n.get('value') else ""))
        for r in rels[:5]:
            log.append(f"      Rel:  {r.get('source_id','?')} —[{r.get('relationship_type','?')}]→ {r.get('target_id','?')}")
        log.append("")
        yield "\n".join(log), "", _make_kg_stats(), ""

        # Merge into KG
        log.append("   🔗 Merging into KG (augmented files only)...")
        yield "\n".join(log), "", _make_kg_stats(), ""
        nn, nr = _ingester._merge(nodes, rels, paper)
        _ingester._seen.add(paper.arxiv_id)
        log.append(f"   ✅ Added {nn} new nodes, {nr} new relationships")
        log.append(f"   ✅ Original CSVs preserved — augmented files updated")
        log.append("")
        yield "\n".join(log), "", _make_kg_stats(), ""

        rows.append({
            "Paper":    paper.title[:45],
            "ArXiv":    paper.arxiv_id,
            "Published":paper.published,
            "Nodes+":   nn,
            "Rels+":    nr,
            "Status":   "✅ Merged",
        })
        time.sleep(0.5)

    # Save augmented KG
    log.append("💾 Saving augmented KG (originals untouched)...")
    yield "\n".join(log), "", _make_kg_stats(), ""
    _ingester._save_kg()
    _ingester._save_seen()

    log.append("=" * 56)
    log.append(f"✅ INGESTION COMPLETE")
    log.append(f"   KG after: {len(_ingester.nodes_df)} nodes, {len(_ingester.rels_df)} rels")
    log.append(f"   New nodes: +{sum(r['Nodes+'] for r in rows)}")
    log.append(f"   New rels:  +{sum(r['Rels+'] for r in rows)}")
    log.append(f"   Saved to:  dft_kg_nodes_augmented.csv")
    log.append(f"            dft_kg_relationships_augmented.csv")
    log.append(f"   Original:  dft_kg_nodes.csv (UNCHANGED)")
    log.append(f"            dft_kg_relationships.csv (UNCHANGED)")
    log.append("=" * 56)

    # Build results table
    table_html = _make_results_table(rows)
    yield "\n".join(log), table_html, _make_kg_stats(), _make_kg_stats()


def _make_kg_stats():
    if _ingester is None:
        return "<p style='color:#64748b'>Engine not loaded.</p>"
    # Read ORIGINAL file from disk (not in-memory which gets updated during ingestion)
    import pandas as pd
    orig_nodes = len(pd.read_csv(_ingester.nodes_path))
    orig_rels  = len(pd.read_csv(_ingester.rels_path))

    # Check augmented files
    aug_path = os.path.join(
        os.path.dirname(_ingester.nodes_path),
        "dft_kg_nodes_augmented.csv"
    )
    aug_nodes = aug_rels = 0
    if os.path.exists(aug_path):
        import pandas as pd
        aug_nodes = len(pd.read_csv(aug_path))
        aug_rels  = len(pd.read_csv(aug_path.replace("nodes","relationships")))

    return f"""
<div style='font-family:monospace;font-size:0.85em;color:#e6edf3;padding:10px;background:#161b22;border-radius:8px;border-left:3px solid #3fb950'>
<b style='color:#3fb950'>📊 Knowledge Graph Status</b><br><br>
<b>Original KG (preserved):</b><br>
&nbsp;&nbsp;Nodes: {orig_nodes:,} &nbsp;|&nbsp; Relationships: {orig_rels:,}<br><br>
<b>Augmented KG (auto-ingested):</b><br>
&nbsp;&nbsp;Nodes: {aug_nodes:,} (+{aug_nodes-orig_nodes:,}) &nbsp;|&nbsp; Rels: {aug_rels:,} (+{aug_rels-orig_rels:,})<br><br>
<b style='color:#64748b'>Original CSVs are never modified ✅</b>
</div>"""


def _make_results_table(rows):
    if not rows:
        return "<p style='color:#64748b'>No papers processed yet.</p>"
    h = ("<table style='width:100%;border-collapse:collapse;font-family:monospace;"
         "font-size:0.82em;color:#e6edf3'>"
         "<tr style='background:#1e3a5f;color:#93c5fd'>"
         "<th style='padding:6px'>Paper</th><th>ArXiv</th>"
         "<th>Published</th><th>Nodes+</th><th>Rels+</th><th>Status</th></tr>")
    body = ""
    for i, r in enumerate(rows):
        bg = "#0d1117" if i%2==0 else "#161b22"
        body += (f"<tr style='background:{bg}'>"
                 f"<td style='padding:5px'>{r['Paper']}</td>"
                 f"<td>{r['ArXiv']}</td><td>{r['Published']}</td>"
                 f"<td style='color:#3fb950;text-align:center'>{r['Nodes+']}</td>"
                 f"<td style='color:#3fb950;text-align:center'>{r['Rels+']}</td>"
                 f"<td>{r['Status']}</td></tr>")
    return h + body + "</table>"


def build_ingestion_tab():
    with gr.Tab("🔄 Auto-KGR Ingestion"):
        gr.Markdown(
            "### Auto-KGR Paper Ingestion Pipeline\n"
            "Automatically searches ArXiv for new DFT papers, extracts knowledge graph "
            "nodes and relationships using LLM, and merges them into an augmented KG.\n\n"
            "**Your original CSVs are never modified.**"
        )

        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("#### Search Parameters")
                keyword_input = gr.Textbox(
                    label="ArXiv Search Keyword",
                    value="density functional theory benchmark",
                    placeholder="e.g. DFT dispersion correction, hybrid functional benchmark",
                )
                max_papers_slider = gr.Slider(
                    minimum=1, maximum=5, value=2, step=1,
                    label="Max Papers to Process",
                    info="Each paper takes ~30-60 seconds to process",
                )
                ingest_btn = gr.Button("🚀 Start Auto-Ingestion", variant="primary")

                gr.Markdown("#### KG Status")
                kg_stats = gr.HTML(_make_kg_stats())

                gr.Markdown(
                    "#### What happens:\n"
                    "1. 📡 Search ArXiv API for new DFT papers\n"
                    "2. ⬇️ Download PDF and extract text\n"
                    "3. 🤖 LLM extracts nodes & relationships\n"
                    "4. 🔗 Merge into augmented KG\n"
                    "5. 💾 Save to augmented CSV files\n\n"
                    "**Original files preserved:**\n"
                    "- `dft_kg_nodes.csv` ✅\n"
                    "- `dft_kg_relationships.csv` ✅"
                )

            with gr.Column(scale=6):
                gr.Markdown("#### Live Pipeline Log")
                log_box = gr.Textbox(
                    label="", lines=22, interactive=False,
                    placeholder="Pipeline log appears here...",
                )
                gr.Markdown("#### Papers Processed")
                results_table = gr.HTML(
                    "<p style='color:#64748b'>Results appear after ingestion.</p>"
                )

        ingest_btn.click(
            fn=run_ingestion,
            inputs=[keyword_input, max_papers_slider],
            outputs=[log_box, results_table, kg_stats, kg_stats],
        )

    return kg_stats
