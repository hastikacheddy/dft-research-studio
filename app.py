from __future__ import annotations
import argparse, logging, os, re, time
from typing import Generator

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

import gradio as gr, requests
from rouge_score import rouge_scorer
from dft_research_studio.utils.logging_config import setup_logging
try:
    from dft_research_studio.agents.tab3_ingestion import build_ingestion_tab, init_ingestion_engines
    HAS_INGESTION = True
except ImportError:
    HAS_INGESTION = False


setup_logging(run_tag="app", log_dir="logs")
logger = logging.getLogger(__name__)

LITSERVE_URL  = os.getenv("LITSERVE_URL","http://localhost:8000/predict")
SERVE_API_KEY = os.getenv("SERVE_API_KEY","")
EXPERIMENT_TYPES = [
    # Part 1: Pure LLM Baselines
    "Zero-Shot Prompting",
    "Template Prompting",
    "Chain-of-Thought (CoT) Prompting",
    # Part 2: Standard IR Baselines
    "BM25 Retriever",
    "Cross-Encoder Reranker",
    "BM25 + Reranker RAG",
    "Standard RAG",
    # Part 3: Graph-Based Architectures
    "GraphRAG",
    "Graph Deterministic",
    "Multi-Agent System (GraphRAG Fallback)",
    "Multi-Agent System (BM25+Reranker Fallback)",
]
TASK_TYPES = [
    "Non-covalent interactions (S66, NCIBLIND10)",
    "Main-group thermochemistry (W4-11, G3/05)",
    "Transition metal complexes (TMC32, MOR41)",
    "Reaction barrier heights (BH76, HTBH38)",
    "Dispersion-dominated systems (S22, S66x8)",
    "Excited states / charge transfer (TDDFT)",
    "Conformational energies (SCONF, PCONF21)",
    "Large non-covalent complexes (S30L)",
    "Hydrogen bonding (HB375)",
    "Pi-pi stacking interactions",
    "Metal-organic frameworks (MOR41)",
    "Atomisation energies (W4-11)",
    "Ionisation potentials (IP13)",
    "Electron affinities (EA13)",
    "Proton affinities (PA8)",
    "Thermochemical kinetics (TKNC306)",
]
BASIS_SETS = [
    "def2-TZVP","def2-QZVP","def2-SVP","def2-TZVPP",
    "6-311+G(2d,2p)","6-311+G(d,p)","6-31G*","6-31+G*",
    "aug-cc-pVTZ","aug-cc-pVDZ","cc-pVTZ","cc-pVDZ","cc-pVQZ",
    "ma-def2-TZVP","ma-def2-SVP",
    "jorge-TZP","jorge-DZP",
    "def2-TZVP + RIJCOSX (ORCA)","def2-QZVP + RI (PSI4)",
]

_runner = None
_viz    = None
_qa_pairs = []
_debate_orchestrator = None

def _init_standalone():
    global _runner, _viz, _qa_pairs, _debate_orchestrator
    from dft_research_studio.config import Config
    from dft_research_studio.data import DFTDataManager
    from dft_research_studio.utils import ExperimentOrchestrator
    from dft_research_studio.visualization import GraphVisualizer
    from dft_research_studio.agents import DebateOrchestrator
    logger.info("Loading engines...")
    cfg = Config()
    dm  = DFTDataManager(cfg)
    dm.remove_disconnected_nodes()
    try:
        _runner = ExperimentOrchestrator(dm, cfg)
    except Exception as exc:
        logger.warning("ExperimentOrchestrator failed (no VectorDBs?): %s", exc)
        _runner = None
    _viz    = GraphVisualizer(dm.graph, dm.nodes_df, dm.rels_df)
    _qa_pairs = _runner.qa_pairs
    _debate_orchestrator = DebateOrchestrator(
        rag_engine=_runner.engines.rag(0.0),
        graph_engine=_runner.engines.graph(0.0),
        llm=_runner.engines.default_llm,
        max_rounds=3,
        nodes_df=dm.nodes_df,
        rels_df=dm.rels_df,
    )
    logger.info("All engines ready.")
    # Init Auto-KGR ingestion engine
    if HAS_INGESTION:
        init_ingestion_engines(cfg, dm.nodes_df, dm.rels_df, _runner.engines.default_llm)
        logger.info("Auto-KGR ingestion engine ready.")


def _call_backend(question, exp_type, ratio):
    if _runner is not None:
        return _runner.get_chatbot_answer(question, exp_type, ratio)
    headers = {"X-API-Key": SERVE_API_KEY} if SERVE_API_KEY else {}
    try:
        r = requests.post(LITSERVE_URL,
            json={"question":question,"experiment_type":exp_type,"distractor_ratio":ratio},
            headers=headers, timeout=120)
        r.raise_for_status()
        d = r.json()
        return d["answer"], d["metrics"], d["trace"]
    except Exception as exc:
        return f"Error: {exc}", "", ""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');
*{font-family:'JetBrains Mono','Courier New',monospace !important}
body,.gradio-container,.main,.wrap,.gap,.block,.form,.panel,.container,
.tabitem,div.svelte-1kyusxi,div.svelte-1gfkzqq,.tabs,.tab-content{
    background:#141d2b !important;font-family:'JetBrains Mono','Courier New',monospace !important}
.gradio-container{max-width:1400px !important}
h1{color:#9fef00 !important;font-size:1.2rem !important;font-weight:700 !important;letter-spacing:0.08em !important}
h2,h3{color:#a4b1cd !important;font-weight:700 !important}
p,li,span{color:#a4b1cd !important;font-size:0.82rem !important}
.tab-nav{background:#111927 !important;border-bottom:1px solid #1f2d3d !important}
.tab-nav button{font-size:0.72rem !important;font-weight:700 !important;color:#5a6a7e !important;
    background:transparent !important;text-transform:uppercase !important;
    letter-spacing:0.1em !important;border-bottom:2px solid transparent !important}
.tab-nav button.selected{color:#9fef00 !important;border-bottom:2px solid #9fef00 !important;background:transparent !important}
.tab-nav button:hover{color:#a4b1cd !important}
label,.label-wrap span{font-size:0.68rem !important;font-weight:700 !important;
    text-transform:uppercase !important;letter-spacing:0.1em !important;color:#5a6a7e !important}
input,textarea,select,.scroll-hide{background:#0d1520 !important;color:#9fef00 !important;
    font-family:'JetBrains Mono',monospace !important;font-size:0.8rem !important;
    caret-color:#9fef00 !important;border-color:#1f2d3d !important}
input::placeholder,textarea::placeholder{color:#3a4a5e !important}
input:focus,textarea:focus,select:focus{outline:none !important;border-color:#9fef00 !important}
button.primary,button.primary:hover{background:#9fef00 !important;color:#141d2b !important;
    font-weight:700 !important;letter-spacing:0.06em !important;text-transform:uppercase !important}
button.primary:hover{background:#c5ff4d !important}
button.secondary{background:transparent !important;color:#9fef00 !important;
    border:1px solid #9fef00 !important;font-weight:700 !important}
.message.user{background:#0c2040 !important;color:#38bdf8 !important}
.message.bot{background:#0d1520 !important;color:#a4b1cd !important}
.chatbot,.chat-wrap{background:#0d1520 !important;border-color:#1f2d3d !important}
code,pre,.code-wrap,.cm-editor,.cm-line{background:#0d1520 !important;
    color:#9fef00 !important;border-color:#1f2d3d !important}
input[type=range]{accent-color:#9fef00 !important}
.prose,.markdown,p,li,ol,ul{color:#a4b1cd !important}
strong,b{color:#e2e8f0 !important}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#141d2b}
::-webkit-scrollbar-thumb{background:#1f2d3d;border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:#2d3f52}
.metrics-container{margin-top:8px;padding:6px 10px;border-left:2px solid #9fef00;
    font-size:0.7rem;color:#5a6a7e;background:#0d1520}
"""

def _get_cache():
    import json, os
    path = "/teamspace/studios/this_studio/demo_cache.json"
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            return {}
    return {}

def _save_cache(cache):
    import json
    path = "/teamspace/studios/this_studio/demo_cache.json"
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)

def _get_cache():
    import json, os
    path = "/teamspace/studios/this_studio/demo_cache.json"
    try:
        return json.load(open(path)) if os.path.exists(path) else {}
    except Exception:
        return {}

def _save_cache(cache):
    import json
    with open("/teamspace/studios/this_studio/demo_cache.json","w") as f:
        json.dump(cache, f, indent=2)

def run_all_architectures(question, selected_arch, ratio):
    import concurrent.futures, time, re
    ALL_ARCHS = [
        "Zero-Shot Prompting",
        "Template Prompting",
        "Chain-of-Thought (CoT) Prompting",
        "BM25 Retriever",
        "Cross-Encoder Reranker",
        "BM25 + Reranker RAG",
        "Standard RAG",
        "GraphRAG",
        "Graph Deterministic",
        "Multi-Agent System (GraphRAG Fallback)",
        "Multi-Agent System (BM25+Reranker Fallback)",
    ]

    cache     = _get_cache()
    cache_key = question.strip().lower()[:100]
    cached    = cache.get(cache_key, {})
    results   = {}
    to_run    = []

    for arch in ALL_ARCHS:
        if arch == selected_arch:
            to_run.append(arch)
        elif arch in cached:
            results[arch] = dict(cached[arch])
            results[arch]["from_cache"] = True
        else:
            to_run.append(arch)

    def run_one(arch):
        t0 = time.time()
        try:
            answer, metrics, trace = _call_backend(question, arch, ratio)
            latency   = time.time() - t0
            ans_lower = answer.lower()

            # KG nodes hit
            nodes = len(set(re.findall(r"VR_[A-Za-z0-9_().-]+", answer)))

            # Citation detection — benchmark names, MAE values, paper IDs
            has_citation = bool(
                nodes > 0 or
                re.search(r"[A-Z][a-z]+20\d\d", answer) or
                any(b in answer for b in ["S66","S22","GMTKN55","TMC32","BH76","W4-11","MAE","MAD","kcal","WTMAD","pm"]) or
                re.search(r"\d+\.\d+\s*(kcal|%|pm|kJ)", answer)
            )
            ans_len = len(answer.split())

            # Error type classification
            if not answer or ans_len < 8 or "not implemented" in ans_lower:
                error_type = "Type I"
            elif any(w in ans_lower for w in ["not explicitly","not stated","not provided","not found","no relevant","cannot determine","not available"]):
                error_type = "Type I"
            elif any(w in ans_lower for w in ["might","possibly","i think","probably","i am not sure","i cannot confirm"]):
                error_type = "Type II"
            elif nodes == 0 and "however" in ans_lower and ans_len < 50:
                error_type = "Type III"
            else:
                error_type = "none"

            return arch, {
                "answer":       answer,
                "nodes_hit":    nodes,
                "has_citation": has_citation,
                "ans_len":      ans_len,
                "latency":      round(latency, 1),
                "error_type":   error_type,
                "metrics":      metrics,
                "from_cache":   False,
            }
        except Exception as exc:
            return arch, {
                "answer": str(exc), "nodes_hit": 0, "has_citation": False,
                "ans_len": 0, "latency": 0.0, "error_type": "Type I",
                "metrics": "", "from_cache": False,
            }

    if to_run:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futs = {ex.submit(run_one, a): a for a in to_run}
            for f in concurrent.futures.as_completed(futs):
                a, d = f.result()
                results[a] = d

    new_entries = {a:d for a,d in results.items()
                   if a != selected_arch and not d.get("from_cache")}
    if new_entries:
        cached.update(new_entries)
        cache[cache_key] = cached
        _save_cache(cache)

    return results


def build_comparison_html(results, selected_arch):
    if not results:
        return ""

    # Winner = KG nodes + citation + words. Penalize errors & zero answers.
    def score(a):
        d = results[a]
        if not isinstance(d, dict):
            return (-1, 0, 0, 0)
        nodes = d.get("nodes_hit", 0)
        cite  = int(d.get("has_citation", False))
        words = d.get("ans_len", 0)
        err   = d.get("error_type", "none")
        penalty = 0 if err == "none" else -10
        if words == 0:
            penalty = -20
        return (penalty, nodes, cite, words)

    best_arch  = max(results.keys(), key=score)
    total_time = sum(r.get("latency", 0) for r in results.values())

    rows = ""
    for arch, data in results.items():
        latency    = data.get("latency", 0)
        nodes      = data.get("nodes_hit", 0)
        has_cite   = data.get("has_citation", False)
        error_type = data.get("error_type", "—")
        is_best    = arch == best_arch
        is_sel     = arch == selected_arch
        cite_html  = "<span style='color:#9fef00;font-weight:700'>✓</span>" if has_cite else "<span style='color:#ef4444'>✗</span>"

        lat_c  = "color:#9fef00" if latency < 2 else "color:#f59e0b" if latency < 5 else "color:#ef4444"
        node_c = "color:#9fef00" if nodes > 0 else "color:#3a4a5e"

        # Error type badge
        if error_type == "Type I":
            err_html = "<span style='font-size:8px;padding:1px 5px;border-radius:2px;background:#1a0a0a;color:#ef4444;border:1px solid #4d1d1d'>Type I</span>"
        elif error_type == "Type II":
            err_html = "<span style='font-size:8px;padding:1px 5px;border-radius:2px;background:#1a1000;color:#f59e0b;border:1px solid #4d3000'>Type II</span>"
        elif error_type == "Type III":
            err_html = "<span style='font-size:8px;padding:1px 5px;border-radius:2px;background:#0a0a1a;color:#38bdf8;border:1px solid #1d3a5a'>Type III</span>"
        else:
            err_html = "<span style='color:#3a4a5e;font-size:9px'>none</span>"

        row_bg = "background:#0c1929" if is_sel else "background:#0a150a" if is_best else ""
        tags   = ""
        if is_best: tags += "<span style='font-size:8px;padding:1px 5px;border-radius:2px;font-weight:700;background:#0a1a0a;color:#9fef00;border:1px solid #1d4d1d;margin-left:4px'>BEST</span>"
        if is_sel:  tags += "<span style='font-size:8px;padding:1px 5px;border-radius:2px;font-weight:700;background:#0c2040;color:#38bdf8;border:1px solid #1d3a5a;margin-left:4px'>SELECTED</span>"
        star = "★ " if is_best else ""

        rows += f"""<tr style='{row_bg}'>
            <td style='padding:5px 10px;color:#a4b1cd;white-space:nowrap;font-size:10px'>{star}{arch}{tags}</td>
            <td style='padding:5px 10px;text-align:center;font-size:11px'>{cite_html}</td>
            <td style='padding:5px 10px;color:#a4b1cd;font-size:10px'>{data.get("ans_len",0)}</td>
            <td style='padding:5px 10px;{lat_c};white-space:nowrap;font-size:10px'>{latency}s</td>
            <td style='padding:5px 10px;{node_c};text-align:center;font-size:10px'>{nodes if nodes else "—"}</td>
            <td style='padding:5px 10px;text-align:center'>{err_html}</td>
        </tr>"""

    return f"""
<div style='background:#0d1520;border-radius:4px;overflow:hidden;margin-top:12px;font-family:Courier New,monospace'>
  <div style='background:#111927;padding:6px 12px;display:flex;align-items:center;gap:8px'>
    <span style='font-size:8px;color:#5a6a7e;text-transform:uppercase;letter-spacing:0.12em;font-weight:700'>
      // architecture comparison · all 11 ran in background
    </span>
    <span style='font-size:8px;color:#9fef00;margin-left:auto'>● complete · {round(total_time,1)}s total</span>
  </div>
  <table style='width:100%;border-collapse:collapse;font-size:10px'>
    <tr style='background:#111927'>
      <th style='padding:5px 10px;color:#5a6a7e;text-align:left;font-size:8px;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1f2d3d'>architecture</th>
      <th style='padding:5px 10px;color:#5a6a7e;text-align:left;font-size:8px;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1f2d3d'>citation</th>
      <th style='padding:5px 10px;color:#5a6a7e;text-align:left;font-size:8px;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1f2d3d'>words</th>
      <th style='padding:5px 10px;color:#5a6a7e;text-align:left;font-size:8px;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1f2d3d'>latency</th>
      <th style='padding:5px 10px;color:#5a6a7e;text-align:left;font-size:8px;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1f2d3d'>KG nodes</th>
      <th style='padding:5px 10px;color:#5a6a7e;text-align:left;font-size:8px;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid #1f2d3d'>error type</th>
    </tr>
    {rows}
  </table>
</div>"""


def build_meta_verdict_html(results, selected_arch, question):
    """Pick winner deterministically, use LLM only for explanation."""
    if not results or _runner is None:
        return ""

    # Deterministic winner — same logic as comparison table
    def _score(a):
        d = results[a]
        if not isinstance(d, dict): return (-1, 0, 0, 0)
        nodes = d.get("nodes_hit", 0)
        cite  = int(d.get("has_citation", False))
        words = d.get("ans_len", 0)
        err   = d.get("error_type", "none")
        penalty = 0 if err == "none" else -10
        if words == 0: penalty = -20
        return (penalty, nodes, cite, words)

    best_arch  = max(results.keys(), key=_score)
    best_data  = results[best_arch] if isinstance(results[best_arch], dict) else {}
    best_nodes = best_data.get("nodes_hit", 0)
    best_cite  = best_data.get("has_citation", False)
    best_words = best_data.get("ans_len", 0)

    summary = "\n".join(
        f"  {arch}: nodes={d.get('nodes_hit',0)} citation={d.get('has_citation',False)} words={d.get('ans_len',0)} latency={d.get('latency',0)}s error={d.get('error_type','none')}"
        for arch, d in results.items() if isinstance(d, dict)
    )

    reason  = ""
    pattern = ""
    try:
        llm = _runner.engines.default_llm
        try:
            from dft_research_studio.agents.meta_reasoning_engine import REASONING_MODEL
            import types
            meta_llm = types.SimpleNamespace()
            meta_llm.client     = llm.client
            meta_llm.model_name = REASONING_MODEL
            def _gen(prompt, max_tokens=400):
                resp = meta_llm.client.chat.completions.create(
                    model=meta_llm.model_name,
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content, 0, 0
            meta_llm.generate = _gen
        except Exception:
            meta_llm = llm

        prompt = f"""/no_think
The winner is already determined: {best_arch}.
Explain WHY {best_arch} performed best for this DFT query.

QUERY: {question}
RESULTS:
{summary}
WINNER: {best_arch} (nodes={best_nodes}, citation={best_cite}, words={best_words})

Reply in EXACTLY this format with no preamble:
REASON: <one clear sentence why {best_arch} retrieved better — reference KG traversal, retrieval mechanism, or query type>
PATTERN: <one clear sentence generalising when to prefer this architecture vs others>

Do NOT start with "Okay" or "Let me think". Be direct and technical."""

        response, *_ = meta_llm.generate(prompt, max_tokens=500)

        for line in response.split("\n"):
            l = line.strip()
            if l.startswith("REASON:"):  reason  = l.split(":",1)[1].strip()
            if l.startswith("PATTERN:"): pattern = l.split(":",1)[1].strip()

        if not reason:
            lines = [l for l in response.strip().split("\n") if len(l) > 20]
            reason = lines[0][:200] if lines else f"{best_arch} achieved {best_nodes} KG nodes with citation verification."
        if not pattern:
            pattern = "Graph-based architectures outperform dense RAG for structured KG lookups."

    except Exception as exc:
        reason  = f"{best_arch} achieved {best_nodes} KG node hits with {'citation verified' if best_cite else 'no citation'}."
        pattern = "Graph-based architectures outperform dense RAG for structured KG lookups."

    return f"""
<div style='background:#150a1e;border-left:2px solid #c084fc;border-radius:0 4px 4px 0;padding:10px 12px;margin-top:8px;font-family:Courier New,monospace'>
  <div style='font-size:8px;color:#c084fc;text-transform:uppercase;letter-spacing:0.12em;font-weight:700;margin-bottom:8px'>
    // meta_reasoner · qwen3-32b · architecture selection verdict
  </div>
  <div style='font-size:10px;color:#a4b1cd;line-height:1.8;margin-bottom:4px'>
    <span style='font-size:8px;padding:1px 6px;border-radius:2px;font-weight:700;background:#0a1a0a;color:#9fef00;border:1px solid #1d4d1d;margin-right:6px'>WINNER</span>
    <span style='color:#9fef00;font-weight:700'>{best_arch}</span> (KG nodes: {best_nodes} · citation: {'✓' if best_cite else '✗'} · {best_words} words)
  </div>
  <div style='font-size:11px;color:#a4b1cd;line-height:1.8;margin-bottom:4px;word-wrap:break-word;white-space:normal'>
    <span style='font-size:8px;padding:1px 6px;border-radius:2px;font-weight:700;background:#0a0e1a;color:#38bdf8;border:1px solid #1d3a5a;margin-right:6px'>REASON</span>
    {reason}
  </div>
  <div style='font-size:10px;color:#a4b1cd;line-height:1.8'>
    <span style='font-size:8px;padding:1px 6px;border-radius:2px;font-weight:700;background:#150a1e;color:#c084fc;border:1px solid #3b1a5a;margin-right:6px'>PATTERN</span>
    {pattern}
  </div>
</div>"""


def _quick_q1(h,e,r): yield from research_query_stream("MAE of PBE0 on S66?",h,e,r)
def _quick_q2(h,e,r): yield from research_query_stream("Best functional for TMC32?",h,e,r)
def _quick_q3(h,e,r): yield from research_query_stream("Why does B3LYP fail for barrier heights?",h,e,r)

def research_query_stream(user_query, history, exp_type, ratio):
    if not user_query.strip(): yield history, "", "", ""; return
    history = list(history)
    history.append({"role":"user","content":user_query})
    history.append({"role":"assistant","content":"Querying knowledge graph..."})
    yield history, "", "", ""
    answer, metrics_raw, trace_data = _call_backend(user_query, exp_type, ratio)
    graph_archs = ["GraphRAG", "Graph Deterministic", "Multi-Agent System (GraphRAG Fallback)"]
    show_graph = any(a in exp_type for a in graph_archs)
    graph_html = "" if show_graph else ""
    if _viz and ("Graph" in exp_type or "Multi-Agent" in exp_type):
        found_ids = list(set(re.findall(r"ID:\s*([\w\d_-]+)", str(trace_data))))
        if found_ids:
            try:
                import tempfile
                from pyvis.network import Network
                G = _viz.nx_graph
                nodes_to_show = set(found_ids)
                for nid in found_ids:
                    if nid in G:
                        nodes_to_show.update(list(G.successors(nid))[:8])
                        nodes_to_show.update(list(G.predecessors(nid))[:8])
                sub = G.subgraph(list(nodes_to_show)[:40])
                net = Network(height="400px",width="100%",directed=True,bgcolor="#0d1117",font_color="#e6edf3")
                net.set_options('{"physics":{"forceAtlas2Based":{"gravitationalConstant":-80,"springLength":120},"solver":"forceAtlas2Based","minVelocity":0.75},"edges":{"arrows":"to","color":{"color":"#30363d","highlight":"#00f3ff"}}}')
                for node in sub.nodes():
                    attrs = G.nodes[node]
                    color = "#ff7b72" if node in found_ids else "#388bfd"
                    size  = 18 if node in found_ids else 8
                    title = f"ID: {node}<br>Type: {attrs.get('label','Node')}"
                    if attrs.get("value") and str(attrs.get("value")) != "nan":
                        title += f"<br>Value: {attrs['value']} {attrs.get('unit','')}"
                    net.add_node(node, label=str(node)[:18], color=color, size=size, title=title)
                for u,v,data in sub.edges(data=True):
                    net.add_edge(u,v,title=data.get("relationship",""),color="#30363d")
                tmp = tempfile.NamedTemporaryFile(suffix=".html",delete=False)
                net.save_graph(tmp.name)
                html_content = open(tmp.name).read()
                os.unlink(tmp.name)
                graph_html = f'<iframe srcdoc="{html_content.replace(chr(34),chr(39))}" style="width:100%;height:420px;border:none;border-radius:8px;" sandbox="allow-scripts allow-same-origin"></iframe>'
            except Exception as exc:
                logger.warning("Graph render failed: %s", exc)
    rouge_disp = ""
    for pair in _qa_pairs:
        if pair["question"].strip().lower() in user_query.lower():
            s = rouge_scorer.RougeScorer(["rougeL"],use_stemmer=True).score(pair["ground_truth"],answer)["rougeL"].fmeasure
            rouge_disp = f" | ROUGE-L: {s:.3f}"; break
    partial = ""
    for word in answer.split(" "):
        partial += word + " "
        history[-1] = {"role":"assistant","content":partial}
        yield history, graph_html, "", ""
        time.sleep(0.01)
    history[-1] = {"role":"assistant","content":partial+f"\n\n<div class='metrics-container'>⚙ {exp_type} | noise={ratio} | {metrics_raw}{rouge_disp}</div>"}
    yield history, graph_html, "<p style='color:#5a6a7e;font-family:Courier New,monospace;font-size:10px'>// running all architectures in background...</p>", ""

    # Run all architectures in background
    all_results = run_all_architectures(user_query, exp_type, ratio)
    cmp_html    = build_comparison_html(all_results, exp_type)
    meta_html   = build_meta_verdict_html(all_results, exp_type, user_query)
    yield history, graph_html, cmp_html, meta_html

def run_debate(molecule, smiles, xyz, task, basis, charge, mult):
    if not molecule.strip(): yield "Enter a molecule name.","","","","_No molecule._",""; return
    if _debate_orchestrator is None: yield "Run with --standalone.","","","","",""; return
    for t,p,o,g,s,m in _debate_orchestrator.run_streaming(
        task=task.split(" (")[0], molecule=molecule, xyz=xyz, smiles=smiles,
        charge=int(charge), multiplicity=int(mult)):
        yield t,p,o,g,s,m

def get_model_info():
    return (
        "**Retrieval agents:** llama-3.3-70b-versatile (Advisor + Safety Officer)\n\n"
        "**Meta-reasoning:** qwen/qwen3-32b (Criteria evaluation + Strategy adaptation + Process reasoning)"
    )

def followup_chat_fn(user_message, history, psi4_ctx, transcript_ctx, summary_ctx):
    if not user_message.strip(): yield history; return
    history = list(history)
    history.append({"role":"user","content":user_message})
    history.append({"role":"assistant","content":"..."})
    yield history
    if _runner is None:
        history[-1] = {"role":"assistant","content":"Run with --standalone."}
        yield history; return
    llm = _runner.engines.default_llm
    prompt = f"""DFT Research Studio assistant. Debate outcome: {summary_ctx[:800]}
Input file: {psi4_ctx[:600]}. Answer the user accurately.
USER: {user_message}"""
    resp, *_ = llm.generate(prompt, max_tokens=400)
    partial = ""
    for word in resp.split(" "):
        partial += word + " "
        history[-1] = {"role":"assistant","content":partial}
        yield history
        time.sleep(0.01)

def build_demo(standalone):
    mode = "Standalone" if standalone else f"LitServe → {LITSERVE_URL}"
    with gr.Blocks(title="DFT Research Studio") as demo:
        gr.HTML(f"""
<div style='background:#0d1520;padding:12px 20px;border-bottom:1px solid #1f2d3d;font-family:JetBrains Mono,Courier New,monospace;margin-bottom:8px'>
  <div style='display:flex;align-items:center;gap:12px;margin-bottom:10px'>
    <div style='display:flex;gap:6px'>
      <div style='width:12px;height:12px;border-radius:50%;background:#ff5f57;box-shadow:0 0 6px #ff5f57'></div>
      <div style='width:12px;height:12px;border-radius:50%;background:#febc2e;box-shadow:0 0 6px #febc2e'></div>
      <div style='width:12px;height:12px;border-radius:50%;background:#28c840;box-shadow:0 0 6px #28c840'></div>
    </div>
    <span style='font-size:16px;font-weight:700;color:#9fef00;letter-spacing:0.1em;text-shadow:0 0 10px #9fef0088'>AUTO-KGR</span>
    <span style='font-size:16px;color:#a4b1cd;font-weight:400'>// research studio</span>
    <div style='margin-left:auto;display:flex;gap:8px'>
      <span style='font-size:10px;padding:3px 10px;border-radius:2px;font-weight:700;background:#0a1a0a;color:#9fef00;border:1px solid #9fef00;box-shadow:0 0 8px #9fef0055;text-shadow:0 0 6px #9fef00'>{mode.upper()}</span>
      <span style='font-size:10px;padding:3px 10px;border-radius:2px;font-weight:700;background:#0a0e1a;color:#38bdf8;border:1px solid #38bdf8;box-shadow:0 0 8px #38bdf855;text-shadow:0 0 6px #38bdf8'>LLAMA-3.3-70B</span>
      <span style='font-size:10px;padding:3px 10px;border-radius:2px;font-weight:700;background:#150a1e;color:#c084fc;border:1px solid #c084fc;box-shadow:0 0 8px #c084fc55;text-shadow:0 0 6px #c084fc'>QWEN3-32B</span>
    </div>
  </div>
  <div style='display:flex;gap:0;border-top:1px solid #1f2d3d;padding-top:10px'>
    <div style='padding:0 20px 0 0;border-right:1px solid #1f2d3d;margin-right:20px'>
      <div style='font-size:20px;font-weight:700;color:#9fef00;text-shadow:0 0 8px #9fef0066'>19,726</div>
      <div style='font-size:9px;color:#5a6a7e;text-transform:uppercase;letter-spacing:0.12em;margin-top:2px'>KG nodes</div>
    </div>
    <div style='padding:0 20px;border-right:1px solid #1f2d3d;margin-right:20px'>
      <div style='font-size:20px;font-weight:700;color:#9fef00;text-shadow:0 0 8px #9fef0066'>97,924</div>
      <div style='font-size:9px;color:#5a6a7e;text-transform:uppercase;letter-spacing:0.12em;margin-top:2px'>relationships</div>
    </div>
    <div style='padding:0 20px;border-right:1px solid #1f2d3d;margin-right:20px'>
      <div style='font-size:20px;font-weight:700;color:#9fef00;text-shadow:0 0 8px #9fef0066'>25</div>
      <div style='font-size:9px;color:#5a6a7e;text-transform:uppercase;letter-spacing:0.12em;margin-top:2px'>DFT papers</div>
    </div>
    <div style='padding:0 20px;border-right:1px solid #1f2d3d;margin-right:20px'>
      <div style='font-size:20px;font-weight:700;color:#9fef00;text-shadow:0 0 8px #9fef0066'>11</div>
      <div style='font-size:9px;color:#5a6a7e;text-transform:uppercase;letter-spacing:0.12em;margin-top:2px'>architectures</div>
    </div>
    <div style='padding:0 20px;border-right:1px solid #1f2d3d;margin-right:20px'>
      <div style='font-size:20px;font-weight:700;color:#9fef00;text-shadow:0 0 8px #9fef0066'>5</div>
      <div style='font-size:9px;color:#5a6a7e;text-transform:uppercase;letter-spacing:0.12em;margin-top:2px'>noise ratios</div>
    </div>
    <div style='padding:0 20px'>
      <div style='font-size:20px;font-weight:700;color:#9fef00;text-shadow:0 0 12px #9fef00'>● ONLINE</div>
      <div style='font-size:9px;color:#5a6a7e;text-transform:uppercase;letter-spacing:0.12em;margin-top:2px'>status</div>
    </div>
  </div>
</div>
""")
        with gr.Tabs():
            with gr.Tab("🔬 Research Query"):
                with gr.Row():
                    with gr.Column(scale=3):
                        exp_dd   = gr.Dropdown(choices=EXPERIMENT_TYPES,label="Architecture",value="GraphRAG")
                        ratio_dd = gr.Dropdown(choices=[0.0,0.5,1.0,2.0,3.0],label="Noise Ratio",value=0.0)
                        gr.Markdown("**Try:**")
                        q1_btn = gr.Button("▶ MAE of PBE0 on S66?",        size="sm")
                        q2_btn = gr.Button("▶ Best functional for TMC32?",  size="sm")
                        q3_btn = gr.Button("▶ Why does B3LYP fail for barriers?", size="sm")
                        gr.Markdown("#### KG Topology Trace")
                        t1_graph = gr.HTML(
                            value="<p style='color:#5a6a7e;font-size:10px;font-family:monospace'>// graph trace appears after query</p>"
                        )
                    with gr.Column(scale=7):
                        t1_chat  = gr.Chatbot(label="Research Thread",height=440)
                        t1_input = gr.Textbox(placeholder="Ask any DFT question...",lines=2,label="")
                        with gr.Row():
                            t1_btn   = gr.Button("Query Knowledge Graph",variant="primary",scale=3)
                            t1_clear = gr.Button("Clear",scale=1)
                        t1_compare = gr.HTML(
                            value="<p style='color:#5a6a7e;font-size:10px'>// architecture comparison appears after first query</p>"
                        )
                        t1_meta    = gr.HTML(
                            value="<p style='color:#5a6a7e;font-size:10px'>// meta-reasoner verdict appears after first query</p>"
                        )
                t1_btn.click(fn=research_query_stream,inputs=[t1_input,t1_chat,exp_dd,ratio_dd],outputs=[t1_chat,t1_graph,t1_compare,t1_meta])
                t1_clear.click(fn=lambda:([],"","",""),outputs=[t1_chat,t1_graph,t1_compare,t1_meta])
                q1_btn.click(fn=_quick_q1, inputs=[t1_chat,exp_dd,ratio_dd], outputs=[t1_chat,t1_graph,t1_compare,t1_meta])
                q2_btn.click(fn=_quick_q2, inputs=[t1_chat,exp_dd,ratio_dd], outputs=[t1_chat,t1_graph,t1_compare,t1_meta])
                q3_btn.click(fn=_quick_q3, inputs=[t1_chat,exp_dd,ratio_dd], outputs=[t1_chat,t1_graph,t1_compare,t1_meta])

            with gr.Tab("⚛️ Auto-KGR Lab"):
                gr.Markdown("### Dialectic Multi-Agent Debate\n**🔵 Advisor** proposes from vector store · **🔴 Safety Officer** enforces physics via KG · Loop until accepted")
                with gr.Row():
                    with gr.Column(scale=4):
                        mol_name = gr.Dropdown(
                            choices=[
                                "water dimer","benzene dimer","methane dimer",
                                "ethanol","acetone","caffeine",
                                "glycine","alanine","phenylalanine",
                                "Fe(CO)5","Ni(CO)4","Cr(CO)6",
                                "ferrocene","cisplatin","vitamin B12",
                                "aspirin","ibuprofen","paracetamol",
                                "C60 fullerene","graphene fragment",
                                "DNA base pair (AT)","DNA base pair (GC)",
                                "hydrogen fluoride dimer","ammonia dimer",
                                "formic acid dimer","acetic acid dimer",
                            ],
                            label="Molecule / System",
                            value="water dimer",
                            allow_custom_value=True,
                            info="Select or type your own molecule",
                        )
                        mol_smiles = gr.Textbox(label="SMILES (optional)",placeholder="e.g. O.O",lines=1)
                        mol_xyz    = gr.Textbox(label="XYZ (optional)",lines=5)
                        with gr.Row():
                            charge = gr.Number(label="Charge",value=0,precision=0)
                            mult   = gr.Number(label="Multiplicity",value=1,precision=0)
                        task_dd  = gr.Dropdown(choices=TASK_TYPES,label="Task Type",value="Non-covalent interactions (S66, NCIBLIND10)",info="Safety Officer queries KG failure modes for this task")
                        basis_dd = gr.Dropdown(choices=BASIS_SETS,label="Starting Basis Set",value="def2-TZVP",info="Advisor starts here — Safety Officer may override")
                        debate_btn = gr.Button("▶  Start Dialectic Debate",variant="primary")
                        gr.Markdown(
                            "**🔵 Advisor:** llama-3.3-70b-versatile → vector store\n\n"
                            "**🔴 Safety Officer:** llama-3.3-70b-versatile → KG\n\n"
                            "**🟡 Meta-Reasoner:** qwen/qwen3-32b → reasons about reasoning"
                        )
                    with gr.Column(scale=6):
                        transcript_box = gr.Textbox(label="Live Debate Transcript — includes Meta-Engineering Reasoning Chain",lines=20,interactive=False,placeholder="Debate streams here...\n\nYou will see:\n🔵 Advisor steps\n🔴 Safety Officer steps\n🟡 Qwen3 meta-reasoning steps")
                        summary_box = gr.Markdown("_Start the debate._")

                gr.Markdown("#### KG Evidence — MAE Table")
                mae_html = gr.HTML("<p style='color:#64748b'>Appears after debate.</p>")
                gr.Markdown("#### Generated Input Files")
                fmt_radio = gr.Radio(choices=["PSI4","ORCA","Gaussian 16"],value="PSI4",label="Format")
                code_box  = gr.Code(label="",language="python",lines=22,interactive=True)

                gr.Markdown("#### Follow-up Chat")
                fu_chat  = gr.Chatbot(label="",height=280)
                with gr.Row():
                    fu_input = gr.Textbox(placeholder="Why was X rejected? How do I run this? Expected accuracy?",lines=1,label="",scale=5)
                    fu_btn   = gr.Button("Ask",scale=1)

                psi4_st = gr.State(""); orca_st = gr.State(""); g16_st = gr.State("")
                tran_st = gr.State(""); sum_st  = gr.State("")

                def _debate_and_store(mol,smiles,xyz,task,basis,ch,mu):
                    for t,p,o,g,s,m in run_debate(mol,smiles,xyz,task,basis,ch,mu):
                        yield t,p,s,m,p,o,g,t,s

                def _switch_fmt(fmt,psi4,orca,g16):
                    if fmt=="ORCA": return orca
                    if fmt=="Gaussian 16": return g16
                    return psi4

                debate_btn.click(fn=_debate_and_store,
                    inputs=[mol_name,mol_smiles,mol_xyz,task_dd,basis_dd,charge,mult],
                    outputs=[transcript_box,code_box,summary_box,mae_html,psi4_st,orca_st,g16_st,tran_st,sum_st])
                fmt_radio.change(fn=_switch_fmt,inputs=[fmt_radio,psi4_st,orca_st,g16_st],outputs=[code_box])
                fu_btn.click(fn=followup_chat_fn,inputs=[fu_input,fu_chat,psi4_st,tran_st,sum_st],outputs=[fu_chat])
            # Tab 3: Auto-KGR Ingestion
            if HAS_INGESTION:
                build_ingestion_tab()

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_PORT","7860")))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    if args.standalone:
        _init_standalone()
    build_demo(standalone=args.standalone).queue(max_size=20).launch(
        server_name=args.host, server_port=args.port,
        share=False, show_error=True, css=CSS)
