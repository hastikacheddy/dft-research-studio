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

setup_logging(run_tag="app", log_dir="logs")
logger = logging.getLogger(__name__)

LITSERVE_URL  = os.getenv("LITSERVE_URL","http://localhost:8000/predict")
SERVE_API_KEY = os.getenv("SERVE_API_KEY","")
EXPERIMENT_TYPES = ["GraphRAG","Graph Deterministic","Standard RAG","Multi-Agent System (Standard RAG Fallback)"]
TASK_TYPES = [
    "Non-covalent interactions (S66, NCIBLIND10)",
    "Main-group thermochemistry (W4-11, G3/05)",
    "Transition metal complexes (TMC32, MOR41)",
    "Reaction barrier heights (BH76, HTBH38)",
    "Dispersion-dominated systems (S22, S66x8)",
    "Excited states / charge transfer (TDDFT)",
]
BASIS_SETS = ["def2-TZVP","def2-QZVP","6-311+G(2d,2p)","aug-cc-pVTZ","cc-pVTZ","def2-SVP","6-31G*"]

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
    _runner = ExperimentOrchestrator(dm, cfg)
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

def research_query_stream(user_query, history, exp_type, ratio):
    if not user_query.strip():
        yield history, ""
        return
    history = list(history)
    history.append({"role":"user","content":user_query})
    history.append({"role":"assistant","content":"Querying knowledge graph..."})
    yield history, ""
    answer, metrics_raw, trace_data = _call_backend(user_query, exp_type, ratio)
    graph_html = ""
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
                net = Network(height="400px", width="100%", directed=True,
                              bgcolor="#0d1117", font_color="#e6edf3")
                net.set_options('{"physics":{"forceAtlas2Based":{"gravitationalConstant":-80,"springLength":120},"solver":"forceAtlas2Based","minVelocity":0.75},"edges":{"arrows":"to","color":{"color":"#30363d","highlight":"#00f3ff"}}}')
                for node in sub.nodes():
                    attrs = G.nodes[node]
                    color = "#ff7b72" if node in found_ids else "#388bfd"
                    size  = 18 if node in found_ids else 8
                    title = f"ID: {node}<br>Type: {attrs.get('label','Node')}"
                    if attrs.get("value") and str(attrs.get("value")) != "nan":
                        title += f"<br>Value: {attrs['value']} {attrs.get('unit','')}"
                    net.add_node(node, label=str(node)[:18], color=color, size=size, title=title)
                for u, v, data in sub.edges(data=True):
                    net.add_edge(u, v, title=data.get("relationship",""), color="#30363d")
                tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
                net.save_graph(tmp.name)
                html_content = open(tmp.name).read()
                os.unlink(tmp.name)
                graph_html = (
                    f'<iframe srcdoc="{html_content.replace(chr(34),chr(39))}" '
                    f'style="width:100%;height:420px;border:none;border-radius:8px;" '
                    f'sandbox="allow-scripts allow-same-origin"></iframe>'
                )
            except Exception as exc:
                logger.warning("Graph render failed: %s", exc)
    rouge_disp = ""
    for pair in _qa_pairs:
        if pair["question"].strip().lower() in user_query.lower():
            s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True).score(
                pair["ground_truth"], answer)["rougeL"].fmeasure
            rouge_disp = f" | ROUGE-L: {s:.3f}"
            break
    partial = ""
    for word in answer.split(" "):
        partial += word + " "
        history[-1] = {"role":"assistant","content":partial}
        yield history, graph_html
        time.sleep(0.01)
    history[-1] = {"role":"assistant","content":
        partial + f"\n\n⚙ {exp_type} | noise={ratio} | {metrics_raw}{rouge_disp}"}
    yield history, graph_html

def run_debate(molecule, smiles, xyz, task, basis, charge, mult):
    if not molecule.strip():
        yield "Enter a molecule name.", "", "", "", "_No molecule._", ""
        return
    if _debate_orchestrator is None:
        yield "Run with --standalone.", "", "", "", "", ""
        return
    for t, p, o, g, s, m in _debate_orchestrator.run_streaming(
        task=task.split(" (")[0],
        molecule=molecule, xyz=xyz, smiles=smiles,
        charge=int(charge), multiplicity=int(mult),
    ):
        yield t, p, o, g, s, m

def followup_chat_fn(user_message, history, psi4_ctx, transcript_ctx, summary_ctx):
    if not user_message.strip():
        yield history
        return
    history = list(history)
    history.append({"role":"user","content":user_message})
    history.append({"role":"assistant","content":"..."})
    yield history
    if _runner is None:
        history[-1] = {"role":"assistant","content":"Run with --standalone."}
        yield history
        return
    llm = _runner.engines.default_llm
    prompt = f"""DFT Research Studio assistant. Debate outcome: {summary_ctx[:800]}
Input file: {psi4_ctx[:600]}. Answer the user's question accurately.
USER: {user_message}"""
    resp, *_ = llm.generate(prompt, max_tokens=400)
    partial = ""
    for word in resp.split(" "):
        partial += word + " "
        history[-1] = {"role":"assistant","content":partial}
        yield history
        time.sleep(0.01)

def build_demo(standalone):
    mode = "Standalone" if standalone else "LitServe"
    with gr.Blocks(title="DFT Research Studio") as demo:
        gr.Markdown(f"# 🧪 DFT Research Studio  —  {mode}")

        with gr.Tabs():

            # ── Tab 1 ─────────────────────────────────────────────────
            with gr.Tab("🔬 Research Query"):
                with gr.Row():
                    with gr.Column(scale=3):
                        exp_dd   = gr.Dropdown(choices=EXPERIMENT_TYPES, label="Architecture", value="GraphRAG")
                        ratio_dd = gr.Dropdown(choices=[0.0,0.5,1.0,2.0,3.0], label="Noise Ratio", value=0.0)
                        gr.Markdown("**Try:**\n- MAE of PBE0 on S66?\n- Best functional for TMC32?\n- Why does B3LYP fail for barriers?")
                    with gr.Column(scale=7):
                        t1_chat  = gr.Chatbot(label="Research Thread", height=420)
                        t1_input = gr.Textbox(placeholder="Ask any DFT question...", lines=2, label="")
                        with gr.Row():
                            t1_btn   = gr.Button("Query Knowledge Graph", variant="primary", scale=3)
                            t1_clear = gr.Button("Clear", scale=1)
                        t1_graph = gr.HTML()
                t1_btn.click(fn=research_query_stream,
                    inputs=[t1_input, t1_chat, exp_dd, ratio_dd],
                    outputs=[t1_chat, t1_graph])
                t1_clear.click(fn=lambda: ([], ""), outputs=[t1_chat, t1_graph])

            # ── Tab 2 ─────────────────────────────────────────────────
            with gr.Tab("⚛️ Auto-KGR Lab"):
                gr.Markdown(
                    "**🔵 Advisor** proposes from vector store · "
                    "**🔴 Safety Officer** checks KG physics · "
                    "Loop until accepted · MAE table + 3 input formats"
                )

                # Row 1: inputs + transcript
                with gr.Row():
                    with gr.Column(scale=4):
                        mol_name   = gr.Textbox(label="Molecule", value="water dimer")
                        mol_smiles = gr.Textbox(label="SMILES (optional, auto-converts to 3D)", placeholder="e.g. O.O   c1ccccc1   CC(=O)O", lines=1)
                        mol_xyz    = gr.Textbox(label="XYZ geometry (optional)", lines=5,
                            placeholder="O   0.000  0.000  0.117\nH   0.000  0.757 -0.468\nH   0.000 -0.757 -0.468")
                        with gr.Row():
                            charge = gr.Number(label="Charge", value=0, precision=0)
                            mult   = gr.Number(label="Multiplicity", value=1, precision=0)
                        task_dd  = gr.Dropdown(choices=TASK_TYPES, label="Task", value="Non-covalent interactions (S66, NCIBLIND10)")
                        basis_dd = gr.Dropdown(choices=BASIS_SETS, label="Starting Basis", value="def2-TZVP")
                        debate_btn = gr.Button("▶  Start Dialectic Debate", variant="primary")

                    with gr.Column(scale=6):
                        transcript_box = gr.Textbox(label="Live Debate Transcript", lines=18,
                            interactive=False, placeholder="Debate streams here...")
                        summary_box = gr.Markdown("_Start the debate._")

                # Row 2: MAE table
                gr.Markdown("#### KG Evidence — MAE Table")
                mae_html = gr.HTML("<p style='color:#64748b'>Appears after debate.</p>")

                # Row 3: input files — use Radio + single Code block to avoid nested tabs
                gr.Markdown("#### Generated Input Files")
                fmt_radio  = gr.Radio(choices=["PSI4","ORCA","Gaussian 16"], value="PSI4", label="Format")
                code_box   = gr.Code(label="", language="python", lines=22, interactive=True)

                # Row 4: follow-up chat
                gr.Markdown("#### Follow-up Chat")
                fu_chat  = gr.Chatbot(label="", height=280)
                with gr.Row():
                    fu_input = gr.Textbox(placeholder="Why was X rejected? How do I run this? Expected accuracy?",
                                         lines=1, label="", scale=5)
                    fu_btn   = gr.Button("Ask", scale=1)

                # State
                psi4_state = gr.State("")
                orca_state = gr.State("")
                g16_state  = gr.State("")
                tran_state = gr.State("")
                sum_state  = gr.State("")

                def _run_debate(mol, smiles, xyz, task, basis, ch, mu):
                    for t, p, o, g, s, m in run_debate(mol, smiles, xyz, task, basis, ch, mu):
                        yield t, p, s, m, p, o, g, t, s

                def _switch_format(fmt, psi4, orca, g16):
                    if fmt == "ORCA":      return orca
                    if fmt == "Gaussian 16": return g16
                    return psi4

                debate_btn.click(
                    fn=_run_debate,
                    inputs=[mol_name, mol_smiles, mol_xyz, task_dd, basis_dd, charge, mult],
                    outputs=[transcript_box, code_box, summary_box, mae_html,
                             psi4_state, orca_state, g16_state, tran_state, sum_state],
                )
                fmt_radio.change(
                    fn=_switch_format,
                    inputs=[fmt_radio, psi4_state, orca_state, g16_state],
                    outputs=[code_box],
                )
                fu_btn.click(
                    fn=followup_chat_fn,
                    inputs=[fu_input, fu_chat, psi4_state, tran_state, sum_state],
                    outputs=[fu_chat],
                )

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADIO_PORT","7860")))
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    if args.standalone:
        _init_standalone()
    build_demo(standalone=args.standalone).queue(max_size=20).launch(
        server_name=args.host, server_port=args.port,
        share=False, show_error=True,
    )
