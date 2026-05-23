"""
app_hf.py
---------
Hugging Face Spaces frontend.
Calls the LitServe backend hosted on Lightning.ai.
Set LITSERVE_URL and SERVE_API_KEY in HF Space secrets.
"""
import os, re, time
import gradio as gr
import requests
from rouge_score import rouge_scorer

LITSERVE_URL  = os.getenv("LITSERVE_URL", "")
SERVE_API_KEY = os.getenv("SERVE_API_KEY", "")

EXPERIMENT_TYPES = [
    "GraphRAG",
    "Graph Deterministic",
    "Standard RAG",
    "Multi-Agent System (Standard RAG Fallback)",
]
TASK_TYPES = [
    "Non-covalent interactions (S66, NCIBLIND10)",
    "Main-group thermochemistry (W4-11, G3/05)",
    "Transition metal complexes (TMC32, MOR41)",
    "Reaction barrier heights (BH76, HTBH38)",
    "Dispersion-dominated systems (S22, S66x8)",
    "Excited states / charge transfer (TDDFT)",
]
BASIS_SETS = ["def2-TZVP","def2-QZVP","6-311+G(2d,2p)","aug-cc-pVTZ","cc-pVTZ","def2-SVP"]

def _call_backend(question, exp_type, ratio):
    if not LITSERVE_URL:
        return "⚠️ LITSERVE_URL not configured in Space secrets.", "", ""
    headers = {"X-API-Key": SERVE_API_KEY} if SERVE_API_KEY else {}
    try:
        r = requests.post(LITSERVE_URL,
            json={"question": question, "experiment_type": exp_type, "distractor_ratio": ratio},
            headers=headers, timeout=120)
        r.raise_for_status()
        d = r.json()
        return d["answer"], d["metrics"], d["trace"]
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot reach backend. Is LitServe running on Lightning.ai?", "", ""
    except Exception as exc:
        return f"⚠️ Error: {exc}", "", ""

def _call_debate(molecule, smiles, xyz, task, basis, charge, mult):
    if not LITSERVE_URL:
        yield "⚠️ LITSERVE_URL not configured.", "", "", "", "_No backend._", ""
        return
    headers = {"X-API-Key": SERVE_API_KEY} if SERVE_API_KEY else {}
    try:
        r = requests.post(
            LITSERVE_URL.replace("/predict", "/debate"),
            json={"molecule": molecule, "smiles": smiles, "xyz": xyz,
                  "task": task, "basis": basis, "charge": int(charge),
                  "multiplicity": int(mult)},
            headers=headers, timeout=300, stream=True)
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                import json
                d = json.loads(line)
                yield d.get("transcript",""), d.get("psi4",""), d.get("orca",""), \
                      d.get("gaussian",""), d.get("summary",""), d.get("mae_html","")
    except Exception as exc:
        yield f"Error: {exc}", "", "", "", "", ""

def research_stream(user_query, history, exp_type, ratio):
    if not user_query.strip():
        yield history, ""
        return
    history = list(history)
    history.append({"role":"user","content":user_query})
    history.append({"role":"assistant","content":"Querying knowledge graph..."})
    yield history, ""
    answer, metrics_raw, _ = _call_backend(user_query, exp_type, ratio)
    partial = ""
    for word in answer.split(" "):
        partial += word + " "
        history[-1] = {"role":"assistant","content":partial}
        yield history, ""
        time.sleep(0.01)
    history[-1] = {"role":"assistant","content":
        partial + f"\n\n⚙ {exp_type} | noise={ratio} | {metrics_raw}"}
    yield history, ""

def debate_stream(molecule, smiles, xyz, task, basis, charge, mult):
    for t, p, o, g, s, m in _call_debate(molecule, smiles, xyz, task, basis, charge, mult):
        yield t, p, s, m

def build_demo():
    with gr.Blocks(title="DFT Research Studio") as demo:
        gr.Markdown(
            "# 🧪 DFT Research Studio\n"
            "*Auto-KGR: Knowledge-Graph-Grounded Retrieval for Quantum Chemistry*\n\n"
            "> **Backend:** LitServe on Lightning.ai · "
            "> **KG:** 19,726 nodes · 97,924 edges · 25 DFT papers"
        )
        with gr.Tabs():
            with gr.Tab("🔬 Research Query"):
                gr.Markdown("Query the DFT knowledge graph using four retrieval architectures.")
                with gr.Row():
                    with gr.Column(scale=3):
                        exp_dd   = gr.Dropdown(choices=EXPERIMENT_TYPES, label="Architecture", value="GraphRAG")
                        ratio_dd = gr.Dropdown(choices=[0.0,0.5,1.0,2.0,3.0], label="Distractor Noise Ratio", value=0.0)
                        gr.Markdown("**Try:**\n- MAE of PBE0 on S66?\n- Best functional for TMC32?\n- Why does B3LYP fail for barriers?")
                    with gr.Column(scale=7):
                        t1_chat  = gr.Chatbot(label="Research Thread", height=420)
                        t1_input = gr.Textbox(placeholder="Ask any DFT question...", lines=2, label="")
                        with gr.Row():
                            t1_btn   = gr.Button("Query Knowledge Graph", variant="primary", scale=3)
                            t1_clear = gr.Button("Clear", scale=1)
                        t1_graph = gr.HTML()
                t1_btn.click(fn=research_stream,
                    inputs=[t1_input, t1_chat, exp_dd, ratio_dd],
                    outputs=[t1_chat, t1_graph])
                t1_clear.click(fn=lambda:([],""), outputs=[t1_chat, t1_graph])

            with gr.Tab("⚛️ Auto-KGR Lab"):
                gr.Markdown(
                    "### Dialectic Multi-Agent Debate\n"
                    "**🔵 Advisor** proposes from vector store · "
                    "**🔴 Safety Officer** enforces physics via KG · "
                    "Loop until accepted · MAE table + PSI4/ORCA/Gaussian output"
                )
                with gr.Row():
                    with gr.Column(scale=4):
                        mol_name   = gr.Textbox(label="Molecule", value="water dimer")
                        mol_smiles = gr.Textbox(label="SMILES (optional)", placeholder="e.g. O.O", lines=1)
                        mol_xyz    = gr.Textbox(label="XYZ (optional)", lines=4)
                        with gr.Row():
                            charge = gr.Number(label="Charge", value=0, precision=0)
                            mult   = gr.Number(label="Multiplicity", value=1, precision=0)
                        task_dd  = gr.Dropdown(choices=TASK_TYPES, label="Task", value="Non-covalent interactions (S66, NCIBLIND10)")
                        basis_dd = gr.Dropdown(choices=BASIS_SETS, label="Basis Set", value="def2-TZVP")
                        debate_btn = gr.Button("▶ Start Dialectic Debate", variant="primary")
                    with gr.Column(scale=6):
                        transcript_box = gr.Textbox(label="Live Debate Transcript", lines=16, interactive=False)
                        summary_box    = gr.Markdown("_Start the debate._")

                gr.Markdown("#### KG Evidence — MAE Table")
                mae_html = gr.HTML("<p style='color:#64748b'>Appears after debate.</p>")
                gr.Markdown("#### Generated PSI4 Input")
                psi4_out = gr.Code(label="", language="python", lines=20, interactive=True)

                debate_btn.click(fn=debate_stream,
                    inputs=[mol_name, mol_smiles, mol_xyz, task_dd, basis_dd, charge, mult],
                    outputs=[transcript_box, psi4_out, summary_box, mae_html])

    return demo

build_demo().queue(max_size=10).launch()
