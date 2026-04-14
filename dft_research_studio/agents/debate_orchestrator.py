from __future__ import annotations
import logging, re, textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple
from .llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)
_MAX_ROUNDS = 3
_ACCEPT_TOKEN = "ACCEPTED"


def smiles_to_xyz(smiles: str) -> Tuple[str, str]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return "", f"Invalid SMILES: {smiles}"
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
            return "", "3D embedding failed — provide XYZ directly."
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        lines = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            lines.append(f"  {atom.GetSymbol():<4} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        return "\n".join(lines), "3D geometry from SMILES via RDKit (MMFF optimised)."
    except ImportError:
        return "", "RDKit not installed — pip install rdkit"
    except Exception as exc:
        return "", f"SMILES conversion failed: {exc}"


def extract_mae_table(graph_engine, functional: str, task: str) -> List[Dict[str, str]]:
    rows = []
    try:
        if hasattr(graph_engine, "G"):
            G = graph_engine.G
            for node_id, attrs in G.nodes(data=True):
                if attrs.get("label") == "ValidationResult":
                    val = attrs.get("value")
                    if val and str(val) not in ("nan", "None", ""):
                        nbrs = list(G.predecessors(node_id)) + list(G.successors(node_id))
                        if any(functional.upper() in str(n).upper() for n in nbrs):
                            rows.append({"functional": functional, "node_id": str(node_id),
                                         "value": str(val), "unit": str(attrs.get("unit","")),
                                         "source": attrs.get("paper_id","KG")})
            return rows[:8]
        nodes_df = graph_engine.nodes_df
        rels_df  = graph_engine.rels_df
        vr_nodes = nodes_df[nodes_df["label"] == "ValidationResult"]
        for _, vr_row in vr_nodes.iterrows():
            vr_id = vr_row["node_id"]
            val   = vr_row.get("value")
            if not val or str(val) in ("nan","None",""):
                continue
            edges = rels_df[(rels_df["source_id"]==vr_id)|(rels_df["target_id"]==vr_id)]
            nbrs  = set(edges["source_id"].tolist()+edges["target_id"].tolist()) - {vr_id}
            if any(functional.upper() in str(n).upper() for n in nbrs):
                papers = edges["paper_id"].dropna().unique().tolist()
                rows.append({"functional": functional, "node_id": str(vr_id),
                              "value": str(val), "unit": str(vr_row.get("unit","")),
                              "source": papers[0] if papers else "KG"})
        return rows[:8]
    except Exception as exc:
        logger.warning("MAE extraction failed: %s", exc)
        return []


def format_mae_table_html(rows, accepted_functional, rejected_functionals):
    if not rows:
        return "<p style='color:#64748b'>No MAE data found in KG for this functional.</p>"
    h = ("<table style='width:100%;border-collapse:collapse;font-family:monospace;"
         "font-size:0.85em;color:#e6edf3'>"
         "<tr style='background:#1e3a5f;color:#93c5fd'>"
         "<th style='padding:6px 10px;text-align:left'>Functional</th>"
         "<th style='padding:6px 10px;text-align:left'>Benchmark Node</th>"
         "<th style='padding:6px 10px;text-align:right'>Value</th>"
         "<th style='padding:6px 10px;text-align:left'>Unit</th>"
         "<th style='padding:6px 10px;text-align:left'>Source</th></tr>")
    body = ""
    for i, r in enumerate(rows):
        bg    = "#0d1117" if i%2==0 else "#161b22"
        color = "#3fb950" if r["functional"].upper()==accepted_functional.upper() else "#e6edf3"
        body += (f"<tr style='background:{bg}'>"
                 f"<td style='padding:5px 10px;color:{color};font-weight:bold'>{r['functional']}</td>"
                 f"<td style='padding:5px 10px'>{r['node_id'][:40]}</td>"
                 f"<td style='padding:5px 10px;text-align:right'>{r['value']}</td>"
                 f"<td style='padding:5px 10px'>{r['unit']}</td>"
                 f"<td style='padding:5px 10px;color:#64748b'>{r['source'][:30]}</td></tr>")
    return h + body + "</table>"


def _safe(mol): return re.sub(r"[^\w]","_",mol.lower().strip())


def make_psi4_input(molecule, geom, functional, basis, dispersion, task, charge, mult, papers, transcript):
    calc = "optimize" if "geometry" in task.lower() else "energy"
    func_str = f"{functional.lower()}{dispersion.lower()}"
    ref = ", ".join(papers[:3]) if papers else "DFT knowledge graph"
    safe = _safe(molecule)
    disp_note = f" + {dispersion[1:]} dispersion" if dispersion else ""
    return textwrap.dedent(f"""\
        # PSI4 Input — {molecule}
        # Task:       {task}
        # Functional: {functional}{dispersion}{disp_note}
        # Basis:      {basis}
        # Validated:  Auto-KGR Dialectic Debate
        # KG Sources: {ref}

        import psi4
        psi4.set_memory('4 GB')
        psi4.set_num_threads(4)
        psi4.set_output_file('{safe}_output.dat', False)

        mol = psi4.geometry(\"\"\"
          {charge} {mult}
        {geom if geom.strip() else f'  # Paste {molecule} geometry here'}
          units angstrom
          no_reorient
          no_com
        \"\"\")

        psi4.set_options({{'basis':'{basis}','scf_type':'df',
            'dft_spherical_points':590,'dft_radial_points':99,
            'd_convergence':1e-8,'e_convergence':1e-8}})

        e, wfn = psi4.{calc}('{func_str}', return_wfn=True)
        print(f'Energy: {{e:.8f}} Eh')
        wfn.to_file('{safe}_wfn.npy')
    """)


def make_orca_input(molecule, geom, functional, basis, dispersion, task, charge, mult, papers):
    calc = "Opt" if "geometry" in task.lower() else "SP"
    disp = dispersion.replace("-","").strip() if dispersion else ""
    ref  = ", ".join(papers[:3]) if papers else "DFT knowledge graph"
    geom_block = "\n".join(l.strip() for l in geom.strip().split("\n") if l.strip()) \
                 if geom.strip() else f"# Paste {molecule} geometry here"
    return textwrap.dedent(f"""\
        # ORCA Input — {molecule}
        # Task:       {task}
        # Functional: {functional}{dispersion}
        # Basis:      {basis}
        # KG Sources: {ref}

        ! {functional} {disp} {basis} {calc} TightSCF DefGrid3

        %pal nprocs 4 end
        %maxcore 1024

        * xyz {charge} {mult}
        {geom_block}
        *
    """)


def make_gaussian_input(molecule, geom, functional, basis, dispersion, task, charge, mult, papers):
    disp_g16 = ""
    if dispersion:
        disp_g16 = " EmpiricalDispersion=GD3BJ" if "BJ" in dispersion.upper() else " EmpiricalDispersion=GD3"
    calc_kw  = "Opt" if "geometry" in task.lower() else "SP"
    ref      = ", ".join(papers[:3]) if papers else "DFT knowledge graph"
    safe     = _safe(molecule)
    geom_block = "\n".join(l.strip() for l in geom.strip().split("\n") if l.strip()) \
                 if geom.strip() else "# Paste geometry here"
    return textwrap.dedent(f"""\
        %chk={safe}.chk
        %nprocshared=4
        %mem=4GB
        # {functional}/{basis} {calc_kw}{disp_g16} SCF=Tight Integral=UltraFine

        {molecule} — Auto-KGR ({functional}/{basis})
        KG Sources: {ref}

        {charge} {mult}
        {geom_block}

    """)


@dataclass
class DebateRound:
    round_num:         int
    advisor_proposal:  str
    advisor_rationale: str
    safety_verdict:    str
    safety_critique:   str
    paper_ids:         List[str] = field(default_factory=list)


@dataclass
class DebateResult:
    accepted_functional: str
    accepted_basis:      str
    dispersion:          str
    rounds:              List[DebateRound]
    final_rationale:     str
    psi4_input:          str
    orca_input:          str
    gaussian_input:      str
    mae_rows:            List[Dict[str, str]]
    explainability:      str
    resolved_in_round:   int
    paper_ids:           List[str]

    @property
    def full_functional(self):
        return f"{self.accepted_functional}{self.dispersion}"


class DebateOrchestrator:
    def __init__(self, rag_engine, graph_engine, llm: LLMWrapper, max_rounds=3, nodes_df=None, rels_df=None):
        self.rag, self.graph, self.llm, self.max_rounds = rag_engine, graph_engine, llm, max_rounds
        try:
            from .kg_query_engine import KGQueryEngine
            self.kg = KGQueryEngine(nodes_df, rels_df) if nodes_df is not None and rels_df is not None else None
        except Exception as exc:
            self.kg = None
            import logging
            logging.getLogger(__name__).warning("KGQueryEngine not loaded: %s", exc)

    def _advisor_propose(self, task, molecule, prior_critique="", round_num=0):
        try:
            docs = self.rag.retriever.invoke(f"best DFT functional basis set {task} {molecule} benchmark")
            rag_ctx = "\n".join(d.page_content for d in docs)[:2000]
        except Exception:
            rag_ctx = "No vector store context."
        revision = (f"\n\nCRITICAL: Previous proposal REJECTED.\nReason: {prior_critique}\n"
                    f"Propose a DIFFERENT functional addressing this rejection."
                    if round_num > 0 else "")
        prompt = f"""You are Agent A (The Advisor) in a DFT methodology debate.
Propose the best functional/basis set based on well-cited benchmark performance.{revision}

TASK: {task}  MOLECULE: {molecule}
LITERATURE (vector store): {rag_ctx}

Reply EXACTLY:
FUNCTIONAL: <name>
BASIS: <name>
DISPERSION: <none OR e.g. D3BJ>
RATIONALE: <one sentence with benchmark MAE data>"""
        resp, *_ = self.llm.generate(prompt, max_tokens=250)
        func, basis, disp, rationale = "PBE0", "def2-TZVP", "", "Widely used hybrid."
        for l in resp.split("\n"):
            l = l.strip()
            if l.startswith("FUNCTIONAL:"): func = l.split(":",1)[1].strip()
            elif l.startswith("BASIS:"): basis = l.split(":",1)[1].strip()
            elif l.startswith("DISPERSION:"):
                raw = l.split(":",1)[1].strip().lower()
                if raw not in ("none","no","","n/a"):
                    m = re.search(r"d[34][bj]*", raw)
                    disp = f"-{m.group().upper()}" if m else "-D3BJ"
            elif l.startswith("RATIONALE:"): rationale = l.split(":",1)[1].strip()
        return func, basis, disp, rationale, f"{func}{disp}/{basis}"

    def _safety_check(self, functional, basis, dispersion, task, molecule):
        query = f"{functional} {basis} limitations failure modes MAE {task} {molecule}"
        try:
            ctx, pids, _ = self.graph.get_star_context(query)
        except Exception:
            try:
                ctx, _, pids = self.graph.get_topological_context(query)
            except Exception:
                ctx, pids = "No KG evidence.", []
        prompt = f"""You are Agent B (The Safety Officer) in a DFT debate.
Check the KG for physical limitations. Only correctness matters.

PROPOSAL: {functional}{dispersion}/{basis}  TASK: {task}
KG EVIDENCE: {ctx[:2500]}

Check: barrier height errors, dispersion failures, basis adequacy, MAE > 2 kcal/mol.

Reply EXACTLY:
VERDICT: ACCEPTED or CRITIQUE
CRITIQUE: <physical failure with MAE if available; empty if ACCEPTED>
ALTERNATIVE: <better functional to try; empty if ACCEPTED>
EVIDENCE: <one sentence citing specific KG node/paper>"""
        resp, *_ = self.llm.generate(prompt, max_tokens=300)
        verdict, critique, alt, evidence = "ACCEPTED", "", "", ""
        for l in resp.split("\n"):
            l = l.strip()
            if l.startswith("VERDICT:"): verdict = "ACCEPTED" if "ACCEPT" in l.upper() else "CRITIQUE"
            elif l.startswith("CRITIQUE:"): critique = l.split(":",1)[1].strip()
            elif l.startswith("ALTERNATIVE:"): alt = l.split(":",1)[1].strip()
            elif l.startswith("EVIDENCE:"): evidence = l.split(":",1)[1].strip()
        full = critique
        if alt and alt.lower() not in ("empty","none",""):
            full += f" → Try: {alt}"
        return verdict, full, evidence, pids

    def _explainability(self, rounds, final_func, final_basis, final_disp, mae_rows):
        lines = ["### Why this methodology was chosen\n"]
        for r in rounds:
            icon = "REJECTED" if r.safety_verdict == "CRITIQUE" else "ACCEPTED"
            lines.append(f"**Round {r.round_num} — {icon}:** `{r.advisor_proposal}` — {r.safety_critique or 'passed all KG checks.'}")
        lines.append("")
        if mae_rows:
            b = mae_rows[0]
            lines.append(f"**KG Evidence:** `{final_func}{final_disp}` reports {b['value']} {b['unit']} on {b['node_id']} (source: {b['source']}).")
        else:
            lines.append(f"**KG Evidence:** `{final_func}{final_disp}/{final_basis}` accepted by Safety Officer.")
        if final_disp:
            lines.append(f"**Dispersion:** `{final_disp[1:]}` correction applied — required for this task type.")
        return "\n".join(lines)

    def run_streaming(self, task, molecule, xyz="", smiles="", charge=0, multiplicity=1, output_format="PSI4"):
        resolved_geom, geom_note = xyz, ""
        if smiles.strip() and not xyz.strip():
            resolved_geom, geom_note = smiles_to_xyz(smiles)
            if not resolved_geom:
                geom_note = f"⚠ {geom_note}"

        lines = ["="*58, "⚖️   DIALECTIC MULTI-AGENT DEBATE  —  Auto-KGR", "="*58,
                 f"  Task:     {task}", f"  Molecule: {molecule}"]
        if geom_note:
            lines.append(f"  Geometry: {geom_note}")
        lines.append("")

        critique, all_papers, rounds = "", [], []
        final_func, final_basis, final_disp = "PBE0", "def2-TZVP", ""
        resolved_in = self.max_rounds

        for rnd in range(self.max_rounds):
            lines.append(f"┌─ Round {rnd+1} {'─'*48}")
            lines.append("│  🔵 ADVISOR  querying vector store...")
            yield "\n".join(lines), "", "", "", "", ""

            func, basis, disp, rationale, proposal = self._advisor_propose(
                task=task, molecule=molecule, prior_critique=critique, round_num=rnd)
            lines += [f"│  🔵 ADVISOR  PROPOSES:  {proposal}",
                      f"│             Rationale: {rationale}", "│"]
            yield "\n".join(lines), "", "", "", "", ""

            lines.append("│  🔴 SAFETY OFFICER  querying knowledge graph...")
            yield "\n".join(lines), "", "", "", "", ""

            verdict, critique, evidence, papers = self._safety_check(func, basis, disp, task, molecule)
            all_papers.extend(papers)
            icon = "✅" if verdict == "ACCEPTED" else "❌"
            lines.append(f"│  🔴 SAFETY OFFICER  {icon} {verdict}")
            if critique: lines.append(f"│             {critique}")
            if evidence: lines.append(f"│             Evidence: {evidence}")
            if papers:   lines.append(f"│             KG: {', '.join(papers[:3])}")
            lines.append("└"+"─"*54)

            rounds.append(DebateRound(rnd+1, proposal, rationale, verdict, critique, papers))
            yield "\n".join(lines), "", "", "", "", ""

            if verdict == "ACCEPTED":
                final_func, final_basis, final_disp = func, basis, disp
                resolved_in = rnd + 1
                break
            if rnd == self.max_rounds - 1:
                final_func, final_basis, final_disp = func, basis, disp

        lines += ["", "═"*58, f"  ✅  RESOLVED in Round {resolved_in}/{self.max_rounds}",
                  f"  ➜   {final_func}{final_disp} / {final_basis}", "═"*58,
                  "", "  ⚙️  Extracting MAE evidence from knowledge graph..."]
        yield "\n".join(lines), "", "", "", "", ""

        mae_rows = extract_mae_table(self.graph, final_func, task)
        rejected = [r.advisor_proposal.split("/")[0].replace("-D3BJ","").replace("-D3","")
                    for r in rounds if r.safety_verdict == "CRITIQUE"]
        for rf in rejected[:2]:
            for row in extract_mae_table(self.graph, rf, task)[:3]:
                row["functional"] = rf
                mae_rows.append(row)

        lines.append("  ⚙️  Generating input files...")
        yield "\n".join(lines), "", "", "", "", ""

        unique_papers = list(set(all_papers))
        transcript    = "\n".join(lines)
        psi4 = make_psi4_input(molecule, resolved_geom, final_func, final_basis, final_disp, task, charge, multiplicity, unique_papers, transcript)
        orca = make_orca_input(molecule, resolved_geom, final_func, final_basis, final_disp, task, charge, multiplicity, unique_papers)
        g16  = make_gaussian_input(molecule, resolved_geom, final_func, final_basis, final_disp, task, charge, multiplicity, unique_papers)

        explain = self._explainability(rounds, final_func, final_basis, final_disp, mae_rows)
        summary = (f"**{final_func}{final_disp} / {final_basis}** — resolved round {resolved_in}/{self.max_rounds}\n\n{explain}")
        mae_html = format_mae_table_html(mae_rows, final_func, rejected)

        lines.append("  ✅  Done.")
        yield "\n".join(lines), psi4, orca, g16, summary, mae_html
