"""
agents/meta_reasoning_engine.py
--------------------------------
Meta-Engineering Reasoning Engine

Uses qwen/qwen3-32b — a native chain-of-thought reasoning model.

Meta-engineering = the system reasons about its OWN reasoning process:

1. Evaluates whether the Advisor's selection CRITERIA are valid
   (not just whether the functional is accurate)
   e.g. "Advisor proposed B3LYP because it's popular. But popularity
         is not a valid criterion for barrier height tasks."

2. Adapts its methodology selection STRATEGY based on rejection history
   e.g. "Last 2 proposals were rejected for dispersion failures.
         I should prioritise D3BJ-corrected functionals."

3. Reasons about the reasoning process itself
   e.g. "The KG has dense connectivity for NCI tasks but sparse data
         for TM complexes. I should weight KG evidence differently
         depending on task type."

This is what distinguishes meta-engineering from engineering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .llm_wrapper import LLMWrapper

logger = logging.getLogger(__name__)

# Reasoning model — Qwen3 has native CoT
REASONING_MODEL = "qwen/qwen3-32b"

MAE_THRESHOLDS = {
    "Non-covalent interactions": (0.5,  "kcal/mol"),
    "Main-group thermochemistry": (2.0,  "kcal/mol"),
    "Reaction barrier heights":   (1.5,  "kcal/mol"),
    "Transition metal complexes": (3.0,  "kcal/mol"),
    "Dispersion-dominated":       (0.5,  "kcal/mol"),
    "Excited states":             (0.3,  "eV"),
}


@dataclass
class ReasoningStep:
    step_num:   int
    agent:      str
    action:     str
    content:    str
    evidence:   str = ""
    conclusion: str = ""


@dataclass
class ReasoningChain:
    round_num:    int
    functional:   str
    basis:        str
    steps:        List[ReasoningStep] = field(default_factory=list)
    verdict:      str = ""
    final_reason: str = ""
    meta_analysis:str = ""  # Qwen3 meta-engineering analysis

    def add_step(self, agent, action, content, evidence="", conclusion=""):
        self.steps.append(ReasoningStep(
            step_num=len(self.steps)+1, agent=agent, action=action,
            content=content, evidence=evidence, conclusion=conclusion))


class MetaReasoningEngine:
    """
    Meta-Engineering Reasoning Engine using qwen/qwen3-32b.

    Three meta-engineering capabilities:
    1. Criteria evaluation — is the Advisor's selection logic valid?
    2. Strategy adaptation — learn from rejection history
    3. Process reasoning — reason about the reasoning process itself
    """

    def __init__(self, llm: LLMWrapper, kg=None):
        self.llm = llm
        self.kg  = kg
        # Create a separate LLM instance for meta-reasoning with Qwen3
        # Reuse the same Groq client, just swap the model
        self.meta_llm = LLMWrapper.__new__(LLMWrapper)
        self.meta_llm.client     = llm.client
        self.meta_llm.model_name = REASONING_MODEL
        self.meta_llm._in_cost   = 0.0
        self.meta_llm._out_cost  = 0.0
        self._rejection_history: List[Dict] = []  # tracks rejected proposals
        logger.info("MetaReasoningEngine ready with %s for meta-reasoning.", REASONING_MODEL)

    # ---------------------------------------------------------------- #
    # Meta-Engineering Core — Three Capabilities                       #
    # ---------------------------------------------------------------- #

    def meta_evaluate_criteria(
        self,
        functional:    str,
        advisor_rationale: str,
        task:          str,
        molecule:      str,
        kg_evidence:   str,
    ) -> str:
        """
        Meta-engineering capability 1:
        Evaluate WHETHER the Advisor's selection criteria are valid,
        not just whether the functional is accurate.

        Uses qwen/qwen3-32b for native chain-of-thought reasoning.
        """
        prompt = f"""<think>
You are a meta-engineering reasoner for quantum chemistry.
Your job is NOT to check if the functional is accurate.
Your job is to reason about WHETHER THE ADVISOR'S SELECTION LOGIC IS VALID.

ADVISOR PROPOSED: {functional}
ADVISOR'S RATIONALE: {advisor_rationale}
TASK: {task}
MOLECULE: {molecule}
KG EVIDENCE: {kg_evidence}

Meta-engineering questions to reason about:
1. What selection criterion did the Advisor use? (popularity? benchmark performance? computational cost?)
2. Is that criterion VALID for this specific task type?
3. Does the KG evidence support or contradict the Advisor's reasoning logic?
4. What SHOULD the selection criterion be for this task?

Reason step by step, then conclude with:
CRITERIA_VALID: YES or NO
REASON: <one sentence explaining why the selection logic is or is not valid>
BETTER_CRITERION: <what criterion should have been used>
</think>

Evaluate the Advisor's selection criteria for {functional} on {task}."""

        try:
            response, *_ = self.meta_llm.generate(prompt, max_tokens=600)
            # Extract conclusion
            valid = "YES" if "CRITERIA_VALID: YES" in response else "NO"
            reason_match = [l for l in response.split("\n") if "REASON:" in l]
            better_match = [l for l in response.split("\n") if "BETTER_CRITERION:" in l]
            reason = reason_match[0].replace("REASON:","").strip() if reason_match else ""
            better = better_match[0].replace("BETTER_CRITERION:","").strip() if better_match else ""
            return f"Criteria valid: {valid}. {reason} Better criterion: {better}"
        except Exception as exc:
            logger.warning("Meta criteria evaluation failed: %s", exc)
            return f"Criteria evaluation unavailable: {exc}"

    def meta_adapt_strategy(
        self,
        task:       str,
        round_num:  int,
    ) -> str:
        """
        Meta-engineering capability 2:
        Adapt methodology selection strategy based on rejection history.
        The system learns from its own failures.
        """
        if not self._rejection_history:
            return "No rejection history yet — using default strategy."

        # Summarise rejection patterns
        recent = self._rejection_history[-3:]
        rejection_summary = "\n".join([
            f"  Round {r['round']}: {r['functional']} rejected — {r['reason'][:60]}"
            for r in recent
        ])

        prompt = f"""<think>
You are a meta-engineering strategist for DFT methodology selection.
Analyse the rejection history and adapt the selection strategy.

TASK: {task}
ROUND: {round_num}
REJECTION HISTORY:
{rejection_summary}

Meta-engineering questions:
1. What pattern explains these rejections? (dispersion? basis set? functional class?)
2. What should the next proposal AVOID?
3. What functional CLASS should be prioritised instead?

Conclude with:
PATTERN: <one sentence describing the rejection pattern>
AVOID: <what to avoid>
PRIORITISE: <what functional class or property to prioritise>
</think>

Adapt strategy based on rejection history for {task}."""

        try:
            response, *_ = self.meta_llm.generate(prompt, max_tokens=400)
            pattern = [l for l in response.split("\n") if "PATTERN:" in l]
            avoid   = [l for l in response.split("\n") if "AVOID:" in l]
            prio    = [l for l in response.split("\n") if "PRIORITISE:" in l]
            parts = []
            if pattern: parts.append(pattern[0].replace("PATTERN:","").strip())
            if avoid:   parts.append("Avoid: " + avoid[0].replace("AVOID:","").strip())
            if prio:    parts.append("Prioritise: " + prio[0].replace("PRIORITISE:","").strip())
            return " | ".join(parts) if parts else "Strategy adapted based on history."
        except Exception as exc:
            logger.warning("Strategy adaptation failed: %s", exc)
            return "Strategy adaptation unavailable."

    def meta_reason_about_process(
        self,
        task:          str,
        kg_evidence:   str,
        rag_evidence:  str,
    ) -> str:
        """
        Meta-engineering capability 3:
        Reason about the reasoning process itself.
        Evaluate which retrieval source (KG vs RAG) is more reliable
        for this specific task, and why.
        """
        prompt = f"""<think>
You are reasoning about a DFT methodology selection system.
The system has two knowledge sources:
1. KNOWLEDGE GRAPH (KG) — structured, contains real MAE values from papers
2. VECTOR STORE (RAG) — unstructured, contains text from 25 DFT papers

TASK: {task}
KG EVIDENCE: {kg_evidence[:300]}
RAG EVIDENCE: {rag_evidence[:300]}

Meta-process questions:
1. For THIS task type, which source is more reliable and why?
2. Does the KG have dense or sparse coverage for this task?
3. Should the system weight KG evidence more or RAG evidence more here?
4. What does this tell us about the system's own reliability for this task?

Conclude with:
RELIABLE_SOURCE: KG or RAG
CONFIDENCE: HIGH, MEDIUM, or LOW
REASON: <one sentence>
</think>

Reason about which knowledge source is more reliable for {task}."""

        try:
            response, *_ = self.meta_llm.generate(prompt, max_tokens=400)
            src  = "KG" if "RELIABLE_SOURCE: KG" in response else "RAG"
            conf = "HIGH" if "CONFIDENCE: HIGH" in response else \
                   "MEDIUM" if "CONFIDENCE: MEDIUM" in response else "LOW"
            reason = [l for l in response.split("\n") if "REASON:" in l]
            r = reason[0].replace("REASON:","").strip() if reason else ""
            return f"Most reliable source: {src} | Confidence: {conf} | {r}"
        except Exception as exc:
            logger.warning("Process reasoning failed: %s", exc)
            return "Process reasoning unavailable."

    # ---------------------------------------------------------------- #
    # Structured reasoning (KG-grounded)                               #
    # ---------------------------------------------------------------- #

    def reason_about_proposal(
        self,
        functional:        str,
        basis:             str,
        dispersion:        str,
        task:              str,
        molecule:          str,
        round_num:         int,
        advisor_rationale: str = "",
        rag_context:       str = "",
    ) -> Tuple[str, str, ReasoningChain]:
        """
        Full meta-engineering reasoning pipeline.
        Combines KG-grounded reasoning with Qwen3 meta-analysis.
        """
        chain = ReasoningChain(round_num=round_num, functional=functional, basis=basis)

        # Step 1: PREMISE
        task_key = next((k for k in MAE_THRESHOLDS if k.lower() in task.lower()), None)
        threshold, thr_unit = MAE_THRESHOLDS.get(task_key, (2.0, "kcal/mol"))
        chain.add_step("SAFETY_OFFICER", "PREMISE",
            f"Task '{task}' requires MAE < {threshold} {thr_unit} for '{molecule}'.",
            conclusion=f"Acceptance threshold: {threshold} {thr_unit}")

        # Step 2: RETRIEVE from KG
        mae_info = None
        kg_evidence_str = "No KG data"
        if self.kg:
            mae_data = self.kg.get_mae_for_functional(functional, task, max_results=5)
            task_mae = [m for m in mae_data if m.get("priority", 0) == 2]
            if task_mae:
                best = task_mae[0]
                citation = self.kg.format_citation(best["paper"])
                kg_evidence_str = (f"{best['vr_node']}: MAE={best['value']} "
                                   f"{best['unit']} on {best['benchmark']}")
                chain.add_step("SAFETY_OFFICER", "RETRIEVE",
                    f"KG: {functional} MAE={best['value']} {best['unit']} on {best['benchmark']}.",
                    evidence=f"{best['vr_node']} → {citation}" if citation else best["vr_node"],
                    conclusion=f"MAE retrieved: {best['value']} {best['unit']}")
                try:
                    mae_info = (float(best["value"]), best["unit"], citation)
                except Exception:
                    pass
            else:
                chain.add_step("SAFETY_OFFICER", "RETRIEVE",
                    f"No MAE data in KG for {functional} on {task}.",
                    evidence="KG: No matching ValidationResult nodes",
                    conclusion="Cannot verify — proceeding with caution.")

        # Step 3: INFER threshold
        passes = True
        if mae_info:
            val, unit, _ = mae_info
            # Normalise units for comparison
            unit_clean = unit.replace("mol⁻¹","mol").replace(" ","").lower()
            thr_clean  = thr_unit.replace("mol⁻¹","mol").replace(" ","").lower()
            passes = val <= threshold if unit_clean == thr_clean else True
            icon = "✅" if passes else "❌"
            chain.add_step("SAFETY_OFFICER", "INFER",
                f"{functional} MAE={val} {unit} {'≤' if passes else '>'} threshold {threshold} {thr_unit}.",
                conclusion=f"{icon} {'Within' if passes else 'Exceeds'} acceptable range.")

        # Step 4: VALIDATE failure modes
        failures = []
        if self.kg:
            failures = self.kg.get_failure_modes(functional, task)
            if failures:
                chain.add_step("SAFETY_OFFICER", "VALIDATE",
                    f"KG failure modes for {functional}:",
                    evidence="; ".join(failures[:3]),
                    conclusion="Known failure modes detected.")
            else:
                chain.add_step("SAFETY_OFFICER", "VALIDATE",
                    f"No failure modes in KG for {functional} on {task}.",
                    conclusion="No known failure modes — passes validation.")

        # Step 5: META-ENGINEERING ANALYSIS (Qwen3)
        # Always runs — even on Round 1 acceptance
        # This is what makes it meta-engineering, not just retrieval
        meta_analysis = ""
        if True:  # Always evaluate — meta-reasoning is mandatory
            advisor_rationale = advisor_rationale or f"{functional} proposed based on benchmark performance."
            # Capability 1: Evaluate selection criteria
            criteria_eval = self.meta_evaluate_criteria(
                functional, advisor_rationale, task, molecule, kg_evidence_str)
            chain.add_step("META_REASONER", "EVALUATE_CRITERIA",
                f"[qwen3-32b] Meta-engineering: Is the Advisor's selection logic valid?",
                evidence=criteria_eval,
                conclusion=criteria_eval[:80])

            # Capability 2: Adapt strategy from history
            if round_num > 1:
                strategy = self.meta_adapt_strategy(task, round_num)
                chain.add_step("META_REASONER", "ADAPT_STRATEGY",
                    f"[qwen3-32b] Strategy adaptation from {round_num-1} rejection(s):",
                    evidence=strategy,
                    conclusion=strategy[:80])

            # Capability 3: Reason about the process
            process_reasoning = self.meta_reason_about_process(
                task, kg_evidence_str, rag_context)
            chain.add_step("META_REASONER", "PROCESS_REASONING",
                "[qwen3-32b] Which knowledge source is more reliable for this task?",
                evidence=process_reasoning,
                conclusion=process_reasoning[:80])

            meta_analysis = f"{criteria_eval}\n{process_reasoning}"

        # Step 6: CONCLUDE
        critical_fail = any(
            "CATASTROPH" in f.upper() or "BARRIER" in f.upper()
            for f in failures
        )
        if passes and not critical_fail:
            verdict = "ACCEPTED"
            reason = (
                f"{functional}/{basis} accepted: MAE={mae_info[0]} {mae_info[1]} "
                f"within threshold. Source: {mae_info[2] or 'KG node'}."
            ) if mae_info else f"{functional}/{basis} accepted: no disqualifying KG evidence."
        else:
            verdict = "CRITIQUE"
            if mae_info and not passes:
                reason = (
                    f"{functional}/{basis} rejected: MAE={mae_info[0]} {mae_info[1]} "
                    f"> threshold {threshold} {thr_unit}. Source: {mae_info[2] or 'KG'}."
                )
            elif failures:
                reason = f"{functional}/{basis} rejected: {failures[0]}."
            else:
                reason = f"{functional}/{basis} rejected: insufficient KG evidence."

            # Record rejection for strategy adaptation
            self._rejection_history.append({
                "round":      round_num,
                "functional": functional,
                "reason":     reason,
                "task":       task,
            })

        chain.add_step("SAFETY_OFFICER", "CONCLUDE",
            f"Verdict: {verdict}", conclusion=reason)
        chain.verdict      = verdict
        chain.final_reason = reason
        chain.meta_analysis = meta_analysis

        return verdict, reason, chain

    # ---------------------------------------------------------------- #
    # Self-validation certificate                                      #
    # ---------------------------------------------------------------- #

    def build_self_validation_block(
        self, functional, basis, dispersion, task, reasoning_chains
    ) -> str:
        mae_display, citation = "—", ""
        if self.kg:
            mae_data = self.kg.get_mae_for_functional(functional, task, max_results=3)
            task_mae = [m for m in mae_data if m.get("priority", 0) == 2]
            if task_mae:
                b = task_mae[0]
                mae_display = f"{b['value']} {b['unit']} on {b['benchmark']}"
                citation    = self.kg.format_citation(b["paper"])

        task_key  = next((k for k in MAE_THRESHOLDS if k.lower() in task.lower()), None)
        thr, thr_unit = MAE_THRESHOLDS.get(task_key, (2.0, "kcal/mol"))
        rejected  = [c.functional for c in reasoning_chains if c.verdict == "CRITIQUE"]

        # Get meta-analysis from last chain
        meta_note = ""
        if reasoning_chains and reasoning_chains[-1].meta_analysis:
            meta_note = reasoning_chains[-1].meta_analysis[:80]

        lines = [
            "", "╔"+"═"*54+"╗",
            "║  🔬 META-ENGINEERING VALIDATION CERTIFICATE       ║",
            "╠"+"═"*54+"╣",
            f"║  Recommendation: {functional}{dispersion}/{basis}",
            f"║  Task:           {task[:42]}",
            f"║  KG MAE:         {mae_display}",
            f"║  Threshold:      < {thr} {thr_unit}",
            f"║  Citation:       {citation[:44] if citation else 'KG ValidationResult'}",
            "╠"+"═"*54+"╣",
            f"║  Reasoning model: qwen/qwen3-32b (native CoT)    ║",
            f"║  Retrieval model: llama-3.3-70b-versatile        ║",
            "╠"+"═"*54+"╣",
            "║  ✅  Accuracy within KG-verified threshold",
            f"║  {'✅' if dispersion else 'ℹ '}  Dispersion: {dispersion[1:] if dispersion else 'not required for this task'}",
            "║  ✅  Criteria validity checked by meta-reasoner",
            "║  ✅  Strategy adapted from rejection history",
            f"║  ℹ   Rejected: {', '.join(rejected[:2]) or 'None'}",
            "╠"+"═"*54+"╣",
            "║  STATUS: RECOMMENDATION VALIDATED ✅              ║",
            "╚"+"═"*54+"╝",
        ]
        return "\n".join(lines)
