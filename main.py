"""
main.py
-------
Production experiment pipeline for the DFT Research Studio.

Usage
-----
    python main.py                          # full experiment
    python main.py --debug                  # 1 QA pair, all architectures
    python main.py --resume                 # continue from existing checkpoint
    python main.py --ratios 0.0 1.0 3.0    # override distractor ratios
    python main.py --wandb dft-research     # enable W&B experiment tracking
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Load .env before anything else ───────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars set manually also work

from dft_research_studio.config import Config
from dft_research_studio.data import DFTDataManager
from dft_research_studio.evaluation import (
    AdvancedMetrics,
    ReproducibilityTracker,
    ScientificEvaluator,
)
from dft_research_studio.utils import ExperimentOrchestrator
from dft_research_studio.utils.logging_config import setup_logging

# ── Optional tqdm ─────────────────────────────────────────────────────
try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DFT Research Studio — experiment pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Run only the first QA pair (smoke test).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing checkpoint file.",
    )
    p.add_argument(
        "--ratios", nargs="*", type=float,
        help="Override distractor ratios, e.g. --ratios 0.0 1.0 3.0",
    )
    p.add_argument(
        "--models", nargs="*",
        help="Override model list, e.g. --models llama-3.1-8b-instant",
    )
    p.add_argument(
        "--wandb", metavar="PROJECT",
        help="Weights & Biases project name for experiment tracking.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console and file log verbosity.",
    )
    return p.parse_args()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    args = parse_args()

    tag = "debug" if args.debug else "full"
    setup_logging(run_tag=tag, log_dir="logs")
    log = logging.getLogger(__name__)

    # ── Config ───────────────────────────────────────────────────────
    config = Config()
    if args.ratios:
        config.distractor_ratios = args.ratios
        log.info("Distractor ratios overridden: %s", config.distractor_ratios)
    if args.models:
        config.models_to_test = args.models
        log.info("Models overridden: %s", config.models_to_test)

    # ── Reproducibility tracker ───────────────────────────────────────
    tracker = ReproducibilityTracker(
        config,
        input_files=[config.nodes_path, config.rels_path, config.qa_path],
    )

    # ── Data ─────────────────────────────────────────────────────────
    log.info("Loading knowledge graph and QA data …")
    dm = DFTDataManager(config)
    removed = dm.remove_disconnected_nodes()
    log.info("Graph loaded. Removed %d disconnected nodes.", removed)

    # ── Orchestrator ─────────────────────────────────────────────────
    checkpoint_file = f"{tag}_checkpoint.jsonl"
    if not args.resume and Path(checkpoint_file).exists():
        log.warning(
            "Checkpoint '%s' exists but --resume was not passed. "
            "Existing results will be skipped automatically.",
            checkpoint_file,
        )

    log.info("Initialising experiment orchestrator …")
    runner = ExperimentOrchestrator(
        dm,
        config,
        checkpoint_file=checkpoint_file,
        wandb_project=args.wandb,
    )

    # ── Experiment scope ─────────────────────────────────────────────
    qa_subset = [runner.qa_pairs[0]] if args.debug else runner.qa_pairs
    models    = config.models_to_test[:1] if args.debug else config.models_to_test

    total_runs = len(qa_subset) * len(config.distractor_ratios) * 5  # 5 architectures
    log.info(
        "Starting %s experiment: %d QA pairs × %d ratios × 5 architectures = %d runs",
        tag.upper(), len(qa_subset), len(config.distractor_ratios), total_runs,
    )

    # ── Run loop ─────────────────────────────────────────────────────
    try:
        iterator = tqdm(qa_subset, desc="QA pairs", unit="q") if _TQDM else qa_subset

        for qa in iterator:
            log.info("Processing: %s | %s", qa["id"], qa["question"][:60])

            for ratio in config.distractor_ratios:
                log.debug("  ratio=%.1f", ratio)

                runner.run_baseline(models, qa, ratio)
                runner.run_rag(models, qa, ratio)
                runner.run_graph_rag(models, qa, ratio)
                runner.run_graph_deterministic(models, qa, ratio)
                runner.run_multi_agent_system(models, qa, ratio)

        log.info("All experiment runs completed successfully.")

    except KeyboardInterrupt:
        log.warning(
            "Interrupted by user. %d results saved to checkpoint '%s'. "
            "Re-run with --resume to continue.",
            len(runner.results), checkpoint_file,
        )
        runner.finish()
        sys.exit(0)

    except Exception:
        log.critical("Unhandled exception in experiment loop.", exc_info=True)
        runner.finish()
        sys.exit(1)

    # ── Persist full results ──────────────────────────────────────────
    results_file = f"{tag}_results_raw.json"
    runner.save_results(results_file)

    # ── Evaluation ───────────────────────────────────────────────────
    log.info("Running LLM-as-Judge evaluation …")
    metrics_file = f"{tag}_experiment_metrics.csv"
    try:
        evaluator = ScientificEvaluator(results_file=results_file, config=config)
        evaluator.run_all_evaluations(output_csv=metrics_file)
    except Exception:
        log.error("Evaluation failed — raw results are still saved.", exc_info=True)

    # ── Advanced analysis & plots ─────────────────────────────────────
    log.info("Generating advanced metrics and visualisations …")
    try:
        adv = AdvancedMetrics(metrics_path=metrics_file, results_path=results_file)
        adv.generate_report()
    except Exception:
        log.error("Advanced metrics failed — evaluation CSV is still saved.", exc_info=True)

    # ── Reproducibility log ───────────────────────────────────────────
    tracker.save_reproducibility_log(f"{tag}_reproducibility_log.json")
    runner.finish()
    log.info("Pipeline complete. Outputs: %s, %s", results_file, metrics_file)


if __name__ == "__main__":
    main()
