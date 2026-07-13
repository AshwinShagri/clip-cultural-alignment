"""
run_all.py — Master pipeline runner
=====================================
Single entry point for all phases of the CLIP Cultural Alignment project.

Usage:
    # Run everything end-to-end (after data is already fetched):
    python run_all.py --phase all

    # Individual phases:
    python run_all.py --phase 0         # Environment check
    python run_all.py --phase data      # Fetch GeoDE subset
    python run_all.py --phase embed     # Compute all embeddings
    python run_all.py --phase taxonomy  # Phase 1: Taxonomy + S(c,r)
    python run_all.py --phase q1        # Q1: Image geometry
    python run_all.py --phase q2        # Q2: Alignment gap
    python run_all.py --phase q3        # Q3: Prompt intervention
    python run_all.py --phase qualitative  # Qualitative + UMAP
    python run_all.py --phase figures   # All publication figures

    # Dry run for testing (limit=10 images per cell):
    python run_all.py --phase all --dry-run

    # Single model only:
    python run_all.py --phase all --model openclip

Logs every run to results/log.csv.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

# ─── Set working dir to project root ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import load_config, get_logger

logger = get_logger("run_all")


def parse_args():
    p = argparse.ArgumentParser(
        description="CLIP Cultural Alignment — Master Pipeline"
    )
    p.add_argument(
        "--phase", required=True,
        choices=["0", "env", "data", "embed", "taxonomy",
                 "q1", "q2", "q3", "analysis", "qualitative",
                 "figures", "all"],
        help="Which phase to run"
    )
    p.add_argument("--model", nargs="+", default=["openclip", "siglip2"],
                   choices=["openclip", "siglip2"])
    p.add_argument("--language", nargs="+", default=["es", "ar"])
    p.add_argument("--dry-run", action="store_true",
                   help="Use --limit 10 for data fetch")
    p.add_argument("--limit", type=int, default=None,
                   help="Images per cell override (for testing)")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if cached")
    p.add_argument("--config", default=None)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def run_step(cmd: list[str], step_name: str) -> bool:
    """Run a subprocess step, return True if successful."""
    logger.info(f"\n{'─'*60}")
    logger.info(f"STEP: {step_name}")
    logger.info(f"CMD:  {' '.join(cmd)}")
    logger.info(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info(f"✓ {step_name} completed in {elapsed:.1f}s")
        return True
    else:
        logger.error(f"✗ {step_name} FAILED (exit code {result.returncode})")
        return False


def phase_env(args, cfg):
    """Phase 0: Environment sanity check."""
    logger.info("=== Phase 0: Environment Check ===")
    import torch
    cuda_ok  = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda_ok else 0

    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  CUDA available: {cuda_ok}")
    logger.info(f"  GPU: {gpu_name}  ({vram_gb:.1f} GB)")
    logger.info(f"  Python: {sys.version.split()[0]}")

    # Quick model load test
    logger.info("  Testing model loading ...")
    from transformers import AutoModel, AutoProcessor
    for model_name in ["openclip"]:   # quick test with just one
        repo_id = cfg["models"][model_name]["repo_id"]
        try:
            proc = AutoProcessor.from_pretrained(repo_id)
            logger.info(f"    ✓ {model_name} processor loaded")
        except Exception as e:
            logger.error(f"    ✗ {model_name} failed: {e}")
            return False

    logger.info("  ✓ Environment OK")
    return True


def phase_data(args, cfg):
    """Phase 1: Fetch GeoDE subset."""
    limit = args.limit or (10 if args.dry_run else None)
    cmd = [sys.executable, "src/fetch_data.py"]
    if limit:
        cmd += ["--limit", str(limit)]
    if args.dry_run:
        cmd += ["--dry-run"]
    if args.config:
        cmd += ["--config", args.config]
    return run_step(cmd, "Data Fetch (GeoDE)")


def phase_embed(args, cfg):
    """Compute image + text embeddings for all specified models."""
    all_ok = True
    for model_name in args.model:
        for modality in ["image", "text"]:
            cmd = [
                sys.executable, "src/embed.py",
                "--model", model_name,
                "--modality", modality,
                "--batch-size", str(args.batch_size)
            ]
            if args.overwrite:
                cmd += ["--overwrite"]
            if args.config:
                cmd += ["--config", args.config]
            ok = run_step(cmd, f"Embeddings: {model_name} {modality}")
            all_ok = all_ok and ok
    return all_ok


def phase_taxonomy(args, cfg):
    """Phase 1: Taxonomy + S(c,r) analysis."""
    cmd = [sys.executable, "src/taxonomy.py",
           "--model"] + args.model
    if args.config:
        cmd += ["--config", args.config]
    return run_step(cmd, "Taxonomy Analysis")


def phase_analysis(question: str, args, cfg):
    """Run statistical analysis for a given question."""
    cmd = [
        sys.executable, "src/analysis.py",
        "--question", question,
        "--model"] + args.model + [
        "--language"] + args.language
    if args.config:
        cmd += ["--config", args.config]
    return run_step(cmd, f"Analysis: {question.upper()}")


def phase_qualitative(args, cfg):
    """Qualitative analysis + UMAP."""
    cmd = [sys.executable, "src/qualitative.py",
           "--model"] + args.model
    if args.config:
        cmd += ["--config", args.config]
    return run_step(cmd, "Qualitative Analysis + UMAP")


def phase_figures(args, cfg):
    """Generate all publication figures."""
    cmd = [sys.executable, "src/visualize.py",
           "--model"] + args.model
    if args.config:
        cmd += ["--config", args.config]
    return run_step(cmd, "Figure Generation")


def print_summary(cfg):
    """Print final summary of all results."""
    import pandas as pd
    log_path = Path(cfg["paths"]["log"])
    if log_path.exists():
        df = pd.read_csv(log_path)
        logger.info(f"\n{'='*60}")
        logger.info(f"RESULTS SUMMARY ({len(df)} log entries)")
        logger.info(f"{'='*60}")

        # Key statistics
        q2_rows = df[df["phase"].str.contains("Q2", na=False) &
                     df["metric_name"].str.contains("welch_p", na=False)]
        if not q2_rows.empty:
            logger.info("\nQ2 Welch p-values:")
            for _, row in q2_rows.iterrows():
                sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 \
                      else "*" if row["p_value"] < 0.05 else "n.s."
                logger.info(f"  {row['model']} | {row['language']} | {row['prompt_template']}: "
                            f"p={row['p_value']:.4f} {sig}")

        logger.info(f"\nFigures saved to: {cfg['paths']['figures']}")
        figs = sorted(Path(cfg["paths"]["figures"]).glob("*.png"))
        for f in figs:
            logger.info(f"  {f.name}")
    else:
        logger.info("No results log found.")


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    logger.info(f"\n{'='*60}")
    logger.info(f"CLIP Cultural Alignment Pipeline")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"Models: {args.model}")
    logger.info(f"Languages: {args.language}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*60}")

    phase = args.phase
    all_ok = True

    if phase in ("0", "env", "all"):
        ok = phase_env(args, cfg)
        all_ok = all_ok and ok
        if not ok and phase == "all":
            logger.error("Environment check failed — aborting")
            sys.exit(1)

    if phase in ("data", "all"):
        ok = phase_data(args, cfg)
        all_ok = all_ok and ok

    if phase in ("embed", "all"):
        ok = phase_embed(args, cfg)
        all_ok = all_ok and ok

    if phase in ("taxonomy", "all"):
        ok = phase_taxonomy(args, cfg)
        all_ok = all_ok and ok

    if phase in ("q1", "analysis", "all"):
        ok = phase_analysis("q1", args, cfg)
        all_ok = all_ok and ok

    if phase in ("q2", "analysis", "all"):
        ok = phase_analysis("q2", args, cfg)
        all_ok = all_ok and ok

    if phase in ("q3", "analysis", "all"):
        ok = phase_analysis("q3", args, cfg)
        all_ok = all_ok and ok

    if phase in ("qualitative", "all"):
        ok = phase_qualitative(args, cfg)
        all_ok = all_ok and ok

    if phase in ("figures", "all"):
        ok = phase_figures(args, cfg)
        all_ok = all_ok and ok

    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline {'COMPLETE ✓' if all_ok else 'FINISHED WITH ERRORS ✗'}")
    logger.info(f"{'='*60}")

    print_summary(cfg)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
