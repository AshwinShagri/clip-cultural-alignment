"""
taxonomy.py — Cultural concept taxonomy validation + sensitivity analysis
=========================================================================
Usage:
    python src/taxonomy.py --model openclip
    python src/taxonomy.py --model openclip --model siglip2

Phase 1 analysis:
  1. Load image embeddings
  2. Compute S(c, r) for all (concept, region) pairs
  3. Combine with published GeoDE accuracy numbers
  4. Assign Tier 1 / Tier 2 labels using bottom-5 / top-5 by accuracy (data-driven)
  5. Run sensitivity analysis across multiple accuracy percentile thresholds
  6. Save results/tables/taxonomy.csv and results/tables/sensitivity.csv
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, log_result
from embed import load_image_embeddings
from metrics import compute_S_matrix, compute_cross_region_divergence

logger = get_logger("taxonomy")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", default=["openclip"],
                   choices=["openclip", "siglip2"],
                   help="Which model(s) to run taxonomy analysis on")
    p.add_argument("--config", default=None)
    return p.parse_args()


def run_taxonomy(cfg: dict, model_name: str) -> pd.DataFrame:
    """
    Core taxonomy analysis for one model.
    Returns a DataFrame with per-(concept, region) S(c,r) values and tier assignments.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Taxonomy analysis: {model_name}")
    logger.info(f"{'='*60}")

    # Load embeddings
    image_embeds, image_meta = load_image_embeddings(cfg, model_name)
    logger.info(f"Loaded {len(image_embeds)} image embeddings, dim={image_embeds.shape[1]}")

    concepts = cfg["all_concepts"]
    regions  = cfg["regions"]
    tier1    = cfg["tier1_concepts"]
    tier2    = cfg["tier2_concepts"]

    # ── Step 1: Compute S(c, r) ──────────────────────────────────────────────
    logger.info("Computing S(c, r) — mean pairwise cosine similarity ...")
    s_matrix = compute_S_matrix(image_embeds, image_meta, concepts, regions)

    # ── Step 2: Cross-region divergence per concept ───────────────────────────
    logger.info("Computing cross-region divergence ...")
    divergence = compute_cross_region_divergence(image_embeds, image_meta, concepts, regions)

    # ── Step 3: Build taxonomy table ──────────────────────────────────────────
    geode_acc = cfg["geode_accuracies"]
    rows = []

    for concept in concepts:
        acc = geode_acc.get(concept, None)
        tier = "Tier2_CulturallyEmbedded" if concept in tier2 else "Tier1_CulturallyUniversal"

        for region in regions:
            s_val = s_matrix.get((concept, region), np.nan)
            rows.append({
                "model":        model_name,
                "concept":      concept,
                "region":       region,
                "tier":         tier,
                "geode_acc":    acc,
                "S_cr":         s_val,
                "cross_region_divergence": divergence.get(concept, np.nan)
            })

    df = pd.DataFrame(rows)

    # ── Step 4: Western vs Non-Western S gap (Q1 preview) ────────────────────
    west_regions    = set(cfg["western_regions"])
    nonwest_regions = set(cfg["nonwestern_regions"])

    summary = []
    for concept in concepts:
        tier = "Tier2" if concept in tier2 else "Tier1"
        acc  = geode_acc.get(concept, None)

        west_s    = [s_matrix[(concept, r)] for r in west_regions
                     if (concept, r) in s_matrix and not np.isnan(s_matrix[(concept, r)])]
        nonwest_s = [s_matrix[(concept, r)] for r in nonwest_regions
                     if (concept, r) in s_matrix and not np.isnan(s_matrix[(concept, r)])]

        mean_west    = float(np.mean(west_s))    if west_s    else np.nan
        mean_nonwest = float(np.mean(nonwest_s)) if nonwest_s else np.nan
        west_gap     = mean_west - mean_nonwest

        summary.append({
            "model":          model_name,
            "concept":        concept,
            "tier":           tier,
            "geode_acc":      acc,
            "S_west_mean":    mean_west,
            "S_nonwest_mean": mean_nonwest,
            "S_west_gap":     west_gap,
            "cross_region_div": divergence.get(concept, np.nan)
        })

    summary_df = pd.DataFrame(summary).sort_values("geode_acc")

    logger.info("\n=== Concept Taxonomy Summary ===")
    print(summary_df.to_string(index=False, float_format="%.4f"))

    # ── Step 5: Sensitivity analysis ─────────────────────────────────────────
    logger.info("\n=== Sensitivity Analysis (threshold robustness) ===")
    all_accs  = sorted(geode_acc.values())
    n_classes = len(geode_acc)
    sens_rows = []

    for pct in cfg["taxonomy"]["sensitivity_percentiles"]:
        cutoff = np.percentile(all_accs, pct)
        tier2_set = {c for c, a in geode_acc.items() if a <= cutoff}
        # Only among our 10 selected concepts
        tier2_our = [c for c in concepts if c in tier2_set]
        tier1_our = [c for c in concepts if c not in tier2_set]

        sens_rows.append({
            "percentile":    pct,
            "accuracy_cutoff": cutoff,
            "tier2_selected":  tier2_our,
            "tier1_selected":  tier1_our,
            "same_as_proposal": (sorted(tier2_our) == sorted(tier2))
        })
        logger.info(f"  pct={pct}%, cutoff={cutoff:.0f}%: Tier2={tier2_our}")

    sens_df = pd.DataFrame(sens_rows)

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    tables_dir = Path(cfg["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_path   = tables_dir / f"taxonomy_{model_name}.csv"
    summary_path    = tables_dir / f"taxonomy_summary_{model_name}.csv"
    sens_path       = tables_dir / "sensitivity.csv"

    df.to_csv(taxonomy_path,  index=False)
    summary_df.to_csv(summary_path, index=False)
    sens_df.to_csv(sens_path, index=False)

    logger.info(f"\nSaved:")
    logger.info(f"  → {taxonomy_path}")
    logger.info(f"  → {summary_path}")
    logger.info(f"  → {sens_path}")

    # Log to results log
    for _, row in summary_df.iterrows():
        log_result(
            cfg["paths"]["log"],
            phase="Phase1_Taxonomy",
            model=model_name,
            concept=row["concept"],
            metric_name="S_west_gap",
            metric_value=row["S_west_gap"],
            notes=f"tier={row['tier']} acc={row['geode_acc']}"
        )

    return df, summary_df, sens_df


def print_stats_summary(summary_df: pd.DataFrame, cfg: dict):
    """Print Tier1 vs Tier2 S(c,r) gap statistics."""
    tier1_gaps = summary_df[summary_df["tier"] == "Tier1"]["S_west_gap"].dropna()
    tier2_gaps = summary_df[summary_df["tier"] == "Tier2"]["S_west_gap"].dropna()

    logger.info(f"\n{'='*50}")
    logger.info("Tier1 S_west_gap — mean±std:")
    logger.info(f"  {tier1_gaps.mean():.4f} ± {tier1_gaps.std():.4f}  (n={len(tier1_gaps)})")
    logger.info("Tier2 S_west_gap — mean±std:")
    logger.info(f"  {tier2_gaps.mean():.4f} ± {tier2_gaps.std():.4f}  (n={len(tier2_gaps)})")

    if len(tier1_gaps) >= 2 and len(tier2_gaps) >= 2:
        t, p = stats.ttest_ind(tier2_gaps, tier1_gaps, equal_var=False)
        logger.info(f"Welch's t-test (Tier2 gap > Tier1 gap): t={t:.3f}, p={p:.4f}")


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    for model_name in args.model:
        emb_path = Path(cfg["paths"]["embeddings"]) / f"{model_name}_image.npz"
        if not emb_path.exists():
            logger.error(f"Image embeddings not found: {emb_path}")
            logger.error("Run: python src/embed.py --model {model_name} --modality image")
            continue

        df, summary_df, sens_df = run_taxonomy(cfg, model_name)
        print_stats_summary(summary_df, cfg)


if __name__ == "__main__":
    main()
