"""
analysis.py — Statistical analysis for Q1, Q2, Q3
===================================================
Usage:
    python src/analysis.py --question q1 --model openclip
    python src/analysis.py --question q2 --model openclip siglip2
    python src/analysis.py --question q3 --model openclip siglip2
    python src/analysis.py --question all --model openclip siglip2

Q1: Image geometry — Does S(c, r) differ between Tier 1 and Tier 2?
     Also: Western vs non-Western S gap comparison.

Q2: Alignment gap — Is Δ_L larger for Tier 2 than Tier 1?
     Runs both region-paired (n=6) and image-level Welch's t-tests.
     Also runs Spearman correlation between S gap and Δ_L (confound check).

Q3: Prompt intervention — Does G (prompt gain) differ by tier, model, language?
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
from embed import load_image_embeddings, load_text_embeddings
from metrics import (
    compute_S_matrix, compute_cross_region_divergence,
    compute_delta_L, compute_prompt_gain, compute_asymmetry_score,
    aggregate_deltas_by_tier, per_region_tier_means, compute_raw_cosines
)

logger = get_logger("analysis")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--question", choices=["q1", "q2", "q3", "all"], required=True)
    p.add_argument("--model", nargs="+", default=["openclip", "siglip2"],
                   choices=["openclip", "siglip2"])
    p.add_argument("--language", nargs="+", default=["es", "ar"])
    p.add_argument("--config", default=None)
    return p.parse_args()


# ─── Q1: Visual geometry ──────────────────────────────────────────────────────

def run_q1(cfg: dict, models: list[str]) -> pd.DataFrame:
    """
    Q1: For culturally embedded object categories, does CLIP's image encoder
    produce higher inter-region feature divergence than for culturally universal categories?
    """
    logger.info("\n" + "="*60)
    logger.info("Q1: Image Geometry Analysis")
    logger.info("="*60)

    tier1    = cfg["tier1_concepts"]
    tier2    = cfg["tier2_concepts"]
    regions  = cfg["regions"]
    west_r   = set(cfg["western_regions"])
    nonwest_r = set(cfg["nonwestern_regions"])
    tables_dir = Path(cfg["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for model_name in models:
        logger.info(f"\n--- Model: {model_name} ---")
        image_embeds, image_meta = load_image_embeddings(cfg, model_name)

        concepts = cfg["all_concepts"]
        s_matrix = compute_S_matrix(image_embeds, image_meta, concepts, regions)
        divergence = compute_cross_region_divergence(image_embeds, image_meta, concepts, regions)

        for concept in concepts:
            tier = "Tier2" if concept in tier2 else "Tier1"
            acc  = cfg["geode_accuracies"].get(concept, None)

            s_by_region = {r: s_matrix.get((concept, r), np.nan) for r in regions}
            west_s    = [s_by_region[r] for r in west_r   if not np.isnan(s_by_region.get(r, np.nan))]
            nonwest_s = [s_by_region[r] for r in nonwest_r if not np.isnan(s_by_region.get(r, np.nan))]

            row = {
                "model":       model_name,
                "concept":     concept,
                "tier":        tier,
                "geode_acc":   acc,
                "div_overall": divergence.get(concept, np.nan),
                "S_west_mean": np.mean(west_s) if west_s else np.nan,
                "S_nonwest_mean": np.mean(nonwest_s) if nonwest_s else np.nan,
                "S_west_gap":  (np.mean(west_s) - np.mean(nonwest_s))
                               if (west_s and nonwest_s) else np.nan,
            }
            for r in regions:
                row[f"S_{r}"] = s_by_region[r]
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    # ── Statistical test: is divergence higher for Tier 2? ───────────────────
    for model_name in models:
        mdf = df[df["model"] == model_name]
        t1_div = mdf[mdf["tier"] == "Tier1"]["div_overall"].dropna()
        t2_div = mdf[mdf["tier"] == "Tier2"]["div_overall"].dropna()

        logger.info(f"\n[{model_name}] Cross-region divergence:")
        logger.info(f"  Tier1 (universal): {t1_div.mean():.4f} ± {t1_div.std():.4f}")
        logger.info(f"  Tier2 (cultural):  {t2_div.mean():.4f} ± {t2_div.std():.4f}")

        if len(t1_div) >= 2 and len(t2_div) >= 2:
            t, p = stats.ttest_ind(t2_div, t1_div, equal_var=False)
            logger.info(f"  Welch t-test: t={t:.3f}, p={p:.4f} {'*' if p < 0.05 else ''}")
            log_result(cfg["paths"]["log"],
                       phase="Q1", model=model_name,
                       metric_name="cross_region_divergence_ttest_p",
                       metric_value=p, p_value=p, test_type="Welch_ind",
                       n_samples=len(t1_div)+len(t2_div))

        # West gap analysis
        t1_gap = mdf[mdf["tier"] == "Tier1"]["S_west_gap"].dropna()
        t2_gap = mdf[mdf["tier"] == "Tier2"]["S_west_gap"].dropna()
        logger.info(f"\n[{model_name}] Western coherence gap (S_west - S_nonwest):")
        logger.info(f"  Tier1: {t1_gap.mean():.4f} (n={len(t1_gap)})")
        logger.info(f"  Tier2: {t2_gap.mean():.4f} (n={len(t2_gap)})")

        for concept_row in mdf.itertuples():
            logger.info(f"  {concept_row.concept:<25} [{concept_row.tier}] "
                        f"acc={concept_row.geode_acc}% "
                        f"div={concept_row.div_overall:.4f} "
                        f"west_gap={concept_row.S_west_gap:.4f}")

    out_path = tables_dir / "q1_image_geometry.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved → {out_path}")
    return df


# ─── Q2: Alignment gap ────────────────────────────────────────────────────────

def run_q2(cfg: dict, models: list[str], languages: list[str]) -> pd.DataFrame:
    """
    Q2: Does the cross-lingual alignment gap Δ_L differ between Tier 1 and Tier 2?
    Runs both:
      (A) Region-paired t-test (n=6 per tier)
      (B) Image-level Welch's t-test (high power)
    Also: Spearman correlation between S gap and Δ_L (confound mitigation).
    """
    logger.info("\n" + "="*60)
    logger.info("Q2: Cross-Lingual Alignment Gap Analysis")
    logger.info("="*60)

    tier1    = cfg["tier1_concepts"]
    tier2    = cfg["tier2_concepts"]
    regions  = cfg["regions"]
    tables_dir = Path(cfg["paths"]["tables"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for model_name in models:
        logger.info(f"\n--- Model: {model_name} ---")
        image_embeds, image_meta = load_image_embeddings(cfg, model_name)
        text_embeds, text_meta   = load_text_embeddings(cfg, model_name)
        concepts = cfg["all_concepts"]

        # Also get S matrix for Spearman correlation
        s_matrix   = compute_S_matrix(image_embeds, image_meta, concepts, regions)
        divergence = compute_cross_region_divergence(image_embeds, image_meta, concepts, regions)

        for lang in languages:
            logger.info(f"\n  Language: {lang}")

            for pid in ["P1", "P2", "P3"]:
                deltas = compute_delta_L(
                    image_embeds, image_meta,
                    text_embeds, text_meta,
                    language=lang, prompt_id=pid,
                    concepts=concepts, regions=regions
                )
                raw_metrics = compute_raw_cosines(
                    image_embeds, image_meta,
                    text_embeds, text_meta,
                    language=lang, prompt_id=pid,
                    concepts=concepts, regions=regions
                )

                if not deltas:
                    logger.warning(f"    No Δ_L values for {lang}/{pid}")
                    continue

                # Aggregate
                asym = compute_asymmetry_score(deltas, tier1, tier2, regions)
                t1_imgs, t2_imgs = aggregate_deltas_by_tier(deltas, tier1, tier2, regions)
                t1_reg, t2_reg   = per_region_tier_means(deltas, tier1, tier2, regions)

                logger.info(f"    [{pid}] A = {asym['asymmetry']:.4f}  "
                            f"(Tier1={asym['tier1_mean']:.4f}, Tier2={asym['tier2_mean']:.4f})")

                # (A) Region-paired test
                if len(t1_reg) >= 2 and len(t2_reg) >= 2 and len(t1_reg) == len(t2_reg):
                    t_pair, p_pair = stats.ttest_rel(t2_reg, t1_reg)
                    logger.info(f"    [{pid}] Region-paired t: t={t_pair:.3f}, p={p_pair:.4f} "
                                f"{'*' if p_pair < 0.05 else ''}")
                    log_result(cfg["paths"]["log"],
                               phase="Q2", model=model_name, language=lang,
                               prompt_template=pid,
                               metric_name="delta_L_region_paired_p",
                               metric_value=p_pair, p_value=p_pair,
                               test_type="paired_ttest",
                               n_samples=len(t1_reg))
                else:
                    t_pair, p_pair = np.nan, np.nan

                # (B) Image-level Welch's test
                if len(t1_imgs) >= 2 and len(t2_imgs) >= 2:
                    t_ind, p_ind = stats.ttest_ind(t2_imgs, t1_imgs, equal_var=False)
                    logger.info(f"    [{pid}] Image-level Welch: t={t_ind:.3f}, p={p_ind:.6f} "
                                f"{'*' if p_ind < 0.05 else ''} (n1={len(t1_imgs)}, n2={len(t2_imgs)})")
                    log_result(cfg["paths"]["log"],
                               phase="Q2", model=model_name, language=lang,
                               prompt_template=pid,
                               metric_name="delta_L_welch_p",
                               metric_value=p_ind, p_value=p_ind,
                               test_type="Welch_ind",
                               n_samples=len(t1_imgs)+len(t2_imgs))
                else:
                    t_ind, p_ind = np.nan, np.nan

                # Per-concept-per-region rows
                for concept in concepts:
                    tier = "Tier2" if concept in tier2 else "Tier1"
                    for region in regions:
                        key = (concept, region)
                        if key in deltas:
                            arr = deltas[key]
                            raw = raw_metrics.get(key, {})
                            mean_cos_en = float(np.mean(raw.get("cos_en", [np.nan])))
                            mean_cos_L  = float(np.mean(raw.get("cos_L", [np.nan])))
                            all_rows.append({
                                "model": model_name, "language": lang,
                                "prompt_id": pid, "concept": concept,
                                "region": region, "tier": tier,
                                "mean_delta_L": float(np.mean(arr)),
                                "std_delta_L":  float(np.std(arr)),
                                "mean_cos_en":  mean_cos_en,
                                "mean_cos_L":   mean_cos_L,
                                "n_images":     len(arr),
                                "S_west_gap": np.mean([s_matrix.get((concept, r), np.nan)
                                                       for r in cfg["western_regions"]]) -
                                              np.mean([s_matrix.get((concept, r), np.nan)
                                                       for r in cfg["nonwestern_regions"]]),
                                "div_overall": divergence.get(concept, np.nan)
                            })

            # Spearman correlation: S gap ↔ Δ_L (P1, confound check)
            logger.info(f"\n  [Spearman rho: S_west_gap vs mean_Δ_L_P1, lang={lang}]")
            mdf = pd.DataFrame([r for r in all_rows
                                 if r["model"] == model_name and r["language"] == lang
                                 and r["prompt_id"] == "P1"])
            if not mdf.empty and "S_west_gap" in mdf.columns:
                concept_agg = mdf.groupby("concept").agg(
                    {"mean_delta_L": "mean", "S_west_gap": "mean", "div_overall": "mean", "mean_cos_en": "mean"}
                ).dropna()
                if len(concept_agg) >= 3:
                    rho_gap, p_rho_gap = stats.spearmanr(
                        concept_agg["S_west_gap"], concept_agg["mean_delta_L"])
                    rho_div, p_rho_div = stats.spearmanr(
                        concept_agg["div_overall"], concept_agg["mean_delta_L"])
                    logger.info(f"    Spearman(S_west_gap, Δ_L_P1): ρ={rho_gap:.3f}, p={p_rho_gap:.4f}")
                    logger.info(f"    Spearman(div_overall,  Δ_L_P1): ρ={rho_div:.3f}, p={p_rho_div:.4f}")
                    log_result(cfg["paths"]["log"],
                               phase="Q2_Confound", model=model_name, language=lang,
                               metric_name="spearman_rho_S_gap_vs_delta",
                               metric_value=rho_gap, p_value=p_rho_gap,
                               test_type="spearmanr", n_samples=len(concept_agg))

            # Run OLS Confound Check (OLS Regression: Δ_L ~ intercept + is_Tier2 + cos_en)
            if not mdf.empty:
                y = mdf["mean_delta_L"].values
                x1 = (mdf["tier"] == "Tier2").astype(float).values
                x2 = mdf["mean_cos_en"].values
                n = len(y)
                if n >= 4:
                    X = np.column_stack([np.ones(n), x1, x2])
                    try:
                        beta = np.linalg.inv(X.T @ X) @ X.T @ y
                        preds = X @ beta
                        residuals = y - preds
                        rss = np.sum(residuals**2)
                        df_residual = n - 3
                        s2 = rss / df_residual
                        cov = s2 * np.linalg.inv(X.T @ X)
                        se = np.sqrt(np.diag(cov))
                        t_stats = beta / se
                        p_vals = 2 * stats.t.sf(np.abs(t_stats), df=df_residual)

                        logger.info(f"    OLS Regression [Δ_L ~ Intercept + is_Tier2 + cos_en] (N={n}):")
                        logger.info(f"      Intercept: beta={beta[0]:.4f}, t={t_stats[0]:.2f}, p={p_vals[0]:.4f}")
                        logger.info(f"      is_Tier2:  beta={beta[1]:.4f}, t={t_stats[1]:.2f}, p={p_vals[1]:.4f}")
                        logger.info(f"      cos_en:    beta={beta[2]:.4f}, t={t_stats[2]:.2f}, p={p_vals[2]:.4f}")

                        log_result(cfg["paths"]["log"],
                                   phase="Q2_Confound_OLS", model=model_name, language=lang,
                                   metric_name="ols_tier2_coefficient",
                                   metric_value=beta[1], p_value=p_vals[1],
                                   test_type="ols_regression", n_samples=n,
                                   notes=f"t={t_stats[1]:.2f}")
                    except Exception as e:
                        logger.warning(f"    OLS Confound Regression failed: {e}")

    df = pd.DataFrame(all_rows)
    out_path = tables_dir / "q2_alignment_gap.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved → {out_path}")
    return df


# ─── Q3: Prompt intervention ──────────────────────────────────────────────────

def run_q3(cfg: dict, models: list[str], languages: list[str]) -> pd.DataFrame:
    """
    Q3: Does culturally-aware prompting (P3) reduce the alignment gap,
    and does the effect interact with concept type and model?
    """
    logger.info("\n" + "="*60)
    logger.info("Q3: Prompt Intervention Analysis")
    logger.info("="*60)

    tier1    = cfg["tier1_concepts"]
    tier2    = cfg["tier2_concepts"]
    regions  = cfg["regions"]
    tables_dir = Path(cfg["paths"]["tables"])

    all_rows = []

    for model_name in models:
        logger.info(f"\n--- Model: {model_name} ---")
        image_embeds, image_meta = load_image_embeddings(cfg, model_name)
        text_embeds, text_meta   = load_text_embeddings(cfg, model_name)
        concepts = cfg["all_concepts"]

        for lang in languages:
            logger.info(f"  Language: {lang}")

            delta_P1 = compute_delta_L(image_embeds, image_meta,
                                       text_embeds, text_meta,
                                       language=lang, prompt_id="P1",
                                       concepts=concepts, regions=regions)
            delta_P3 = compute_delta_L(image_embeds, image_meta,
                                       text_embeds, text_meta,
                                       language=lang, prompt_id="P3",
                                       concepts=concepts, regions=regions)
            delta_P2 = compute_delta_L(image_embeds, image_meta,
                                       text_embeds, text_meta,
                                       language=lang, prompt_id="P2",
                                       concepts=concepts, regions=regions)

            gains_P3 = compute_prompt_gain(delta_P1, delta_P3)
            gains_P2 = compute_prompt_gain(delta_P1, delta_P2)

            for concept in concepts:
                tier = "Tier2" if concept in tier2 else "Tier1"
                for region in regions:
                    key = (concept, region)
                    row = {
                        "model": model_name, "language": lang,
                        "concept": concept, "region": region, "tier": tier,
                        "mean_delta_P1": float(np.mean(delta_P1[key])) if key in delta_P1 else np.nan,
                        "mean_delta_P2": float(np.mean(delta_P2[key])) if key in delta_P2 else np.nan,
                        "mean_delta_P3": float(np.mean(delta_P3[key])) if key in delta_P3 else np.nan,
                        "gain_P3": gains_P3.get(key, np.nan),
                        "gain_P2": gains_P2.get(key, np.nan),
                    }
                    all_rows.append(row)

            # Summary per tier
            df_tmp = pd.DataFrame(all_rows)
            df_tmp = df_tmp[(df_tmp["model"] == model_name) & (df_tmp["language"] == lang)]

            for tier_name, tier_concepts in [("Tier1", tier1), ("Tier2", tier2)]:
                subset = df_tmp[df_tmp["tier"] == tier_name]
                g3_mean = subset["gain_P3"].mean()
                g2_mean = subset["gain_P2"].mean()
                logger.info(f"    [{tier_name}] mean G(P3)={g3_mean:.4f}  mean G(P2)={g2_mean:.4f}")
                log_result(cfg["paths"]["log"],
                           phase="Q3", model=model_name, language=lang,
                           metric_name="mean_prompt_gain_P3",
                           metric_value=g3_mean,
                           notes=f"tier={tier_name}")

            # Test: is G(P3) larger for Tier2?
            t2_gains = df_tmp[df_tmp["tier"] == "Tier2"]["gain_P3"].dropna()
            t1_gains = df_tmp[df_tmp["tier"] == "Tier1"]["gain_P3"].dropna()
            if len(t1_gains) >= 2 and len(t2_gains) >= 2:
                t, p = stats.ttest_ind(t2_gains, t1_gains, equal_var=False)
                logger.info(f"    Welch t (Tier2 G vs Tier1 G): t={t:.3f}, p={p:.4f} "
                            f"{'*' if p < 0.05 else ''}")
                log_result(cfg["paths"]["log"],
                           phase="Q3", model=model_name, language=lang,
                           metric_name="prompt_gain_tier_diff_p",
                           metric_value=p, p_value=p, test_type="Welch_ind",
                           n_samples=len(t1_gains)+len(t2_gains))

    df = pd.DataFrame(all_rows)
    out_path = Path(cfg["paths"]["tables"]) / "q3_prompt_gain.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved → {out_path}")
    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.question in ("q1", "all"):
        run_q1(cfg, args.model)

    if args.question in ("q2", "all"):
        run_q2(cfg, args.model, args.language)

    if args.question in ("q3", "all"):
        run_q3(cfg, args.model, args.language)

    logger.info(f"\nResults logged → {cfg['paths']['log']}")


if __name__ == "__main__":
    main()
