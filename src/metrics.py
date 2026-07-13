"""
metrics.py — Core metric implementations
==========================================
Implements:
  S(c, r)    – mean pairwise cosine similarity (image geometry, Metric 1)
  Δ_L(c, r)  – cross-lingual alignment gap per image (Metric 2)
  G(c, L, m) – prompt gain (Metric 3)
  A(L, m, p) – cultural asymmetry score

All functions operate on pre-loaded, L2-normalized numpy arrays.
"""

import numpy as np
from typing import Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils import mean_pairwise_cosine_similarity, cosine_similarity


# ─── Metric 1: Image geometry ────────────────────────────────────────────────

def compute_S_matrix(
    image_embeds: np.ndarray,
    image_meta: list[dict],
    concepts: list[str],
    regions: list[str]
) -> dict:
    """
    Compute S(c, r) for every (concept, region) pair.
    Returns: s_matrix: {(concept, region): float}
    """
    # Pre-index images by (concept, region)
    img_index = {}
    for i, m in enumerate(image_meta):
        k = (m["object"], m["region"])
        img_index.setdefault(k, []).append(i)

    s_matrix = {}
    for concept in concepts:
        for region in regions:
            indices = img_index.get((concept, region), [])
            if len(indices) < 2:
                s_matrix[(concept, region)] = np.nan
            else:
                subset = image_embeds[indices]
                s_matrix[(concept, region)] = mean_pairwise_cosine_similarity(subset)

    return s_matrix


def compute_cross_region_divergence(
    image_embeds: np.ndarray,
    image_meta: list[dict],
    concepts: list[str],
    regions: list[str]
) -> dict:
    """
    Cross-region divergence for concept c:
    1 - mean cross-region cosine similarity.
    Higher = more divergence across regions.
    Returns: {concept: float}
    """
    # Pre-index images by (concept, region)
    img_index = {}
    for i, m in enumerate(image_meta):
        k = (m["object"], m["region"])
        img_index.setdefault(k, []).append(i)

    divergence = {}
    for concept in concepts:
        subsets = {}
        for region in regions:
            indices = img_index.get((concept, region), [])
            if indices:
                subsets[region] = image_embeds[indices]

        if len(subsets) < 2:
            divergence[concept] = np.nan
            continue

        cross_sims = []
        region_list = list(subsets.keys())
        for i in range(len(region_list)):
            for j in range(i + 1, len(region_list)):
                cross_sim = subsets[region_list[i]] @ subsets[region_list[j]].T
                cross_sims.append(cross_sim.mean())

        divergence[concept] = 1.0 - float(np.mean(cross_sims))

    return divergence


# ─── Metric 2: Cross-lingual alignment gap ───────────────────────────────────

def compute_delta_L(
    image_embeds: np.ndarray,
    image_meta: list[dict],
    text_embeds: np.ndarray,
    text_meta: list[dict],
    language: str,
    prompt_id: str = "P1",
    concepts: Optional[list[str]] = None,
    regions: Optional[list[str]] = None
) -> dict:
    """
    Compute Δ_L(c, r) = cos(v, t^EN_c) - cos(v, t^L_c) for each image.

    Returns:
        deltas: {(concept, region): np.ndarray of per-image Δ_L values}
    """
    if language == "en":
        raise ValueError("Δ_L is undefined for the reference language (EN)")

    # Pre-index text embeddings for O(1) lookup: (concept, lang, pid, region) -> embed
    text_index = {}
    for i, m in enumerate(text_meta):
        key = (m["concept"], m["lang"], m["prompt_id"], m["region"])
        text_index[key] = text_embeds[i]

    def get_text_embed(concept, lang, pid, region="global"):
        val = text_index.get((concept, lang, pid, region), None)
        if val is None and region != "global":
            # Fall back to global if no region-specific prompt exists (e.g. for P1)
            val = text_index.get((concept, lang, pid, "global"), None)
        return val

    deltas = {}

    if concepts is None:
        concepts = list(set(m["object"] for m in image_meta))
    if regions is None:
        regions = list(set(m["region"] for m in image_meta))

    # Pre-index image embeddings for fast filtering
    img_index = {}
    for i, m in enumerate(image_meta):
        k = (m["object"], m["region"])
        if k not in img_index:
            img_index[k] = []
        img_index[k].append(i)

    for concept in concepts:
        for region in regions:
            t_en = get_text_embed(concept, "en", prompt_id, region)
            t_L  = get_text_embed(concept, language, prompt_id, region)

            if t_en is None or t_L is None:
                continue

            indices = img_index.get((concept, region), [])
            if not indices:
                continue

            imgs = image_embeds[indices]  # (N, D)

            # cos(v, t_en) for each image
            cos_en = imgs @ t_en   # (N,)
            cos_L  = imgs @ t_L    # (N,)
            delta  = cos_en - cos_L

            deltas[(concept, region)] = delta

    return deltas


def compute_raw_cosines(
    image_embeds: np.ndarray,
    image_meta: list[dict],
    text_embeds: np.ndarray,
    text_meta: list[dict],
    language: str,
    prompt_id: str = "P1",
    concepts: Optional[list[str]] = None,
    regions: Optional[list[str]] = None
) -> dict:
    """
    Compute raw cos(v, t_en) and cos(v, t_L) for each image.
    Returns:
        metrics: {(concept, region): {"cos_en": np.ndarray, "cos_L": np.ndarray}}
    """
    if language == "en":
        raise ValueError("Raw cosines are undefined for language 'en'")

    text_index = {}
    for i, m in enumerate(text_meta):
        key = (m["concept"], m["lang"], m["prompt_id"], m["region"])
        text_index[key] = text_embeds[i]

    def get_text_embed(concept, lang, pid, region="global"):
        val = text_index.get((concept, lang, pid, region), None)
        if val is None and region != "global":
            val = text_index.get((concept, lang, pid, "global"), None)
        return val

    metrics = {}

    if concepts is None:
        concepts = list(set(m["object"] for m in image_meta))
    if regions is None:
        regions = list(set(m["region"] for m in image_meta))

    img_index = {}
    for i, m in enumerate(image_meta):
        k = (m["object"], m["region"])
        if k not in img_index:
            img_index[k] = []
        img_index[k].append(i)

    for concept in concepts:
        for region in regions:
            t_en = get_text_embed(concept, "en", prompt_id, region)
            t_L  = get_text_embed(concept, language, prompt_id, region)

            if t_en is None or t_L is None:
                continue

            indices = img_index.get((concept, region), [])
            if not indices:
                continue

            imgs = image_embeds[indices]
            cos_en = imgs @ t_en
            cos_L  = imgs @ t_L

            metrics[(concept, region)] = {
                "cos_en": cos_en,
                "cos_L": cos_L
            }

    return metrics



# ─── Metric 3: Prompt gain ───────────────────────────────────────────────────

def compute_prompt_gain(
    delta_P1: dict,
    delta_P3: dict,
) -> dict:
    """
    G(c, L, m) = mean(Δ_L under P1) - mean(Δ_L under P3)

    Positive G means P3 (cultural prompt) reduces the alignment gap.

    Args:
        delta_P1: {(concept, region): np.ndarray}  from compute_delta_L with P1
        delta_P3: {(concept, region): np.ndarray}  from compute_delta_L with P3

    Returns:
        gains: {(concept, region): float}
    """
    gains = {}
    for key in delta_P1:
        if key in delta_P3:
            mean_p1 = float(np.mean(delta_P1[key]))
            mean_p3 = float(np.mean(delta_P3[key]))
            gains[key] = mean_p1 - mean_p3
    return gains


# ─── Cultural Asymmetry Score ─────────────────────────────────────────────────

def compute_asymmetry_score(
    deltas: dict,
    tier1_concepts: list[str],
    tier2_concepts: list[str],
    regions: list[str]
) -> dict:
    """
    A(L, m, p) = mean(Δ_L for Tier 2) - mean(Δ_L for Tier 1)

    Returns per-region and global asymmetry values.
    """
    tier1_vals = []
    tier2_vals = []

    for concept in tier1_concepts:
        for region in regions:
            key = (concept, region)
            if key in deltas:
                tier1_vals.extend(deltas[key].tolist())

    for concept in tier2_concepts:
        for region in regions:
            key = (concept, region)
            if key in deltas:
                tier2_vals.extend(deltas[key].tolist())

    return {
        "tier1_mean": float(np.mean(tier1_vals)) if tier1_vals else np.nan,
        "tier2_mean": float(np.mean(tier2_vals)) if tier2_vals else np.nan,
        "asymmetry":  float(np.mean(tier2_vals) - np.mean(tier1_vals))
                      if (tier1_vals and tier2_vals) else np.nan,
        "n_tier1": len(tier1_vals),
        "n_tier2": len(tier2_vals),
    }


# ─── Aggregation helpers ──────────────────────────────────────────────────────

def aggregate_deltas_by_tier(
    deltas: dict,
    tier1_concepts: list[str],
    tier2_concepts: list[str],
    regions: list[str]
):
    """Returns (tier1_image_deltas, tier2_image_deltas) as flat numpy arrays."""
    tier1_vals, tier2_vals = [], []

    for concept in tier1_concepts:
        for region in regions:
            key = (concept, region)
            if key in deltas:
                tier1_vals.extend(deltas[key].tolist())

    for concept in tier2_concepts:
        for region in regions:
            key = (concept, region)
            if key in deltas:
                tier2_vals.extend(deltas[key].tolist())

    return np.array(tier1_vals), np.array(tier2_vals)


def per_region_tier_means(
    deltas: dict,
    tier1_concepts: list[str],
    tier2_concepts: list[str],
    regions: list[str]
):
    """
    Returns (tier1_region_means, tier2_region_means) — one mean per region.
    Used for the region-paired t-test (n=6).
    """
    tier1_means, tier2_means = [], []

    for region in regions:
        t1 = []
        for c in tier1_concepts:
            k = (c, region)
            if k in deltas:
                t1.extend(deltas[k].tolist())

        t2 = []
        for c in tier2_concepts:
            k = (c, region)
            if k in deltas:
                t2.extend(deltas[k].tolist())

        if t1:
            tier1_means.append(float(np.mean(t1)))
        if t2:
            tier2_means.append(float(np.mean(t2)))

    return np.array(tier1_means), np.array(tier2_means)
