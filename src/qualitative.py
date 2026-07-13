"""
qualitative.py — Qualitative nearest-neighbor analysis + image grids
=====================================================================
Usage:
    python src/qualitative.py --model openclip --top-n 5
    python src/qualitative.py --model openclip siglip2 --top-n 5

For the 5 (concept, region, language) triples with highest mean Δ_L under P1:
  1. Retrieve 5 nearest neighbors in image embedding space for a representative query
  2. Check whether P3 prompting shifts the neighborhood
  3. Export image grids as PNG files to results/figures/

Also produces:
  - UMAP visualization of image embeddings colored by concept and region
  - t-SNE alternative visualization
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger
from embed import load_image_embeddings, load_text_embeddings
from metrics import compute_delta_L

logger = get_logger("qualitative")

# Color palette for regions
REGION_COLORS = {
    "WestAsia":     "#E63946",
    "Africa":       "#F4A261",
    "EastAsia":     "#2A9D8F",
    "SouthEastAsia":"#457B9D",
    "Americas":     "#1D3557",
    "Europe":       "#A8DADC"
}

TIER_COLORS = {
    "Tier1": "#2196F3",
    "Tier2": "#F44336"
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  nargs="+", default=["openclip"],
                   choices=["openclip", "siglip2"])
    p.add_argument("--top-n", type=int, default=5,
                   help="Number of top (concept, region, lang) triples to visualize")
    p.add_argument("--nn-k",  type=int, default=5,
                   help="Number of nearest neighbors to retrieve")
    p.add_argument("--config", default=None)
    return p.parse_args()


# ─── Nearest-neighbor retrieval ───────────────────────────────────────────────

def get_nearest_neighbors(
    query_embed: np.ndarray,
    all_embeds: np.ndarray,
    all_meta: list[dict],
    k: int = 5,
    exclude_concept: str = None,
    exclude_region: str = None
) -> list[tuple[int, float, dict]]:
    """
    Find k nearest neighbors of query_embed in all_embeds (cosine similarity).
    Optionally exclude the same (concept, region) as the query.

    Returns: list of (index, similarity, meta_dict)
    """
    sims = all_embeds @ query_embed  # (N,) cosine similarities

    # Exclude same concept+region (or all exact matches)
    for i, m in enumerate(all_meta):
        if (exclude_concept and m["object"] == exclude_concept and
                exclude_region and m["region"] == exclude_region):
            sims[i] = -999.0  # exclude

    # Sort descending
    sorted_idx = np.argsort(-sims)
    top_k = [(int(idx), float(sims[idx]), all_meta[idx]) for idx in sorted_idx[:k]]
    return top_k


def load_image(raw_dir: Path, meta: dict) -> Image.Image | None:
    try:
        return Image.open(raw_dir / meta["image_path"]).convert("RGB")
    except Exception as e:
        logger.warning(f"Could not load {meta['image_path']}: {e}")
        return None


# ─── Image grid export ───────────────────────────────────────────────────────

def make_nn_grid(
    query_img: Image.Image,
    nn_imgs: list[Image.Image | None],
    nn_metas: list[dict],
    query_meta: dict,
    delta_val: float,
    title: str,
    out_path: str,
    thumbnail_size: int = 200
):
    """Export a query + nearest-neighbors image grid."""
    n_cols = 1 + len(nn_imgs)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.5, 3.5))

    fig.patch.set_facecolor("#1a1a2e")

    def show_img(ax, img, label, color="#ffffff", is_query=False):
        if img is not None:
            thumb = img.resize((thumbnail_size, thumbnail_size), Image.LANCZOS)
            ax.imshow(np.array(thumb))
            if is_query:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#FFD700")
                    spine.set_linewidth(4)
        else:
            ax.set_facecolor("#333")
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    color="white", transform=ax.transAxes)
        ax.set_xlabel(label, fontsize=7, color=color, wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)

    # Query image
    q_label = (f"QUERY\n{query_meta['object']}\n"
                f"{query_meta['region']}\nΔ_L={delta_val:.3f}")
    show_img(axes[0], query_img, q_label, color="#FFD700", is_query=True)

    # Nearest neighbors
    for i, (nn_img, nn_meta) in enumerate(zip(nn_imgs, nn_metas)):
        region_color = REGION_COLORS.get(nn_meta["region"], "#aaa")
        lbl = f"NN{i+1}\n{nn_meta['object']}\n{nn_meta['region']}"
        show_img(axes[i + 1], nn_img, lbl, color=region_color)

    fig.suptitle(title, fontsize=9, color="white", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  Saved grid → {out_path}")


# ─── UMAP / t-SNE embedding visualization ────────────────────────────────────

def plot_embedding_umap(
    image_embeds: np.ndarray,
    image_meta: list[dict],
    cfg: dict,
    model_name: str,
    figures_dir: Path,
    color_by: str = "region"   # "region" | "tier" | "concept"
):
    """Project embeddings with UMAP and plot."""
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("umap-learn not installed; skipping UMAP plot")
        return

    tier1 = set(cfg["tier1_concepts"])
    tier2 = set(cfg["tier2_concepts"])

    logger.info(f"Running UMAP on {len(image_embeds)} embeddings ...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    proj = reducer.fit_transform(image_embeds)  # (N, 2)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    if color_by == "region":
        for region, color in REGION_COLORS.items():
            mask = np.array([m["region"] == region for m in image_meta])
            if mask.sum() > 0:
                ax.scatter(proj[mask, 0], proj[mask, 1],
                           c=color, s=6, alpha=0.6, label=region, linewidths=0)
        ax.legend(loc="best", framealpha=0.3, labelcolor="white",
                  facecolor="#1a1a2e", fontsize=8)

    elif color_by == "tier":
        for tier_name, tier_concepts in [("Tier1 (Universal)", tier1),
                                          ("Tier2 (Cultural)", tier2)]:
            mask = np.array([m["object"] in tier_concepts for m in image_meta])
            color = TIER_COLORS["Tier1" if "Tier1" in tier_name else "Tier2"]
            if mask.sum() > 0:
                ax.scatter(proj[mask, 0], proj[mask, 1],
                           c=color, s=6, alpha=0.6, label=tier_name, linewidths=0)
        ax.legend(loc="best", framealpha=0.3, labelcolor="white",
                  facecolor="#1a1a2e", fontsize=9)

    elif color_by == "concept":
        concepts = cfg["all_concepts"]
        cmap = plt.cm.get_cmap("tab10", len(concepts))
        for i, concept in enumerate(concepts):
            mask = np.array([m["object"] == concept for m in image_meta])
            tier = "▲" if concept in tier2 else "●"
            if mask.sum() > 0:
                ax.scatter(proj[mask, 0], proj[mask, 1],
                           c=[cmap(i)], s=8, alpha=0.6,
                           label=f"{tier} {concept}", linewidths=0)
        ax.legend(loc="best", framealpha=0.3, labelcolor="white",
                  facecolor="#1a1a2e", fontsize=7, ncol=2)

    ax.set_title(f"UMAP: {model_name} image embeddings (color={color_by})",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="gray")
    for spine in ax.spines.values():
        spine.set_color("#333")

    out_path = figures_dir / f"umap_{model_name}_{color_by}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"  UMAP saved → {out_path}")


# ─── Main qualitative pipeline ───────────────────────────────────────────────

def run_qualitative(cfg: dict, model_name: str, top_n: int, nn_k: int):
    """Full qualitative analysis for one model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Qualitative Analysis: {model_name}")
    logger.info(f"{'='*60}")

    raw_dir     = Path(cfg["paths"]["data_raw"])
    figures_dir = Path(cfg["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    image_embeds, image_meta = load_image_embeddings(cfg, model_name)
    text_embeds, text_meta   = load_text_embeddings(cfg, model_name)

    concepts = cfg["all_concepts"]
    regions  = cfg["regions"]
    tier2    = cfg["tier2_concepts"]

    # ── Step 1: UMAP visualizations ──────────────────────────────────────────
    for color_by in ("region", "tier", "concept"):
        plot_embedding_umap(image_embeds, image_meta, cfg, model_name,
                            figures_dir, color_by=color_by)

    # ── Step 2: Find top (concept, region, lang) triples by mean Δ_L (P1) ───
    logger.info("\nFinding top Δ_L triples (P1) ...")
    delta_rows = []

    for lang in [l for l in cfg["languages"] if l != "en"]:
        deltas = compute_delta_L(
            image_embeds, image_meta, text_embeds, text_meta,
            language=lang, prompt_id="P1",
            concepts=concepts, regions=regions
        )
        for (concept, region), arr in deltas.items():
            delta_rows.append({
                "concept": concept, "region": region, "language": lang,
                "mean_delta_P1": float(np.mean(arr)),
                "tier": "Tier2" if concept in tier2 else "Tier1"
            })

    delta_df = pd.DataFrame(delta_rows).sort_values("mean_delta_P1", ascending=False)
    top_triples = delta_df.head(top_n)

    logger.info(f"\nTop {top_n} triples (highest Δ_L under P1):")
    print(top_triples.to_string(index=False))

    top_triples.to_csv(figures_dir / f"top_delta_triples_{model_name}.csv", index=False)

    # ── Step 3: NN retrieval + grid export ───────────────────────────────────
    for _, triple in top_triples.iterrows():
        concept = triple["concept"]
        region  = triple["region"]
        lang    = triple["language"]
        delta_v = triple["mean_delta_P1"]

        # Find a representative query image (first image from this concept/region)
        query_indices = [i for i, m in enumerate(image_meta)
                         if m["object"] == concept and m["region"] == region]
        if not query_indices:
            continue
        q_idx = query_indices[0]
        q_embed = image_embeds[q_idx]
        q_meta  = image_meta[q_idx]
        q_img   = load_image(raw_dir, q_meta)

        # Get NNs under P1 (no text-conditioning — pure image NN)
        nns = get_nearest_neighbors(
            q_embed, image_embeds, image_meta, k=nn_k,
            exclude_concept=concept, exclude_region=region
        )
        nn_imgs  = [load_image(raw_dir, m) for _, _, m in nns]
        nn_metas = [m for _, _, m in nns]

        out_name = f"nn_grid_{model_name}_{concept.replace(' ', '_')}_{region}_{lang}_P1.png"
        make_nn_grid(
            q_img, nn_imgs, nn_metas, q_meta, delta_v,
            title=f"{model_name} | {concept} ({region}) | lang={lang} | P1 | Δ_L={delta_v:.3f}",
            out_path=str(figures_dir / out_name)
        )

        # Also show NN under P3 — does cultural prompt shift the neighborhood?
        # (We use the concept's P3 text embedding as the query anchor to find
        #  which images are closest to the native-language cultural text)
        def get_text_embed(c, l, pid, reg="global"):
            for i, m in enumerate(text_meta):
                if (m["concept"] == c and m["lang"] == l and
                        m["prompt_id"] == pid and m["region"] == reg):
                    return text_embeds[i]
            return None

        t_P3 = get_text_embed(concept, lang, "P3", region)
        if t_P3 is not None:
            nns_p3 = get_nearest_neighbors(
                t_P3, image_embeds, image_meta, k=nn_k,
                exclude_concept=concept, exclude_region=region
            )
            nn_imgs_p3  = [load_image(raw_dir, m) for _, _, m in nns_p3]
            nn_metas_p3 = [m for _, _, m in nns_p3]

            out_name_p3 = out_name.replace("P1", "P3_textanchor")
            make_nn_grid(
                q_img, nn_imgs_p3, nn_metas_p3, q_meta, delta_v,
                title=f"{model_name} | {concept} ({region}) | lang={lang} | P3-text-anchor",
                out_path=str(figures_dir / out_name_p3)
            )

    logger.info(f"\nAll qualitative outputs saved to {figures_dir}")


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    for model_name in args.model:
        run_qualitative(cfg, model_name, args.top_n, args.nn_k)


if __name__ == "__main__":
    main()
