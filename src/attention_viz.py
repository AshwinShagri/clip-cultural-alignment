"""
attention_viz.py — ViT Attention Map Visualization (Bonus)
===========================================================
Extracts and visualizes attention rollout maps from CLIP/SigLIP2 ViT encoders.
Allows inspection of which image regions the model attends to for Tier1 vs Tier2 concepts.

Usage:
    python src/attention_viz.py --model openclip --concept "spices" --region Africa
    python src/attention_viz.py --model siglip2  --concept "car"    --region Europe
"""

import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, l2_normalize

logger = get_logger("attention_viz")

DARK_BG = "#0d1117"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   choices=["openclip", "siglip2"], default="openclip")
    p.add_argument("--concept", default="spices")
    p.add_argument("--region",  default="Africa")
    p.add_argument("--n-images", type=int, default=4)
    p.add_argument("--config",  default=None)
    return p.parse_args()


def compute_attention_rollout(attn_weights_list: list[np.ndarray]) -> np.ndarray:
    """
    Attention rollout (Abnar & Zuidema, 2020): recursively multiply attention
    matrices across layers to get an overall attention map from [CLS] to patches.

    attn_weights_list: list of (num_heads, seq_len, seq_len) arrays, one per layer
    Returns: (seq_len-1,) array of attention weights from CLS to each patch
    """
    rollout = np.eye(attn_weights_list[0].shape[-1])  # identity start
    for attn in attn_weights_list:
        # Average over heads
        attn_avg = attn.mean(axis=0)       # (seq_len, seq_len)
        # Add residual connection
        attn_res = attn_avg + np.eye(attn_avg.shape[0])
        attn_res /= attn_res.sum(axis=-1, keepdims=True)
        rollout = attn_res @ rollout

    # Check if there is a CLS token or if sequence length is a perfect square
    seq_len = rollout.shape[0]
    is_square = int(np.round(np.sqrt(seq_len))) ** 2 == seq_len
    if is_square:
        # No CLS token (e.g., SigLIP) — take the mean attention over all query tokens
        cls_attn = rollout.mean(axis=0)
    else:
        # CLS token at index 0 (e.g., CLIP) — take CLS attention to all patch tokens
        cls_attn = rollout[0, 1:]
    return cls_attn


def get_attention_maps(model, processor, img: Image.Image, device: str):
    """Extract attention maps from all transformer layers."""
    import torch

    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "pixel_values" or True}

    with torch.no_grad():
        outputs = model.vision_model(
            **{k: v for k, v in inputs.items() if k == "pixel_values"},
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )

    # outputs.attentions: tuple of (batch, heads, seq, seq) per layer
    attn_list = []
    for layer_attn in outputs.attentions:
        attn = layer_attn.squeeze(0).cpu().numpy()  # (heads, seq, seq)
        attn_list.append(attn)

    return attn_list


def visualize_attention(
    img: Image.Image,
    attn_rollout: np.ndarray,
    concept: str,
    region: str,
    model_name: str,
    patch_size: int = 32,
    ax=None
):
    """Overlay attention heatmap on image."""
    img_size = 224
    n_patches = img_size // patch_size
    expected = n_patches * n_patches

    # Resize rollout to (n_patches, n_patches)
    attn_2d = attn_rollout[:expected].reshape(n_patches, n_patches)
    attn_norm = (attn_2d - attn_2d.min()) / (attn_2d.max() - attn_2d.min() + 1e-8)

    # Upsample
    attn_up = np.array(Image.fromarray((attn_norm * 255).astype(np.uint8)).resize(
        (img_size, img_size), Image.BILINEAR
    )) / 255.0

    img_resized = np.array(img.resize((img_size, img_size)))

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(img_resized)
    ax.imshow(attn_up, cmap="hot", alpha=0.5, vmin=0, vmax=1)
    ax.set_title(f"{concept}\n{region}", fontsize=8, color="white")
    ax.axis("off")


def run_attention_visualization(cfg: dict, model_name: str, concept: str,
                                 region: str, n_images: int):
    """Full attention visualization pipeline."""
    import torch
    from transformers import AutoModel, AutoProcessor
    from embed import load_image_embeddings

    logger.info(f"Attention visualization: {model_name} | {concept} | {region}")

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    repo_id  = cfg["models"][model_name]["repo_id"]
    raw_dir  = Path(cfg["paths"]["data_raw"])
    fig_dir  = Path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    model     = AutoModel.from_pretrained(repo_id, attn_implementation="eager").eval().to(device)
    processor = AutoProcessor.from_pretrained(repo_id)

    # Get patch size from model config
    patch_size = getattr(model.vision_model.config, "patch_size", 32)

    # Load image embeddings to find images of this concept/region
    image_embeds, image_meta = load_image_embeddings(cfg, model_name)

    # Find image files for this (concept, region)
    target_meta = [m for m in image_meta
                   if m["object"] == concept and m["region"] == region][:n_images]

    if not target_meta:
        logger.warning(f"No images found for ({concept}, {region})")
        return

    n_imgs = len(target_meta)
    fig, axes = plt.subplots(n_imgs, 2, figsize=(8, 3.5 * n_imgs))
    fig.patch.set_facecolor(DARK_BG)

    if n_imgs == 1:
        axes = axes[np.newaxis, :]

    for i, meta in enumerate(target_meta):
        img_path = raw_dir / meta["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Could not load {img_path}: {e}")
            continue

        # Show original
        axes[i, 0].imshow(np.array(img.resize((224, 224))))
        axes[i, 0].set_title(f"Original\n{concept} | {region}", fontsize=8, color="white")
        axes[i, 0].axis("off")
        axes[i, 0].set_facecolor(DARK_BG)

        # Extract and show attention rollout
        try:
            attn_list = get_attention_maps(model, processor, img, device)
            rollout   = compute_attention_rollout(attn_list)
            visualize_attention(img, rollout, concept, region, model_name,
                                patch_size=patch_size, ax=axes[i, 1])
            axes[i, 1].set_facecolor(DARK_BG)
        except Exception as e:
            logger.warning(f"Attention extraction failed: {e}")
            axes[i, 1].axis("off")

    plt.suptitle(
        f"ViT Attention Rollout: {model_name.upper()}\n"
        f"Concept: '{concept}'  |  Region: {region}",
        fontsize=11, color="white", y=1.01
    )
    plt.tight_layout()

    out = fig_dir / f"attention_{model_name}_{concept.replace(' ', '_')}_{region}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"Saved attention map -> {out}")

    return str(out)


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    run_attention_visualization(
        cfg, args.model, args.concept, args.region, args.n_images
    )


if __name__ == "__main__":
    main()
