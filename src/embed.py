"""
embed.py — Compute and cache image + text embeddings
======================================================
Usage:
    python src/embed.py --model openclip --modality image
    python src/embed.py --model siglip2  --modality image
    python src/embed.py --model openclip --modality text
    python src/embed.py --model siglip2  --modality text

Saves:
    data/embeddings/{model}_{modality}.npz
    Keys: embeddings (N, D), labels (N,), metadata array

For text embeddings, produces one vector per (concept, lang, prompt_id, region)
and stores labels as JSON-encoded strings.
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, l2_normalize, ensure_dir, embedding_path

logger = get_logger("embed")


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model_and_processor(model_cfg: dict, device: str):
    """Load model + processor from HuggingFace. Returns (model, processor)."""
    import torch
    from transformers import AutoModel, AutoProcessor

    repo_id = model_cfg["repo_id"]
    logger.info(f"Loading {repo_id} on {device} ...")
    model = AutoModel.from_pretrained(repo_id).eval().to(device)
    processor = AutoProcessor.from_pretrained(repo_id)
    logger.info(f"  → embed_dim={model_cfg['embed_dim']}")
    return model, processor


def get_image_features(model, processor, images, device: str, batch_size: int = 32):
    """
    Batch-encode a list of PIL images.
    Returns L2-normalized numpy array of shape (N, D).
    """
    import torch
    all_embeds = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        # Extract PyTorch tensor if output is a Hugging Face BaseModelOutputWithPooling
        if not torch.is_tensor(feats):
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            else:
                feats = feats[0]
        feats = feats.float().cpu().numpy()
        all_embeds.append(feats)
        if (start // batch_size) % 5 == 0:
            logger.info(f"  Image batches: {start + len(batch)}/{len(images)}")
    embeds = np.concatenate(all_embeds, axis=0)
    return l2_normalize(embeds)


def get_text_features(model, processor, texts: list[str], device: str, batch_size: int = 64):
    """
    Batch-encode a list of text strings.
    Returns L2-normalized numpy array of shape (N, D).
    """
    import torch
    all_embeds = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        is_siglip = "siglip" in str(type(model)).lower()
        padding_strategy = "max_length" if is_siglip else True
        max_len = 64 if is_siglip else None

        inputs = processor(text=batch, return_tensors="pt", padding=padding_strategy, max_length=max_len, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        # Extract PyTorch tensor if output is a Hugging Face BaseModelOutputWithPooling
        if not torch.is_tensor(feats):
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            else:
                feats = feats[0]
        feats = feats.float().cpu().numpy()
        all_embeds.append(feats)
    embeds = np.concatenate(all_embeds, axis=0)
    return l2_normalize(embeds)


# ─── Image embedding pipeline ────────────────────────────────────────────────

def embed_images(cfg: dict, model_name: str, device: str, batch_size: int):
    """
    Load all images from metadata.csv, compute embeddings, save to disk.
    """
    import pandas as pd
    from PIL import Image

    meta_path = Path(cfg["paths"]["metadata"])
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found at {meta_path}. Run fetch_data.py first."
        )

    df = pd.read_csv(meta_path)
    logger.info(f"Found {len(df)} images in metadata.csv")

    raw_dir = Path(cfg["paths"]["data_raw"])

    # Load all images
    logger.info("Loading images from disk ...")
    images_pil = []
    valid_rows = []
    for _, row in df.iterrows():
        img_path = raw_dir / row["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            images_pil.append(img)
            valid_rows.append(row)
        except Exception as e:
            logger.warning(f"  Could not load {img_path}: {e}")

    logger.info(f"Loaded {len(images_pil)} valid images.")

    model_cfg = cfg["models"][model_name]
    model, processor = load_model_and_processor(model_cfg, device)

    logger.info(f"Computing image embeddings (batch_size={batch_size}) ...")
    embeds = get_image_features(model, processor, images_pil, device, batch_size)

    # Build label arrays
    labels = []
    for row in valid_rows:
        labels.append(json.dumps({
            "image_path": row["image_path"],
            "object": row["object"],
            "region": row["region"],
            "img_idx": int(row.get("img_idx", 0))
        }))

    out_path = embedding_path(cfg["paths"]["embeddings"], model_name, "image")
    ensure_dir(cfg["paths"]["embeddings"])
    np.savez_compressed(out_path,
                        embeddings=embeds,
                        labels=np.array(labels, dtype=object))
    logger.info(f"Saved image embeddings → {out_path}  shape={embeds.shape}")
    return embeds, labels


# ─── Text embedding pipeline ─────────────────────────────────────────────────

def embed_text(cfg: dict, model_name: str, device: str, batch_size: int):
    """
    Compute text embeddings for all (concept, lang, prompt_id, region) combos.
    """
    from utils import build_prompts

    prompts_dict = build_prompts(cfg)

    # Collect in order
    keys   = list(prompts_dict.keys())
    texts  = [prompts_dict[k] for k in keys]
    labels = [json.dumps({"concept": k[0], "lang": k[1], "prompt_id": k[2], "region": k[3]})
              for k in keys]

    logger.info(f"Computing text embeddings for {len(texts)} prompts ...")
    for i, (k, t) in enumerate(zip(keys[:5], texts[:5])):
        logger.info(f"  Sample {i}: {k} → '{t}'")

    model_cfg = cfg["models"][model_name]
    model, processor = load_model_and_processor(model_cfg, device)

    embeds = get_text_features(model, processor, texts, device, batch_size)

    out_path = embedding_path(cfg["paths"]["embeddings"], model_name, "text")
    ensure_dir(cfg["paths"]["embeddings"])
    np.savez_compressed(out_path,
                        embeddings=embeds,
                        labels=np.array(labels, dtype=object))
    logger.info(f"Saved text embeddings → {out_path}  shape={embeds.shape}")
    return embeds, labels


# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_image_embeddings(cfg: dict, model_name: str):
    """Load cached image embeddings. Returns (embeds, meta_list)."""
    path = embedding_path(cfg["paths"]["embeddings"], model_name, "image")
    data = np.load(path, allow_pickle=True)
    embeds = data["embeddings"]
    meta   = [json.loads(s) for s in data["labels"]]
    return embeds, meta


def load_text_embeddings(cfg: dict, model_name: str):
    """Load cached text embeddings. Returns (embeds, meta_list)."""
    path = embedding_path(cfg["paths"]["embeddings"], model_name, "text")
    data = np.load(path, allow_pickle=True)
    embeds = data["embeddings"]
    meta   = [json.loads(s) for s in data["labels"]]
    return embeds, meta


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Compute and cache embeddings")
    p.add_argument("--model",    choices=["openclip", "siglip2"], required=True)
    p.add_argument("--modality", choices=["image", "text"],       required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device",  default=None, help="cuda or cpu (auto-detected if not set)")
    p.add_argument("--overwrite", action="store_true", help="Recompute even if cached")
    p.add_argument("--config",  default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_path = embedding_path(cfg["paths"]["embeddings"], args.model, args.modality)
    if os.path.exists(out_path):
        if not args.overwrite:
            logger.info(f"Embeddings already cached at {out_path}. Use --overwrite to recompute.")
            return

    if args.modality == "image":
        embed_images(cfg, args.model, device, args.batch_size)
    else:
        embed_text(cfg, args.model, device, args.batch_size)


if __name__ == "__main__":
    main()
