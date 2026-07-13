"""
fetch_data.py — Download and cache a curated GeoDE subset
==========================================================
Usage:
    python src/fetch_data.py [--limit 20] [--dry-run]

Streams GeoDE from HuggingFace via the datasets library with predicate
filtering on `object` and `region`.  Images are saved as JPEG files under
data/raw/<concept>/<region>/<idx>.jpg and a metadata CSV is written to
data/raw/metadata.csv.

Use --limit N for a quick dry-run before committing to the full ~3,000-image pull.
"""

import os
import sys
import argparse
import csv
import io
from pathlib import Path

# Add src to path so we can import utils
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger, ensure_dir

logger = get_logger("fetch_data")


def parse_args():
    p = argparse.ArgumentParser(description="Download GeoDE subset")
    p.add_argument("--limit", type=int, default=None,
                   help="Max images per (concept, region) cell — use for dry-run")
    p.add_argument("--dry-run", action="store_true",
                   help="Just count rows, don't save images")
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--images-per-cell", type=int, default=None,
                   help="Override config images_per_class_per_region")
    return p.parse_args()


def fetch_geode_subset(cfg: dict, images_per_cell: int, dry_run: bool, limit_per_cell: int):
    """
    Stream GeoDE and save a balanced subset.
    Returns list of metadata dicts.
    """
    from datasets import load_dataset
    from PIL import Image

    target_concepts = set(cfg["all_concepts"])
    target_regions  = set(cfg["regions"])
    n_per_cell      = limit_per_cell if limit_per_cell else images_per_cell

    raw_dir = Path(cfg["paths"]["data_raw"])
    meta_path = Path(cfg["paths"]["metadata"])
    ensure_dir(str(raw_dir))

    # Counters: {(concept, region): count}
    cell_counts = {(c, r): 0 for c in target_concepts for r in target_regions}
    metadata = []

    logger.info("Loading GeoDE dataset (streaming mode) ...")
    ds = load_dataset("MLap/GeoDE", split="train", streaming=True, trust_remote_code=True)

    total_seen = 0
    total_saved = 0

    for row in ds:
        concept = str(row.get("object", "")).replace("_", " ").lower()
        region  = row.get("region", "")

        if concept not in target_concepts or region not in target_regions:
            continue

        key = (concept, region)
        if cell_counts[key] >= n_per_cell:
            # Check if all cells are full
            if all(v >= n_per_cell for v in cell_counts.values()):
                logger.info("All cells full — stopping early.")
                break
            continue

        total_seen += 1
        cell_counts[key] += 1
        img_idx = cell_counts[key]

        if not dry_run:
            # Save image
            concept_dir = raw_dir / concept.replace(" ", "_") / region
            ensure_dir(str(concept_dir))
            img_path = concept_dir / f"{img_idx:04d}.jpg"

            if not img_path.exists():
                # image column is either PIL.Image or bytes
                img = row["image"]
                if isinstance(img, bytes):
                    img = Image.open(io.BytesIO(img))
                elif not hasattr(img, "save"):
                    img = Image.fromarray(img)
                img = img.convert("RGB")
                img.save(str(img_path), "JPEG", quality=90)

            rel_path = str(img_path.relative_to(raw_dir))
            metadata.append({
                "image_path": rel_path,
                "object": concept,
                "region": region,
                "img_idx": img_idx
            })
            total_saved += 1

            if total_saved % 100 == 0:
                logger.info(f"  Saved {total_saved} images so far ...")
        else:
            metadata.append({
                "image_path": f"DRY_RUN/{concept}/{region}/{img_idx}.jpg",
                "object": concept,
                "region": region,
                "img_idx": img_idx
            })

    logger.info(f"Done. Total seen: {total_seen}, saved: {total_saved if not dry_run else '(dry-run)'}")

    # Summary table
    logger.info("\n=== Cell counts (concept × region) ===")
    concepts_sorted = sorted(target_concepts)
    regions_sorted  = sorted(target_regions)
    header = f"{'Concept':<25}" + "".join(f"{r:<15}" for r in regions_sorted) + "TOTAL"
    logger.info(header)
    for c in concepts_sorted:
        row_counts = [cell_counts[(c, r)] for r in regions_sorted]
        row_str = f"{c:<25}" + "".join(f"{cnt:<15}" for cnt in row_counts) + str(sum(row_counts))
        logger.info(row_str)

    # Write metadata CSV
    if not dry_run and metadata:
        with open(meta_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "object", "region", "img_idx"])
            writer.writeheader()
            writer.writerows(metadata)
        logger.info(f"Metadata written → {meta_path} ({len(metadata)} rows)")

    return metadata, cell_counts


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    images_per_cell = args.images_per_cell or cfg["dataset"]["images_per_class_per_region"]

    logger.info("=== GeoDE Data Fetch ===")
    logger.info(f"Concepts ({len(cfg['all_concepts'])}): {cfg['all_concepts']}")
    logger.info(f"Regions  ({len(cfg['regions'])}): {cfg['regions']}")
    logger.info(f"Images per cell: {images_per_cell} (override limit: {args.limit})")
    logger.info(f"Dry-run: {args.dry_run}")

    if not args.dry_run:
        meta_path = Path(cfg["paths"]["metadata"])
        if meta_path.exists():
            existing = sum(1 for _ in open(meta_path)) - 1
            logger.info(f"metadata.csv already exists ({existing} rows). Delete to re-fetch.")
            return

    metadata, cell_counts = fetch_geode_subset(
        cfg=cfg,
        images_per_cell=images_per_cell,
        dry_run=args.dry_run,
        limit_per_cell=args.limit
    )

    missing = [(c, r) for (c, r), cnt in cell_counts.items() if cnt < images_per_cell]
    if missing:
        logger.warning(f"Cells with fewer than {images_per_cell} images: {missing}")
    else:
        logger.info("All cells have the target number of images.")


if __name__ == "__main__":
    main()
