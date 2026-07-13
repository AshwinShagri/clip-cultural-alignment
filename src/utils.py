"""
utils.py — Shared utilities for the CLIP Cultural Alignment project
"""

import os
import sys
import io
# Force UTF-8 output on Windows (avoids cp1252 encoding errors with Hindi/special chars)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import csv
import yaml
import logging
from datetime import datetime
from pathlib import Path
import numpy as np

# ─── Logging ────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s %(name)s %(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─── Config ─────────────────────────────────────────────────────────────────

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        # Walk up from this file's location to find config.yaml
        p = Path(__file__).parent
        for _ in range(4):
            candidate = p / "config.yaml"
            if candidate.exists():
                config_path = str(candidate)
                break
            p = p.parent
    if config_path is None:
        raise FileNotFoundError("config.yaml not found")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── Results logging ─────────────────────────────────────────────────────────

LOG_FIELDS = [
    "timestamp", "phase", "model", "language", "prompt_template",
    "question", "concept", "region", "metric_name", "metric_value",
    "p_value", "test_type", "n_samples", "notes"
]

def log_result(log_path: str, **kwargs):
    """Append one row to the results log CSV."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    write_header = not os.path.exists(log_path)
    row = {"timestamp": datetime.now().isoformat()}
    row.update(kwargs)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── Embedding helpers ───────────────────────────────────────────────────────

def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2-D array, or a 1-D vector."""
    if v.ndim == 1:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / (norms + 1e-12)


def mean_pairwise_cosine_similarity(embeddings: np.ndarray) -> float:
    """
    Mean pairwise cosine similarity for a set of L2-normalized embeddings.
    Uses N*(N-1) denominator (mean of off-diagonal entries), which is the
    mathematically correct form.  The proposal formula uses N² but that
    slightly underestimates; we note this correction in the methods section.
    """
    n = len(embeddings)
    if n < 2:
        return np.nan
    # Embeddings assumed L2-normalized → dot product = cosine similarity
    sim_matrix = embeddings @ embeddings.T          # (N, N)
    off_diag_sum = sim_matrix.sum() - np.trace(sim_matrix)
    return float(off_diag_sum / (n * (n - 1)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


# ─── Path helpers ────────────────────────────────────────────────────────────

def embedding_path(embed_dir: str, model_name: str, modality: str) -> str:
    return os.path.join(embed_dir, f"{model_name}_{modality}.npz")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ─── Prompt construction ─────────────────────────────────────────────────────

def build_prompts(cfg: dict) -> dict:
    """
    Returns dict: {(concept, lang, prompt_id, region): prompt_string}
    Dynamically looks up {lang}_concepts and {lang}_articles from config,
    so adding a new language only requires config changes, not code changes.
    """
    prompts_out = {}
    templates = cfg["prompts"]
    regions = cfg["regions"]

    for concept in cfg["all_concepts"]:
        for lang in cfg["languages"]:
            lang_templates = templates[lang]

            # Dynamic concept translation lookup: cfg["{lang}_concepts"]
            if lang == "en":
                c = concept
            else:
                lang_concepts_key = f"{lang}_concepts"
                c = cfg.get(lang_concepts_key, {}).get(concept, concept)

            # Dynamic article lookup: cfg["{lang}_articles"]
            lang_articles_key = f"{lang}_articles"
            article = cfg.get(lang_articles_key, {}).get(concept, "")

            for pid, template in lang_templates.items():
                # Build global prompt (no region)
                prompt = template.format(
                    concept=c,
                    article=article,
                    region=""
                ).strip()
                # Clean up artefacts from empty region substitution
                suffixes = [
                    "used in", "يستخدم في", "from", "aus", "de", "في", "من", "in"
                ]
                # Strip longest suffixes first to avoid partial matches
                for s in sorted(suffixes, key=len, reverse=True):
                    # Check for suffix with or without trailing spaces
                    if prompt.endswith(s):
                        prompt = prompt[:-len(s)].strip()
                    elif prompt.endswith(s + " "):
                        prompt = prompt[:-len(s)-1].strip()
                prompts_out[(concept, lang, pid, "global")] = prompt

                # Build region-specific prompts (P2 and P3)
                if pid in ("P2", "P3"):
                    for region in regions:
                        region_name = cfg["region_names"][lang].get(region, region)
                        prompt_r = template.format(
                            concept=c,
                            article=article,
                            region=region_name
                        ).strip()
                        prompts_out[(concept, lang, pid, region)] = prompt_r

    return prompts_out


if __name__ == "__main__":
    cfg = load_config()
    print("Config loaded. Concepts:", cfg["all_concepts"])
    print("Regions:", cfg["regions"])
    print("Models:", list(cfg["models"].keys()))

    # Quick prompt test
    prompts = build_prompts(cfg)
    test_keys = [
        ("spices", "en", "P1", "global"),
        ("spices", "es", "P3", "Africa"),
        ("car",    "ar", "P1", "global"),
    ]
    for k in test_keys:
        print(f"  {k} => {prompts.get(k, 'MISSING')}")
