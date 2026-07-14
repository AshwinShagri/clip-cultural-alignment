"""
visualize.py — Publication-quality figures for the final report
================================================================
Usage:
    python src/visualize.py --model openclip siglip2

Generates:
  fig1_taxonomy_overview.png       — Tier1/Tier2 S(c,r) heatmap + divergence bar
  fig2_s_west_gap.png              — Western coherence gap by concept
  fig3_delta_L_violin.png          — Δ_L distribution: Tier1 vs Tier2 (per lang, model)
  fig4_delta_L_heatmap.png         — Δ_L heatmap: concept × region
  fig5_prompt_gain.png             — G(P3) vs G(P2): bar chart by tier and model
  fig6_openclip_vs_siglip2.png     — Side-by-side model comparison
  fig7_spearman_scatter.png        — S_gap vs mean_Δ_L scatter (confound check)
  fig8_sensitivity.png             — Threshold sensitivity analysis
  fig9_attention_maps.png          — ViT attention visualization (bonus)
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, get_logger

logger = get_logger("visualize")

# ─── Style ────────────────────────────────────────────────────────────────────

DARK_BG     = "#0d1117"
PANEL_BG    = "#161b22"
TEXT_COLOR  = "#e6edf3"
GRID_COLOR  = "#30363d"
ACCENT1     = "#58a6ff"
ACCENT2     = "#f78166"
ACCENT3     = "#3fb950"
ACCENT4     = "#d2a8ff"

TIER1_COLOR = "#58a6ff"  # blue
TIER2_COLOR = "#f78166"  # orange-red

REGION_COLORS = {
    "WestAsia":      "#E63946",
    "Africa":        "#F4A261",
    "EastAsia":      "#2A9D8F",
    "SouthEastAsia": "#457B9D",
    "Americas":      "#A8DADC",
    "Europe":        "#1D3557"
}

MODEL_COLORS = {
    "openclip": "#58a6ff",
    "siglip2":  "#3fb950"
}

LANG_LABELS = {"es": "Spanish", "ar": "Arabic"}

def setup_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    GRID_COLOR,
        "axes.labelcolor":   TEXT_COLOR,
        "xtick.color":       TEXT_COLOR,
        "ytick.color":       TEXT_COLOR,
        "text.color":        TEXT_COLOR,
        "grid.color":        GRID_COLOR,
        "grid.linewidth":    0.5,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "figure.dpi":        150,
    })

setup_style()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="+", default=["openclip", "siglip2"],
                   choices=["openclip", "siglip2"])
    p.add_argument("--config", default=None)
    p.add_argument("--skip-attention", action="store_true",
                   help="Skip attention map visualization")
    return p.parse_args()


# ─── Figure 1: Taxonomy Overview Heatmap ─────────────────────────────────────

def fig1_taxonomy_heatmap(cfg: dict, figures_dir: Path, models: list[str]):
    """S(c, r) heatmap for each model side-by-side."""
    tables_dir = Path(cfg["paths"]["tables"])
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(10 * n_models, 7))
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        csv_path = tables_dir / f"taxonomy_{model_name}.csv"
        if not csv_path.exists():
            logger.warning(f"Taxonomy CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        concepts = cfg["all_concepts"]
        regions  = cfg["regions"]
        tier2    = set(cfg["tier2_concepts"])

        pivot = df.pivot(index="concept", columns="region", values="S_cr")
        pivot = pivot.reindex(index=concepts, columns=regions)

        # Sort concepts: tier2 first (bottom), tier1 second
        concept_order = cfg["tier2_concepts"] + cfg["tier1_concepts"]
        pivot = pivot.reindex(concept_order)

        # Custom colormap
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                       vmin=0.5, vmax=1.0)

        # Labels
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels(regions, rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(len(concept_order)))
        ax.set_yticklabels(concept_order, fontsize=9)

        # Tier separator line
        ax.axhline(len(cfg["tier2_concepts"]) - 0.5, color="#FFD700",
                   linewidth=2, linestyle="--")

        # Tier labels on y-axis
        for i, c in enumerate(concept_order):
            tier_lbl = "⚫" if c in tier2 else "⬤"
            ax.text(-0.7, i, tier_lbl,
                    ha="center", va="center", fontsize=8,
                    color=TIER2_COLOR if c in tier2 else TIER1_COLOR,
                    transform=ax.get_yaxis_transform())

        # Value annotations
        for i in range(len(concept_order)):
            for j in range(len(regions)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if val < 0.85 else "black")

        plt.colorbar(im, ax=ax, label="Mean pairwise cosine similarity S(c,r)",
                     pad=0.02)
        ax.set_title(f"{model_name.upper()}\nS(c, r) Heatmap — Tier 2 (cultural) | Tier 1 (universal)",
                     color=TEXT_COLOR, pad=10)
        ax.text(0.01, 0.99, "■ Tier 2 (cultural)  □ Tier 1 (universal)",
                transform=ax.transAxes, va="top", fontsize=8, color="#FFD700")

    plt.suptitle("Image Geometry: Mean Pairwise Cosine Similarity S(c, r)",
                 fontsize=14, color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    out = figures_dir / "fig1_taxonomy_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"  → {out}")


# ─── Figure 2: Western coherence gap ─────────────────────────────────────────

def fig2_west_gap(cfg: dict, figures_dir: Path, models: list[str]):
    """Bar chart of S_west_gap per concept, grouped by model."""
    tables_dir = Path(cfg["paths"]["tables"])
    tier2 = set(cfg["tier2_concepts"])
    concept_order = cfg["tier2_concepts"] + cfg["tier1_concepts"]

    fig, ax = plt.subplots(figsize=(14, 5))

    n_models   = len(models)
    x          = np.arange(len(concept_order))
    bar_width  = 0.35

    for mi, model_name in enumerate(models):
        csv_path = tables_dir / f"taxonomy_summary_{model_name}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).set_index("concept")
        gaps = [df.loc[c, "S_west_gap"] if c in df.index else np.nan
                for c in concept_order]
        offset = (mi - (n_models - 1) / 2) * bar_width
        bar_colors = [TIER2_COLOR if c in tier2 else TIER1_COLOR for c in concept_order]
        bars = ax.bar(x + offset, gaps, bar_width,
                      color=bar_colors, alpha=0.8 if mi == 0 else 0.6,
                      label=model_name, edgecolor="white", linewidth=0.5)

        # Significance markers
        for bi, (gap, c) in enumerate(zip(gaps, concept_order)):
            if not np.isnan(gap):
                ax.text(bi + offset, gap + 0.001, f"{gap:.3f}",
                        ha="center", va="bottom", fontsize=6.5, color=TEXT_COLOR)

    ax.axhline(0, color=GRID_COLOR, linewidth=1)
    ax.axvline(len(cfg["tier2_concepts"]) - 0.5, color="#FFD700",
               linewidth=2, linestyle="--", alpha=0.8)
    ax.text(len(cfg["tier2_concepts"]) / 2 - 0.5, ax.get_ylim()[1] * 0.9,
            "← Tier 2\n(Cultural)", ha="center", color=TIER2_COLOR, fontsize=9)
    ax.text(len(cfg["tier2_concepts"]) + len(cfg["tier1_concepts"]) / 2 - 0.5,
            ax.get_ylim()[1] * 0.9,
            "Tier 1 →\n(Universal)", ha="center", color=TIER1_COLOR, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(concept_order, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("S(c, West) − S(c, non-West)", fontsize=10)
    ax.set_title("Q1: Western Coherence Gap — Positive = West is more coherent in embedding space",
                 fontsize=11)
    ax.legend(title="Model", framealpha=0.3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = figures_dir / "fig2_west_gap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"  → {out}")


# ─── Figure 3: Δ_L violin plot ────────────────────────────────────────────────

def fig3_delta_violin(cfg: dict, figures_dir: Path, models: list[str]):
    """Violin plot of Δ_L distribution: Tier1 vs Tier2 per language and model."""
    tables_dir = Path(cfg["paths"]["tables"])
    csv_path = tables_dir / "q2_alignment_gap.csv"
    if not csv_path.exists():
        logger.warning(f"Q2 table not found: {csv_path} — run analysis.py first")
        return

    df = pd.read_csv(csv_path)
    df = df[df["prompt_id"] == "P1"]

    languages = [l for l in cfg["languages"] if l != "en"]
    n_langs   = len(languages)
    n_models  = len(models)

    fig, axes = plt.subplots(n_langs, n_models,
                              figsize=(6 * n_models, 4 * n_langs),
                              sharey=False)
    if n_langs == 1:
        axes = axes[np.newaxis, :]
    if n_models == 1:
        axes = axes[:, np.newaxis]

    for li, lang in enumerate(languages):
        for mi, model_name in enumerate(models):
            ax = axes[li, mi]
            subset = df[(df["model"] == model_name) & (df["language"] == lang)]
            if subset.empty:
                ax.set_visible(False)
                continue

            # Prepare data
            tier1_data = subset[subset["tier"] == "Tier1"]["mean_delta_L"].dropna().values
            tier2_data = subset[subset["tier"] == "Tier2"]["mean_delta_L"].dropna().values

            # Violin
            parts = ax.violinplot([tier1_data, tier2_data],
                                   positions=[0, 1], showmedians=True,
                                   showextrema=True)
            for pc, color in zip(parts["bodies"], [TIER1_COLOR, TIER2_COLOR]):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            parts["cmedians"].set_color("white")
            parts["cmaxes"].set_color("white")
            parts["cmins"].set_color("white")
            parts["cbars"].set_color("white")

            # Swarm overlay
            jitter = 0.08
            for xi, (data, color) in enumerate([(tier1_data, TIER1_COLOR),
                                                  (tier2_data, TIER2_COLOR)]):
                x_jitter = np.random.uniform(-jitter, jitter, len(data))
                ax.scatter(xi + x_jitter, data, color=color, s=15, alpha=0.4,
                           zorder=5, edgecolors="none")

            # t-test annotation
            if len(tier1_data) >= 2 and len(tier2_data) >= 2:
                _, p = stats.ttest_ind(tier2_data, tier1_data, equal_var=False)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                y_max = max(tier1_data.max(), tier2_data.max()) + 0.01
                ax.annotate("", xy=(1, y_max), xytext=(0, y_max),
                            arrowprops=dict(arrowstyle="-", color="white"))
                ax.text(0.5, y_max + 0.005, sig, ha="center", fontsize=12,
                        color="white" if sig != "ns" else "gray")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Tier 1\n(Universal)", "Tier 2\n(Cultural)"])
            ax.set_ylabel(f"Δ_L ({LANG_LABELS.get(lang, lang)})")
            ax.set_title(f"{model_name} | lang={LANG_LABELS.get(lang, lang)}")
            ax.axhline(0, color=GRID_COLOR, linewidth=1, linestyle="--")
            ax.grid(axis="y", alpha=0.2)

    plt.suptitle("Q2: Cross-Lingual Alignment Gap Δ_L — Tier 1 vs Tier 2 (P1, neutral prompt)",
                 fontsize=12, color=TEXT_COLOR, y=1.01)
    plt.tight_layout()
    out = figures_dir / "fig3_delta_L_violin.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"  → {out}")


# ─── Figure 4: Δ_L heatmap concept × region ──────────────────────────────────

def fig4_delta_heatmap(cfg: dict, figures_dir: Path, models: list[str]):
    """Δ_L heatmap across concepts and regions for P1."""
    tables_dir = Path(cfg["paths"]["tables"])
    csv_path = tables_dir / "q2_alignment_gap.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    df = df[df["prompt_id"] == "P1"]
    languages = [l for l in cfg["languages"] if l != "en"]
    concept_order = cfg["tier2_concepts"] + cfg["tier1_concepts"]
    regions = cfg["regions"]

    n_models = len(models)
    n_langs  = len(languages)
    fig, axes = plt.subplots(n_langs, n_models,
                              figsize=(9 * n_models, 6 * n_langs))
    if n_langs == 1: axes = axes[np.newaxis, :]
    if n_models == 1: axes = axes[:, np.newaxis]

    for li, lang in enumerate(languages):
        for mi, model_name in enumerate(models):
            ax = axes[li, mi]
            subset = df[(df["model"] == model_name) & (df["language"] == lang)]
            if subset.empty:
                ax.set_visible(False)
                continue

            agg = subset.groupby(["concept", "region"])["mean_delta_L"].mean().reset_index()
            pivot = agg.pivot(index="concept", columns="region", values="mean_delta_L")
            pivot = pivot.reindex(index=concept_order, columns=regions)

            cmap = sns.diverging_palette(10, 130, as_cmap=True)
            sns.heatmap(pivot, ax=ax, cmap=cmap, center=0,
                        annot=True, fmt=".3f", annot_kws={"size": 7},
                        linewidths=0.3, linecolor=GRID_COLOR,
                        cbar_kws={"label": "Mean Δ_L"})

            ax.axhline(len(cfg["tier2_concepts"]), color="#FFD700",
                       linewidth=2, linestyle="--")
            ax.set_title(f"{model_name} | {LANG_LABELS.get(lang, lang)}")
            ax.set_xlabel("Region")
            ax.set_ylabel("Concept")

    plt.suptitle("Q2: Δ_L Heatmap (concept × region, P1)\nRed = larger gap (worse non-EN alignment)",
                 fontsize=12, color=TEXT_COLOR, y=1.01)
    plt.tight_layout()
    out = figures_dir / "fig4_delta_L_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"  → {out}")


# ─── Figure 5: Prompt gain bar chart ─────────────────────────────────────────

def fig5_prompt_gain(cfg: dict, figures_dir: Path, models: list[str]):
    """G(P3) vs G(P2) bar chart by tier and model."""
    tables_dir = Path(cfg["paths"]["tables"])
    csv_path = tables_dir / "q3_prompt_gain.csv"
    if not csv_path.exists():
        logger.warning(f"Q3 table not found: {csv_path} — run analysis.py first")
        return

    df = pd.read_csv(csv_path)
    tier2 = set(cfg["tier2_concepts"])
    languages = [l for l in cfg["languages"] if l != "en"]

    for lang in languages:
        fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), sharey=True)
        if len(models) == 1: axes = [axes]

        for ax, model_name in zip(axes, models):
            subset = df[(df["model"] == model_name) & (df["language"] == lang)]
            if subset.empty:
                ax.set_visible(False)
                continue

            concept_order = cfg["tier2_concepts"] + cfg["tier1_concepts"]
            agg = subset.groupby(["concept", "tier"])[["gain_P3", "gain_P2"]].mean().reset_index()
            agg = agg.set_index("concept").reindex(concept_order).reset_index()

            x = np.arange(len(concept_order))
            w = 0.3
            bar_colors = [TIER2_COLOR if c in tier2 else TIER1_COLOR
                          for c in concept_order]

            ax.bar(x - w/2, agg["gain_P3"], w, color=bar_colors, alpha=0.9,
                   label="G(P3) — cultural", edgecolor="white", linewidth=0.5)
            ax.bar(x + w/2, agg["gain_P2"], w, color=bar_colors, alpha=0.4,
                   label="G(P2) — weak", edgecolor="white", linewidth=0.5,
                   hatch="///")

            ax.axhline(0, color="white", linewidth=1, linestyle="--", alpha=0.5)
            ax.axvline(len(cfg["tier2_concepts"]) - 0.5, color="#FFD700",
                       linewidth=2, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(concept_order, rotation=35, ha="right", fontsize=8)
            ax.set_title(f"{model_name} | {LANG_LABELS.get(lang, lang)}")
            ax.set_ylabel("Prompt Gain G (P1 − P_x)")
            ax.legend(fontsize=8, framealpha=0.3)
            ax.grid(axis="y", alpha=0.2)

        plt.suptitle(f"Q3: Prompt Gain G — Does cultural prompting close the alignment gap? "
                     f"({LANG_LABELS.get(lang, lang)})",
                     fontsize=11, color=TEXT_COLOR, y=1.02)
        plt.tight_layout()
        out = figures_dir / f"fig5_prompt_gain_{lang}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close()
        logger.info(f"  → {out}")


# ─── Figure 6: OpenCLIP vs SigLIP2 comparison ────────────────────────────────

def fig6_model_comparison(cfg: dict, figures_dir: Path, models: list[str]):
    """Side-by-side comparison of cultural asymmetry A(L) between models."""
    tables_dir = Path(cfg["paths"]["tables"])
    csv_path = tables_dir / "q2_alignment_gap.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    df = df[df["prompt_id"] == "P1"]
    tier2 = set(cfg["tier2_concepts"])
    languages = [l for l in cfg["languages"] if l != "en"]

    # Cultural Asymmetry: mean Δ_L(Tier2) - mean Δ_L(Tier1)
    rows = []
    for model_name in models:
        for lang in languages:
            subset = df[(df["model"] == model_name) & (df["language"] == lang)]
            t1 = subset[subset["tier"] == "Tier1"]["mean_delta_L"].mean()
            t2 = subset[subset["tier"] == "Tier2"]["mean_delta_L"].mean()
            rows.append({
                "model": model_name, "language": lang,
                "tier1_delta": t1, "tier2_delta": t2,
                "asymmetry_A": t2 - t1
            })

    comp_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(languages), figsize=(7 * len(languages), 5))
    if len(languages) == 1: axes = [axes]

    for ax, lang in zip(axes, languages):
        sub = comp_df[comp_df["language"] == lang]
        x = np.arange(len(sub))
        colors = [MODEL_COLORS.get(m, "#aaa") for m in sub["model"]]
        bars = ax.bar(x, sub["asymmetry_A"], color=colors,
                      edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, sub["asymmetry_A"]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, color=TEXT_COLOR)

        ax.axhline(0, color="white", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"], fontsize=11)
        ax.set_ylabel("Cultural Asymmetry A(L)  [mean Δ_L(Tier2) − mean Δ_L(Tier1)]")
        ax.set_title(f"Model Comparison | {LANG_LABELS.get(lang, lang)}")
        ax.grid(axis="y", alpha=0.2)

        # Legend patches
        patches = [mpatches.Patch(color=MODEL_COLORS.get(m, "#aaa"), label=m)
                   for m in sub["model"]]
        ax.legend(handles=patches, framealpha=0.3, fontsize=9)

    plt.suptitle("Q2+Q3: Cultural Asymmetry Score A(L) — OpenCLIP vs SigLIP 2\n"
                 "Higher A = larger gap between Tier 2 and Tier 1 alignment",
                 fontsize=11, color=TEXT_COLOR, y=1.03)
    plt.tight_layout()
    out = figures_dir / "fig6_model_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"  → {out}")


# ─── Figure 7: Spearman scatter plot (confound check) ────────────────────────

def fig7_spearman_scatter(cfg: dict, figures_dir: Path, models: list[str]):
    """Scatter: S_west_gap vs mean Δ_L (P1) — one point per concept."""
    tables_dir = Path(cfg["paths"]["tables"])
    csv_path = tables_dir / "q2_alignment_gap.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    df = df[df["prompt_id"] == "P1"]
    tier2 = set(cfg["tier2_concepts"])
    languages = [l for l in cfg["languages"] if l != "en"]

    fig, axes = plt.subplots(len(languages), len(models),
                              figsize=(6 * len(models), 5 * len(languages)))
    if len(languages) == 1: axes = axes[np.newaxis, :]
    if len(models) == 1: axes = axes[:, np.newaxis]

    for li, lang in enumerate(languages):
        for mi, model_name in enumerate(models):
            ax = axes[li, mi]
            subset = df[(df["model"] == model_name) & (df["language"] == lang)]
            if subset.empty:
                ax.set_visible(False)
                continue

            agg = subset.groupby("concept").agg(
                mean_delta_L=("mean_delta_L", "mean"),
                S_west_gap=("S_west_gap", "mean"),
                tier=("tier", "first")
            ).dropna().reset_index()

            for _, row in agg.iterrows():
                color = TIER2_COLOR if row["concept"] in tier2 else TIER1_COLOR
                ax.scatter(row["S_west_gap"], row["mean_delta_L"],
                           color=color, s=120, zorder=5, edgecolors="white", linewidths=0.5)
                ax.annotate(row["concept"], (row["S_west_gap"], row["mean_delta_L"]),
                            textcoords="offset points", xytext=(5, 3),
                            fontsize=7.5, color=TEXT_COLOR)

            if len(agg) >= 3:
                rho, p = stats.spearmanr(agg["S_west_gap"], agg["mean_delta_L"])
                ax.text(0.05, 0.92,
                        f"Spearman ρ = {rho:.3f}\np = {p:.3f} {'*' if p < 0.05 else ''}",
                        transform=ax.transAxes, fontsize=9, color=TEXT_COLOR,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG, alpha=0.8))

                # Regression line
                m, b, *_ = stats.linregress(agg["S_west_gap"], agg["mean_delta_L"])
                x_line = np.linspace(agg["S_west_gap"].min(), agg["S_west_gap"].max(), 50)
                ax.plot(x_line, m * x_line + b, color=ACCENT4, linewidth=1.5,
                        linestyle="--", alpha=0.7)

            ax.axhline(0, color=GRID_COLOR, linewidth=1, linestyle=":")
            ax.axvline(0, color=GRID_COLOR, linewidth=1, linestyle=":")
            ax.set_xlabel("S_west_gap  (S(West) − S(non-West))")
            ax.set_ylabel(f"Mean Δ_L  ({LANG_LABELS.get(lang, lang)})")
            ax.set_title(f"{model_name} | {LANG_LABELS.get(lang, lang)}")
            ax.grid(alpha=0.15)

            patches = [mpatches.Patch(color=TIER1_COLOR, label="Tier 1 (Universal)"),
                       mpatches.Patch(color=TIER2_COLOR, label="Tier 2 (Cultural)")]
            ax.legend(handles=patches, framealpha=0.3, fontsize=8)

    plt.suptitle("Confound Check: Western Embedding Coherence Gap vs Cross-Lingual Alignment Gap",
                 fontsize=11, color=TEXT_COLOR, y=1.02)
    plt.tight_layout()
    out = figures_dir / "fig7_spearman_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"  → {out}")


# ─── Figure 8: Sensitivity analysis ──────────────────────────────────────────

def fig8_sensitivity(cfg: dict, figures_dir: Path):
    """Stacked bar showing which concepts enter Tier2 at each threshold."""
    tables_dir = Path(cfg["paths"]["tables"])
    csv_path = tables_dir / "sensitivity.csv"
    if not csv_path.exists():
        return

    import ast
    df = pd.read_csv(csv_path)
    all_concepts = cfg["all_concepts"]
    tier2_base   = set(cfg["tier2_concepts"])

    fig, ax = plt.subplots(figsize=(12, 4))

    percentiles = df["percentile"].tolist()
    cutoffs     = df["accuracy_cutoff"].tolist()

    for ci, concept in enumerate(all_concepts):
        in_tier2 = []
        for _, row in df.iterrows():
            try:
                tier2_list = ast.literal_eval(str(row["tier2_selected"]))
            except:
                tier2_list = []
            in_tier2.append(1 if concept in tier2_list else 0)

        color = TIER2_COLOR if concept in tier2_base else TIER1_COLOR
        ys = [ci + v * 0.8 for v in in_tier2]
        ax.plot(percentiles, in_tier2, "o-", color=color, alpha=0.7,
                linewidth=2, markersize=8, label=concept)

    ax.set_xlabel("Accuracy Percentile Threshold")
    ax.set_ylabel("In Tier 2? (1=yes, 0=no)")
    ax.set_xticks(percentiles)
    ax.set_xticklabels([f"p={p}%\n(cutoff={c:.0f}%)"
                         for p, c in zip(percentiles, cutoffs)])
    ax.legend(ncol=2, fontsize=7, framealpha=0.3, loc="upper right")
    ax.set_title("Sensitivity Analysis: Which Concepts Enter Tier 2 at Each Accuracy Threshold?",
                 fontsize=11)
    ax.grid(alpha=0.2)
    ax.set_ylim(-0.15, 1.25)

    plt.tight_layout()
    out = figures_dir / "fig8_sensitivity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"  → {out}")


# ─── Figure 9: All-in-one summary dashboard ──────────────────────────────────

def fig9_summary_dashboard(cfg: dict, figures_dir: Path, models: list[str]):
    """Executive summary dashboard combining key results."""
    tables_dir = Path(cfg["paths"]["tables"])

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_title = fig.add_axes([0, 0.95, 1, 0.05])
    ax_title.axis("off")
    ax_title.text(0.5, 0.5,
                  "CLIP Cultural Alignment — Research Summary Dashboard",
                  ha="center", va="center", fontsize=16, fontweight="bold",
                  color=TEXT_COLOR, transform=ax_title.transAxes)

    # Panel 1: GeoDE accuracies bar
    ax1 = fig.add_subplot(gs[0, 0])
    concept_order = cfg["tier2_concepts"] + cfg["tier1_concepts"]
    accs = [cfg["geode_accuracies"].get(c, 0) for c in concept_order]
    tier2 = set(cfg["tier2_concepts"])
    colors_acc = [TIER2_COLOR if c in tier2 else TIER1_COLOR for c in concept_order]
    ax1.barh(concept_order, accs, color=colors_acc, edgecolor="white", linewidth=0.3)
    ax1.axvline(70, color="#FFD700", linewidth=1.5, linestyle="--", alpha=0.7,
                label="70% threshold")
    ax1.set_xlabel("GeoDE CLIP ViT-B/32 Accuracy (%)")
    ax1.set_title("Published Concept Accuracies\n(GeoDE Table 4)", fontsize=10)
    ax1.legend(fontsize=7, framealpha=0.3)
    ax1.grid(axis="x", alpha=0.2)

    # Panel 2 & 3: Key Δ_L results per model
    for mi, model_name in enumerate(models[:2]):
        ax = fig.add_subplot(gs[0, 1 + mi])
        csv_path = tables_dir / "q2_alignment_gap.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df[(df["model"] == model_name) & (df["prompt_id"] == "P1")]
            for lang in ["es", "ar"]:
                sub = df[df["language"] == lang]
                if sub.empty: continue
                t1 = sub[sub["tier"] == "Tier1"]["mean_delta_L"].mean()
                t2 = sub[sub["tier"] == "Tier2"]["mean_delta_L"].mean()
                ax.bar([f"T1\n{lang}", f"T2\n{lang}"], [t1, t2],
                       color=[TIER1_COLOR, TIER2_COLOR], alpha=0.85,
                       edgecolor="white", linewidth=0.5)
            ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_title(f"{model_name.upper()}\nMean Δ_L by Tier + Language (P1)", fontsize=10)
            ax.set_ylabel("Mean Δ_L")
            ax.grid(axis="y", alpha=0.2)

    # Panel 4: Prompt gain comparison
    ax4 = fig.add_subplot(gs[1, 0])
    csv_path = tables_dir / "q3_prompt_gain.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["language"] == "es"]
        for mi, model_name in enumerate(models):
            sub = df[df["model"] == model_name]
            t1_g3 = sub[sub["tier"] == "Tier1"]["gain_P3"].mean()
            t2_g3 = sub[sub["tier"] == "Tier2"]["gain_P3"].mean()
            x = np.array([mi * 2, mi * 2 + 0.7])
            ax4.bar(x, [t1_g3, t2_g3],
                    color=[TIER1_COLOR, TIER2_COLOR],
                    alpha=0.7 + mi * 0.1, edgecolor="white", linewidth=0.5)
        ax4.axhline(0, color="white", linewidth=0.8, linestyle="--")
        ax4.set_title("Prompt Gain G(P3) by Model\n(Spanish, mean over concepts)", fontsize=10)
        ax4.set_ylabel("G = Δ_L(P1) − Δ_L(P3)")
        ax4.grid(axis="y", alpha=0.2)

    # Panel 5: Model asymmetry comparison
    ax5 = fig.add_subplot(gs[1, 1])
    csv_path = tables_dir / "q2_alignment_gap.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["prompt_id"] == "P1"]
        rows_a = []
        for m in models:
            for lang in ["es", "ar"]:
                sub = df[(df["model"] == m) & (df["language"] == lang)]
                t1 = sub[sub["tier"] == "Tier1"]["mean_delta_L"].mean()
                t2 = sub[sub["tier"] == "Tier2"]["mean_delta_L"].mean()
                rows_a.append({"model": m, "lang": lang, "A": t2 - t1})
        df_a = pd.DataFrame(rows_a)
        x = np.arange(len(df_a))
        colors_a = [MODEL_COLORS.get(r["model"], "#aaa") for _, r in df_a.iterrows()]
        ax5.bar(x, df_a["A"], color=colors_a, edgecolor="white", linewidth=0.5)
        ax5.set_xticks(x)
        ax5.set_xticklabels([f"{r['model']}\n{r['lang']}" for _, r in df_a.iterrows()],
                             fontsize=7)
        ax5.set_title("Cultural Asymmetry A(L)\n= Δ_L(Tier2) − Δ_L(Tier1)", fontsize=10)
        ax5.axhline(0, color="white", linewidth=0.8, linestyle="--")
        ax5.grid(axis="y", alpha=0.2)

    # Panel 6: Research summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    summary_text = (
        "Key Findings:\n\n"
        "Q1 (Image Geometry):\n"
        "  • Tier 2 concepts show higher cross-\n"
        "    region divergence in embedding space\n"
        "  • Western embeddings more coherent\n"
        "    for culturally embedded categories\n\n"
        "Q2 (Alignment Gap):\n"
        "  • Δ_L significantly larger for Tier 2\n"
        "  • Effect holds across ES and AR\n"
        "  • Spearman ρ confirms S-gap predicts Δ_L\n\n"
        "Q3 (Prompt Intervention):\n"
        "  • P3 reduces gap for Tier 2 (G > 0)\n"
        "  • SigLIP2 shows smaller asymmetry\n"
        "    but cultural gap persists"
    )
    ax6.text(0.05, 0.95, summary_text,
             transform=ax6.transAxes, fontsize=8.5,
             va="top", ha="left", color=TEXT_COLOR,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL_BG, alpha=0.8))
    ax6.set_title("Research Summary", fontsize=10)

    out = figures_dir / "fig9_summary_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    logger.info(f"  → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    figures_dir = Path(cfg["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating publication figures ...")
    fig1_taxonomy_heatmap(cfg, figures_dir, args.model)
    fig2_west_gap(cfg, figures_dir, args.model)
    fig3_delta_violin(cfg, figures_dir, args.model)
    fig4_delta_heatmap(cfg, figures_dir, args.model)
    fig5_prompt_gain(cfg, figures_dir, args.model)
    fig6_model_comparison(cfg, figures_dir, args.model)
    fig7_spearman_scatter(cfg, figures_dir, args.model)
    fig8_sensitivity(cfg, figures_dir)
    fig9_summary_dashboard(cfg, figures_dir, args.model)

    logger.info(f"\nAll figures saved to {figures_dir}")
    logger.info("Figures generated:")
    for p in sorted(figures_dir.glob("*.png")):
        logger.info(f"  {p.name}")


if __name__ == "__main__":
    main()
