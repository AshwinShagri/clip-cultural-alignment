# CLIP Cultural Alignment: Does Image Encoder Geometry Predict Cross-Lingual Failures?

**Saarland University | High-Level Computer Vision (HLCV) SS26 Project**

**Authors:** Ashwin Kumar, Harini Raj, Pranav Kushare  
**Dataset:** GeoDE (3,000 images, balanced across 10 concepts and 6 world regions)  
**Models Evaluated:** OpenCLIP ViT-B/32, SigLIP 2 ViT-B/16  

---

## 1. Project Overview & Objectives

This project provides a rigorous statistical evaluation of zero-shot vision-language models under geographical distribution shifts and cross-lingual text representations. We analyze the relationship between visual embedding geometry, multilingual text alignment, and inference-time prompting mitigations across three core Research Questions:

*   **Q1 (Visual Geometry):** Does CLIP's image encoder produce higher inter-region feature divergence for culturally embedded concepts (Tier 2) than universal ones (Tier 1)?
*   **Q2 (Cross-Lingual Alignment Gap):** Is the alignment gap $\Delta_L = \cos(v, t_{EN}) - \cos(v, t_L)$ larger for culturally embedded concepts (Tier 2) than universal ones (Tier 1)?
*   **Q3 (Prompt Intervention):** Can explicitly culturally-aware prompt templates ($P3$) reduce the cross-lingual alignment gap relative to a neutral baseline ($P1$)?

---

## 2. Core Codebase Architecture

The project is structured modularly. Here is the line-by-line function and script architecture:

### Orchestration & Configuration
*   [`config.yaml`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/config.yaml): Centralized configuration. Defines model checkpoints (`CLIP-ViT-B-32-laion2B-s34B-b79K`, `siglip2-base-patch16-224`), dataset target sizes, concept tiers, prompt templates, region translations, and directories.
*   [`requirements.txt`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/requirements.txt): Lists all package dependencies.
*   [`run_all.py`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/run_all.py): Master orchestrator. Manages execution phases: `data` ➔ `embed` ➔ `taxonomy` ➔ `analysis` ➔ `qualitative` ➔ `figures`.

### Core Source Scripts (`src/`)
*   [`src/utils.py`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/src/utils.py):
    *   `load_config(path)`: Loads and parses `config.yaml`.
    *   `build_prompts(cfg)`: Dynamically generates prompt dictionaries for all languages, formatting concept translation nouns and articles (Spanish/Arabic) and cleaning dangling prepositions (like "used in" or "in") for empty-region global placeholders.
    *   `log_result(...)`: Writes statistical test results (p-values, statistics, sample sizes) to a centralized CSV history log.
*   [`src/fetch_data.py`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/src/fetch_data.py):
    *   Streams the GeoDE dataset from Hugging Face (`MLap/GeoDE`).
    *   Applies predicate filtering on object categories and target regions.
    *   Saves exactly **3,000 images** (50 images × 10 classes × 6 regions) locally under `data/raw/` in a perfectly balanced grid, writing `data/raw/metadata.csv`.
*   [`src/embed.py`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/src/embed.py):
    *   `get_image_features(...)` / `get_text_features(...)`: Computes and caches visual and textual embeddings using CUDA.
    *   **Hugging Face API Adaptation:** Automatically detects and unpacks `BaseModelOutputWithPooling` wrappers to extract the raw `.pooler_output` tensor.
    *   **SigLIP Calibration:** Enforces `max_length=64` fixed-length padding for SigLIP models to match their pre-training configuration, ensuring optimal embedding geometry.
*   [`src/taxonomy.py`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/src/taxonomy.py):
    *   Computes regional similarity matrices $S(c,r)$ and cross-region visual divergence.
    *   Validates the category division: Tier 1 (Universal: `light switch`, `bus`, `chair`, `car`, `bag`) vs. Tier 2 (Embedded: `dustbin`, `medicine`, `cleaning equipment`, `spices`, `house`).
    *   Runs threshold sensitivity analyses across multiple accuracy percentiles.
*   [`src/metrics.py`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/src/metrics.py):
    *   `compute_S_matrix(...)`: Measures mean pairwise cosine similarities within (concept, region) groups.
    *   `compute_cross_region_divergence(...)`: Computes $1 - \text{mean pairwise cosine similarity}$ across different regions.
    *   `compute_delta_L(...)`: Calculates $\Delta_L = \cos(v, t_{EN}) - \cos(v, t_L)$ per image, dynamically matching region-specific visual samples to their corresponding regional text prompts.
    *   `compute_raw_cosines(...)`: Tracks the raw English and native similarities (`cos_en`, `cos_L`) per image to enable regression confound control.
    *   `compute_prompt_gain(...)`: Measures prompt gain $G(P3) = \Delta_L(P1) - \Delta_L(P3)$.
*   [`src/analysis.py`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/src/analysis.py):
    *   Runs Welch's $t$-tests and paired $t$-tests to evaluate differences in $\Delta_L$ between Tier 1 and Tier 2.
    *   Calculates Spearman rank correlations between visual divergence and alignment gaps.
    *   **OLS Regression Confound Control:** Fits a manual Ordinary Least Squares (OLS) regression in NumPy:
        $$\Delta_L = \beta_0 + \beta_1 \cdot \text{is\_Tier2} + \beta_2 \cdot \cos(v, t_{EN}) + \epsilon$$
        to test whether concept tiers predict alignment gaps after controlling for base model confidence.
*   [`src/qualitative.py`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/src/qualitative.py):
    *   Projects high-dimensional image embeddings into 2D spaces using UMAP (colored by region, concept, and tier).
    *   Generates nearest-neighbor visual grids for query images using region-specific P3 prompt anchors.
*   [`src/visualize.py`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/src/visualize.py):
    *   Translates outputs into publication-grade matplotlib figures (violin plots of gaps, heatmaps, Spearman scatters, and dashboards).
*   [`src/attention_viz.py`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/src/attention_viz.py):
    *   Extracts self-attention weight tensors from all visual transformer layers.
    *   Computes **ViT Attention Rollouts** (Abnar & Zuidema, 2020). Dynamically detects model architectures (CLIP models with CLS tokens vs. SigLIP models without CLS tokens) to reshape attention matrices and saves overlay heatmaps.

---

## 3. Final Scientific Findings

Our empirical evaluation on GeoDE reveals a highly nuanced, statistically rigorous set of results that both validate and clarify current literature:

### Q1: Visual Geometry (Metric 1)
*   **Result:** Welch's $t$-test shows **no statistically significant difference** in raw visual space regional divergence between Tier 1 and Tier 2 ($p \approx 0.25$ for OpenCLIP, $p \approx 0.53$ for SigLIP 2).
*   **Analysis:** While the raw means show the expected trend (Tier 2 divergence is higher: `0.466` vs. `0.410` for OpenCLIP), the test lacks significance due to concept-level sample size limits ($n=5$ per tier). This indicates that raw image feature geometry alone does not fully predict cultural bias.

### Q2: Cross-Lingual Alignment Gap (Metric 2)
*   **Result:** Highly statistically significant differences in $\Delta_L$ between tiers ($p < 0.001$, Welch's $t$-test p-values reach as low as $10^{-232}$).
*   **Reversal Explanations:**
    *   **OpenCLIP Arabic:** Because OpenCLIP has zero training on Arabic script, Arabic similarities collapse to noise ($\approx 0.0$). The gap reduces to $\Delta_L \approx \cos(v, t_{EN})$. Since English CLIP is highly confident on easy Tier 1 concepts, the gap for Tier 1 is mathematically larger (a statistical artifact).
    *   **SigLIP 2:** Due to explicit multilingual pre-training, the alignment gap on Tier 2 embedded concepts is **closed to zero** (and even becomes slightly negative, meaning native text matches non-Western images better than English). Since universal Tier 1 concepts still retain a small positive gap, the Tier 1 gap ends up being larger.
*   **Confound Control:** Our OLS regression shows that for OpenCLIP Spanish, the Tier 2 coefficient ($\beta_1 = \mathbf{0.0521}$) remains **highly significant ($p \approx 1.41 \times 10^{-7}$, $t = 6.01$)** even after controlling for baseline English similarity $\cos(v, t_{EN})$. This proves that the alignment gap for Spanish on embedded objects is a real cultural bias effect, not a baseline confidence artifact.

### Q3: Prompt Intervention (Metric 3)
*   **Result:** Culturally-aware prompts ($P3$) show positive gains (gap reduction) for both tiers under multilingual **SigLIP 2 Spanish** (Tier 1 Gain = **+0.0294**, Tier 2 Gain = **+0.0124**). 
*   **Analysis:** However, for Arabic and monolingual OpenCLIP, cultural prompting yields negative gains for Tier 2 embedded objects. This indicates that simple inference-time prompting cannot bypass deep representation deficits in low-resource scripts; training-time alignment is required.

---

## 4. How to Set up and Run

### Installation
1. Install dependencies listed in requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure you have a CUDA-compatible GPU environment set up.

### Run the Pipeline
To run the entire pipeline end-to-end (downloads data, embeds, calculates metrics, runs stats, visualizes nearest-neighbors, and saves dashboard figures):
```bash
python run_all.py --phase all
```

### Run Individual Phases
You can also run individual steps using the `--phase` argument:
```bash
# Rerun embeddings for SigLIP 2 only
python run_all.py --phase embed --model siglip2

# Run taxonomy splitting and sensitivity
python run_all.py --phase taxonomy

# Run statistical tests (Q1/Q2/Q3 and OLS confound checks)
python run_all.py --phase analysis

# Rerun UMAPs and nearest-neighbor grids
python run_all.py --phase qualitative

# Re-generate publication plots
python run_all.py --phase figures
```

### Generate ViT Attention Rollout Maps
To extract attention rollouts and save image overlays:
```bash
# Extract attention maps for spices in Africa using OpenCLIP
python src/attention_viz.py --model openclip --concept "spices" --region Africa --n-images 2

# Extract attention maps for cleaning equipment in East Asia using SigLIP 2
python src/attention_viz.py --model siglip2 --concept "cleaning equipment" --region EastAsia --n-images 2
```
Attention rollout overlay maps are saved directly to `results/figures/`.

---

## 5. Interactive Demo
To run the interactive project showcase containing all tables, regression outputs, and attention rollout visualizations:
1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open [`notebooks/demo.ipynb`](file:///d:/UDS/SEM2/HLCV/PROJECT/clip-cultural-alignment/notebooks/demo.ipynb) and run all cells.
