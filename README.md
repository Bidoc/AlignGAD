# AlignGAD: MultiScale Zero-Shot Graph Anomaly Detection

AlignGAD detects anomalous nodes at multiple structural scales by combining the original neutralization-generation mechanism with super-graph clustering, enabling zero-shot deployment across heterogeneous graph domains without fine-tuning.

## Method Overview

```
Input graph G = (X, A)
        │
        ▼
┌─────────────────────────────────────────┐
│ MODULE 1: Global Information Unification│
│ • Feature dim alignment via SVD (→ d')  │
│ • Asymmetric Spectral Normalization     │
└─────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ PASS 1 (n nodes, original graph)         │
│ • Module 4: Score via cosine discrepancy │
│ • Module 2 + 3: Cluster → super-graph    │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ PASS 2 (n/2 super-nodes)                 │
│ • Module 4: Score on super-graph         │
│ • Module 2 + 3: Cluster → super-graph    │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│ PASS 3 (n/4 super-nodes)                 │
│ • Module 4: Score on super-graph         │
└──────────────────────────────────────────┘
        │
        ▼
   Aggregate scores across passes
```

### Module Descriptions

| Module | Purpose | Key Components |
|--------|---------|----------------|
| **1. Global Unification** | Cross-domain feature alignment | SVD (top-`d'`), Asymmetric Spectral Normalization with band-specific `α` |
| **2. RFF Clustering** | Differentiable hierarchical clustering | Light smoothing, Random Fourier Features, T-SVD, STE K-means |
| **3. Super-graph Construction** | Build coarser graph for next pass | Binary off-diagonal adjacency, binary self-loop from original edges |
| **4. Scoring Module** | Anomaly score via reconstruction | 3-layer GCN encoder + 2-layer GCN decoder, **shared across passes** |

## Project Structure

```
ms-zerogad/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml                 # All hyperparameters
├── ms_zerogad/
│   ├── modules/
│   │   ├── unification.py           # Module 1
│   │   ├── clustering.py            # Module 2
│   │   ├── supergraph.py            # Module 3
│   │   └── scoring.py               # Module 4
│   ├── pipeline/
│   │   ├── multipass.py             # Multi-pass orchestration
│   │   └── aggregation.py           # Score aggregation strategies
│   ├── data/
│   │   ├── loader.py                # .mat dataset loader
│   │   └── preprocessing.py
│   ├── training/
│   │   ├── train.py                 # Training loop with best-checkpoint tracking
│   │   └── losses.py
│   ├── evaluation/
│   │   └── metrics.py               # AUROC, AUPRC
│   └── utils/
│       ├── stable_ops.py            # STE, numerical utilities
│       └── tracking.py              # Membership tracking across passes
├── notebooks/
│   ├── 00_setup.ipynb
│   ├── 01_test_module1.ipynb        # Individual module unit tests
│   ├── ...                          # 02–06 for other modules
│   ├── 07_train.ipynb               # Main training notebook
│   ├── 08_evaluate.ipynb            # Per-dataset evaluation
│   └── 09_aggregation_analysis.ipynb # Aggregation strategy comparison
├── checkpoints/                     # Saved model checkpoints
├── results/
│   └── per_pass_scores/             # Cached scores per dataset for fast aggregation testing
└── data/
    └── raw/                         # .mat files for source and target graphs
```

## Installation

```bash
git clone <repository_url>
cd ms-zerogad
pip install -r requirements.txt
```

Tested on:
- Python 3.10+
- PyTorch 2.0+
- Google Colab (A100 GPU)

Note: If you run it on Colab, upload this repo to your Google Drive , put it in a folder name Project_GraphML ( the structure should be Project_GraphML/ms-zerogad/.... ) and then mount to the drive as it shown in code 


## Datasets

The project follows the same dataset split as Zero-GAD:

**Source domains (training)**
- Facebook (n=1086)
- Flickr (n=7575)
- BlogCatalog (n=5196)
- ACM (n=16484)

**Target domains (zero-shot evaluation)**
- Citation: Cora (n=2708), Citeseer (n=3327), Pubmed (n=19717)
- Co-purchase: Photo (n=7650), Amazon (n=1418)
- Co-authorship: CS (n=18333)
- Reviews: YelpChi (n=23831)
- Social: Reddit (n=10984)

All datasets are in `.mat` format and should be placed in `data/raw/`.

## Usage

### 1. Setup

Adjust paths in `notebooks/00_setup.ipynb` to point to your data directory. If running on Colab, mount Google Drive accordingly.
Training and evaluate can also be done in this notebbook by running the last 3 cells


### 2. Train

Open `notebooks/07_train.ipynb` and run all cells. Training uses multi-domain collaborative learning: one source graph is sampled per training step.

Default training:
- 100 epochs
- Best-checkpoint saving based on validation AUROC (Cora by default)
- Per-pass scores cached after best checkpoint

### 3. Evaluate

`notebooks/08_evaluate.ipynb` loads the best checkpoint and runs inference on all 8 target datasets, reporting per-pass and aggregated AUROC.

### 4. Aggregation Analysis

`notebooks/09_aggregation_analysis.ipynb` uses cached per-pass scores to test 17 different aggregation strategies in <1 second per strategy (no re-inference required).

## Configuration

All hyperparameters live in `configs/default.yaml`. Below are the key parameters with current best values:

### Module 1: Spectral Normalization

```yaml
module1:
  d_prime: 8                  # Target dimension after SVD
  band_low: 0.5               # Low/mid band boundary
  band_high: 1.5              # Mid/high band boundary
  adaptive_bands: false       # Fixed bands; adaptive performs worse
  alpha_low: 1.0              # Full normalization for low frequencies
  alpha_mid: 0.7              # Light normalization for mid frequencies
  alpha_high: 0.5             # Lightest normalization for high frequencies
```

### Module 2: RFF Clustering

```yaml
module2:
  k_smoothing: 1              # Smoothing order before RFF
  sigma: 1.0                  # RFF kernel bandwidth
  D_rff: 50                   # Final RFF dim = 2 × D_rff = 100
  d_svd: 32                   # T-SVD reduced dimension
  tau: 0.5                    # STE temperature
  kmeans_max_iter: 20
```

### Module 4: Scoring

```yaml
module4:
  d_hidden: 64                # GCN hidden dim
  d_latent: 32                # Latent dim (NOT a compression bottleneck)
  num_encoder_layers: 3
  num_decoder_layers: 2
  dropout: 0.0
```

### Multi-pass and Loss

```yaml
multipass:
  cluster_ratios: [0.5, 0.25, 0.125]   # Pass 1: n/2, Pass 2: n/4, Pass 3: n/8

loss:
  alpha: 2.0                  # Sparsity-aware reconstruction exponent (Zero-GAD)
  beta: 0.5                   # Neutralization loss weight (Zero-GAD)
  loss_weights: [1.0, 1.0, 1.0]   # Equal weighting across passes (default)
```

### Training

```yaml
training:
  num_epochs: 100
  lr: 5e-4
  weight_decay: 5e-4
  grad_clip_norm: 1.0         # Important for STE stability
  eval_interval: 5
  seed: 42
```

## Experimental Results (Current Best)

Mean AUROC across 8 target datasets with best configuration (asymmetric fixed bands):

| Dataset | P1 only | P2 only | P3 only | MEAN_all | MAX_all | Best Strategy |
|---------|---------|---------|---------|----------|---------|---------------|
| Cora | 0.609 | 0.564 | 0.529 | 0.582 | 0.565 | P1_only (0.609) |
| Citeseer | 0.552 | 0.502 | 0.558 | 0.551 | 0.546 | P3_only (0.558) |
| Pubmed | 0.595 | 0.556 | 0.540 | 0.575 | 0.571 | P1_only (0.595) |
| Photo | 0.645 | 0.550 | 0.506 | 0.588 | 0.595 | P1_only (0.645) |
| CS | 0.519 | 0.527 | 0.526 | 0.528 | 0.528 | MEAN_P2P3 (0.529) |
| Amazon | 0.442 | 0.450 | 0.393 | 0.424 | 0.420 | P2_only (0.450) |
| Reddit | 0.480 | 0.461 | 0.449 | 0.459 | 0.455 | P1_only (0.480) |
| YelpChi | 0.493 | 0.631 | 0.628 | 0.593 | 0.556 | P2_only (0.631) |
| **Average** | **0.5546** | **0.5564** | **0.5409** | **0.5637** | **0.5564** | — |

Detailed per-strategy heatmap available in `results/aggregation_heatmap.png`.

## Related Work

- **Zero-GAD** (Zheng et al., MM '25): The base framework. AlignGAD inherits the neutralization-generation autoencoder mechanism and adds hierarchical multi-pass scoring.
- **SASE** (CIKM '24): Source for the RFF-based scalable spectral clustering used in Module 2.


## License

To be determined. Code currently intended for research purposes only.

## Acknowledgments

This project builds on the Zero-GAD framework. The RFF-based scalable clustering approach is inspired by SASE. Datasets follow the same split as the Zero-GAD paper for direct comparability.
