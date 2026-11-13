# ST-GRAD: Reproduction & Explainability Research Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.14.0](https://img.shields.io/badge/pytorch-2.14.0%2Bcu132-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-42%2F42%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Reproducible research and explainability analysis of:  
> **"ST-GRAD: Multi-view cross-attention and bidirectional diffusion for stock trend prediction"**  
> *Yu Fu, Zhengming Huang, Bin Yu (2026)*  
> *Expert Systems With Applications*, 330, 133046. [DOI: 10.1016/j.eswa.2026.133046](https://doi.org/10.1016/j.eswa.2026.133046)

---

## 1. Project Overview

This repository provides an end-to-end, reproducible implementation of the ST-GRAD framework with two strictly isolated paths:
- **Track A (`src/repo_compat/`):** Faithfully reproduces the author's anonymous repository implementation, verifying supplied checkpoints (`best_model_ours.pth`) on the historical validation set.
- **Track B (`src/paper_aligned/`):** Implements the paper's theoretical equations (1)–(34) with zero lookahead bias, raw closing-price label generation, validity masking, and clean validation/blind stress test isolation.
- **Explainability Suite (`src/explainability/`):** Implements post-hoc Integrated Gradients (IG), counterfactual MVCA view interventions, graph channel disentanglement (static vs adaptive), and quantitative faithfulness/stability testing.

---

## 2. Environment Setup

Tested with **Python 3.11.9** and **PyTorch 2.14.0+cu132** (CUDA 13.2) on Windows 11 with an NVIDIA GeForce RTX 4060 Laptop GPU. Dependencies are pinned in [`requirements.txt`](requirements.txt).

```powershell
# 1. Create and activate a virtual environment (Python 3.11)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

PyPI's default Windows `torch` wheel is CPU-only. For GPU training, install the CUDA build first, then the rest:

```powershell
pip install --index-url https://download.pytorch.org/whl/cu132 "torch==2.14.0"
pip install -r requirements.txt
```

All commands below work on CPU-only machines as well (slower). Run every command from the project root.

---

## 3. Quickstart & Command Reference

### A. First-time data preparation (Track A only, required once per clone)
`ST-GRAD-Stock-Prediction-6E7D/data/processed/` is **not** version-controlled (it holds ~153 MB of derived tensors). Rebuild it from the tracked raw CSVs before running Track A or the full test suite:

```powershell
python src/run_repo_compat.py --mode graph
python src/run_repo_compat.py --mode data
```

* `--mode graph` writes `data/processed/normalized_adj_matrix.npy` (≈4 s) and refreshes `data/raw/stock_order.npy` with identical content.
* `--mode data` writes `data/processed/stock_dataset.npz` (≈10 s).

Track B does not need this step: `src/paper_aligned/dataset.py` rebuilds its own dataset from `data/raw/` on first use.

### B. Run Complete Unit Test Suite (42/42 passing)
```bash
python -m unittest discover -s src -p "test_*.py"
```
The test suite covers reproduction benchmarks, mathematical equations, data leakage audits, Gauss-Legendre IG completeness, MVCA view interventions, decoupled graph ablations, and regime benchmark diagnostics.

### C. Unified CLI: Instant Model Explanations & Diagnostics
The project provides a unified CLI orchestrator (`src/cli.py`):

```bash
# 1. Explain a stock prediction (terminal output)
python src/cli.py explain --ticker HPE --split val_clean --index 0

# 2. Export standardized Markdown and responsive HTML report cards
python src/cli.py explain --ticker HPE --split val_clean --index 0 \
  --save_card artifacts/explainability/hpe_card.md \
  --save_html artifacts/explainability/hpe_card.html \
  --save_json artifacts/explainability/hpe_case.json

# 3. Run Phase 5 Dual-Track & Regime Shift Benchmark (Hypothesis H6)
python src/cli.py benchmark-regimes --samples-clean 25 --samples-stress 25

# 4. Run Controlled Hypothesis Suite (H1-H4: Faithfulness, Views, Graphs, Stability)
python src/cli.py evaluate-explanations --track paper_aligned --samples 15

# 5. Evaluate Predictive Metrics on Clean Val or Stress Split
python src/cli.py evaluate --track paper_aligned --split clean_val
python src/cli.py evaluate --track repo_compat --split clean_val

# 6. Audit Data Integrity and Validity Masks
python src/cli.py audit-data
```

### D. Track A: Evaluate Author Checkpoint
```bash
python src/run_repo_compat.py
```
*Expected Output:* ACC: 0.8317, PRE: 0.7651, REC: 0.8911, F1: 0.8233, AUC: 0.9041 (matches Paper Table 8 within standard deviations).

### E. Track B: Full 5-Seed Baseline Training & Reproduction
```bash
python src/paper_aligned/train.py --epochs 50 --batch_size 32 --seeds 42 43 44 45 46
```

---

## 4. Key Performance Results

| Model / Split | ACC | PRE | REC | F1 | AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Paper Table 8 (Published)** | $0.8313 \pm 0.0210$ | $0.7814 \pm 0.0468$ | $0.8799 \pm 0.0350$ | $0.8215 \pm 0.0280$ | $0.9046 \pm 0.0250$ |
| **Track A: Author Checkpoint (`best_model_ours.pth`)** | 0.8317 | 0.7651 | 0.8911 | 0.8233 | 0.9041 |
| **Track B: Clean Validation (Ours, 308 samples)** | **$0.8975 \pm 0.0003$** | **$0.8881 \pm 0.0029$** | **$0.9109 \pm 0.0043$** | **$0.8994 \pm 0.0007$** | **$0.9636 \pm 0.0003$** |
| **Track B: Blind Stress Test (Ours, 61 samples)** | $0.8840 \pm 0.0032$ | $0.8633 \pm 0.0091$ | $0.8828 \pm 0.0080$ | $0.8729 \pm 0.0029$ | $0.9577 \pm 0.0008$ |
| **Track B: Total Out-of-Sample (Ours, 369 samples)** | $0.8953 \pm 0.0005$ | $0.8844 \pm 0.0037$ | $0.9067 \pm 0.0048$ | $0.8954 \pm 0.0007$ | $0.9627 \pm 0.0003$ |

---

## 5. Explainability Hypothesis Evaluation Summary

| Hypothesis | Evaluation Protocol | Quantitative Metric | Finding / Empirical Result | Verdict |
|---|---|---|---|:---:|
| **H1: Attribution Faithfulness** | Feature deletion vs random deletion curves (AOPC) | Faithfulness Ratio: **$4.13\times$** ($p = 0.0033$) | Top IG features degrade confidence $4\times$ faster than random | **CONFIRMED** |
| **H1: Model Randomization** | Cascading top-layer weight randomization (Adebayo) | Post-randomization Spearman: $\rho = 0.594$ | Partial lower-layer signal retained due to residual connections | **QUALIFIED** |
| **H2: MVCA View Distinguishability**| Counterfactual view ablation ($\Psi_{\text{ST}}, \Psi_{\text{DS}}, \Psi_{\text{DT}}$) | ANOVA $F$-Statistic: **10.25** ($p = 0.00024$) | Views provide non-redundant, distinct predictive contributions | **CONFIRMED** |
| **H3: Graph Edge Specificity** | Targeted peer edge ablation vs matched random controls | Edge Influence Ratio: **$5.24\times$** ($p = 0.0041$) | Learned edges drive $5\times$ greater logit impact than matched controls | **CONFIRMED** |
| **H4: Cross-Seed Stability** | Attribution concordance across Seeds 42–46 | Pairwise Spearman Rank Correlation: $\bar{\rho} = \mathbf{0.7132}$ | Feature importance rankings are stable across training initializations | **CONFIRMED** |
| **H4: Perturbation Robustness** | Label-preserving Gaussian noise ($\sigma = 0.02$) | Mean Cosine Similarity: **$0.9998$**, Spearman: **$0.996$** | Explanations are highly resilient to minor input fluctuations | **CONFIRMED** |
| **H6: Regime Shift Invariance** | Clean Validation (308 dates) vs Blind Stress (61 dates) | Feature Rank Spearman: $\rho = \mathbf{0.9930}$, Shift: $+16.6\%$ ST view | Feature hierarchy is invariant; spatial/temporal reliance intensifies | **QUALIFIED** |

---

## 6. Directory Structure

```text
st_grad/
├── artifacts/
│   ├── explainability/              # JSON reports, case records, SVGs, HTML dashboards
│   │   ├── dual_track_regime_benchmark.json
│   │   ├── regime_benchmark_dashboard.html     # Interactive responsive dashboard
│   │   ├── feature_importance_regime_shift.svg # Native vector graphic
│   │   ├── view_hierarchy_regime_shift.svg     # Native vector graphic
│   │   ├── temporal_decay_regime_shift.svg     # Native vector graphic
│   │   ├── graph_channel_regime_shift.svg      # Native vector graphic
│   │   └── hpe_card.html / hpe_card.md
│   └── paper_aligned/               # Checkpoints, predictions, and datasets
├── docs/
│   ├── data_leakage_audit.md        # Leakage tests and data integrity audit
│   ├── data_protocol.md             # 24 technical indicators, labels, and splits
│   ├── experiment_protocol.md       # Exact protocols for training and explainability
│   ├── explainability_methodology.md# Mathematical specification of explainability suite
│   ├── explainability_regime_benchmark.md # Phase 5 Dual-track & regime report
│   ├── limitations.md               # Scientific caveats, leakage analysis, and diagnostics
│   ├── model_architecture.md        # Equations (1)-(34) and parameter breakdown
│   ├── paper_specification.md       # Full formal paper transcription and tensor shapes
│   ├── paper_vs_code.md             # Line-by-line comparison between paper and repository
│   ├── reproduction_benchmark.md    # Multi-seed performance benchmark vs Table 8
│   ├── reproduction_deviations.md   # Discrepancies catalog and resolutions
│   ├── results.md                   # Complete empirical results compilation
│   └── runbook.md                   # Operational command guide and troubleshooting
├── src/
│   ├── cli.py                       # Unified command-line interface orchestrator
│   ├── explain.py                   # High-level instance explanation runner
│   ├── common/                      # Shared metrics supporting validity masks
│   ├── explainability/
│   │   ├── adapter.py               # Polymorphic adapter across Track A and Track B
│   │   ├── attribution.py           # Gauss-Legendre quadrature Integrated Gradients
│   │   ├── explainer.py             # High-level STGradExplainer engine
│   │   ├── faithfulness.py          # Directional AOPC deletion/insertion & randomization
│   │   ├── interventions.py         # MVCA view hooks & decoupled graph edge/node ablations
│   │   ├── plots.py                 # Zero-dependency SVG and HTML visual generator
│   │   ├── regime_benchmark.py      # Dual-track cohort & regime shift evaluator (H6)
│   │   ├── report.py                # Schema validator, JSON/NPZ, and HTML/MD card exporters
│   │   └── stability.py             # Multi-seed stability & noise perturbation gating
│   ├── paper_aligned/               # Track B: theoretical model, data, features, training
│   ├── repo_compat/                 # Track A: repository-faithful reproduction
│   ├── run_repo_compat.py           # Track A runner
│   ├── test_data_leakage.py         # Data integrity audit suite (11 tests)
│   ├── test_explainability.py       # Explainability unit tests (13 tests)
│   ├── test_paper_aligned.py        # Track B model and gradient tests (6 tests)
│   ├── test_paper_spec.py           # Mathematical equation tests (9 tests)
│   └── test_repo_compat.py          # Track A checkpoint loading tests (3 tests)
├── ST-GRAD-Stock-Prediction-6E7D/   # Immutable original author repository
├── instructions.txt                 # Multi-phase project instructions and constraints
├── requirements.txt                 # Pinned Python dependencies
└── README.md                        # Master documentation
```

---

## 7. Citation

If using this implementation or benchmark in your research, cite the original paper:

```bibtex
@article{fu2026stgrad,
  title={ST-GRAD: Multi-view cross-attention and bidirectional diffusion for stock trend prediction},
  author={Fu, Yu and Huang, Zhengming and Yu, Bin},
  journal={Expert Systems with Applications},
  volume={330},
  pages={133046},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.eswa.2026.133046}
}
```
