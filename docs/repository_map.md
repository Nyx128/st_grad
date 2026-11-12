# Repository Map: ST-GRAD Reconnaissance (Phase 0)

**Target Repository:** `ST-GRAD-Stock-Prediction-6E7D`  
**Reference Paper:** Fu, Huang, & Yu (2026), *ST-GRAD: Multi-view cross-attention and bidirectional diffusion for stock trend prediction*, *Expert Systems With Applications* 330, 133046. [DOI: 10.1016/j.eswa.2026.133046](https://doi.org/10.1016/j.eswa.2026.133046).  
**Source Archive Hash (SHA-256):** `690ed2f727a588bb71b6f74c863396a2a2dd129d0c99de8441ec04889a012434`  
**Inspected Date:** October 2026  
**Execution Environment:** Windows 11, Python 3.11.9 (project-local `.venv`), PyTorch 2.14.0+cu132, NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM).

---

## 1. Directory Structure

```text
st_grad/
├── 1-s2.0-S0957417426019573-main.pdf       # Authoritative paper PDF
├── ST_GRAD_PAPER.md                        # Markdown transcription of the research paper
├── ST_GRAD_REPOSITORY_AUDIT.md             # Baseline repository audit
├── ST_GRAD_REPRODUCTION_PLAN.md            # Detailed reproduction specification
├── ST_GRAD_EXPLAINABILITY_RESEARCH_PLAN.md # Explainability research roadmap
├── instructions.txt                        # Research and implementation directives
├── docs/                                   # Documentation artifacts
│   ├── repository_map.md                   # This document
│   ├── paper_vs_code.md                    # Systematic paper vs code discrepancy table
│   ├── reproduction_deviations.md          # Controlled deviations & track isolation
│   └── runbook.md                          # Step-by-step reproduction runbook
├── src/                                    # All agent-written code and tooling
│   ├── run_repo_compat.py                  # Non-destructive repo compatibility runner
│   └── inspect_pdf.py                      # PDF inspection utility
└── ST-GRAD-Stock-Prediction-6E7D/          # Bundled anonymous repository (immutable baseline)
    ├── .gitignore
    ├── README.md                           # Repository documentation
    ├── requirements.txt                    # Pinned dependencies (Pandas 2.0.3, Torch 2.4.1)
    ├── main.py                             # Pipeline orchestrator (modes: data, graph, train, predict, all)
    ├── train.py                            # Model training, evaluation, metrics computation
    ├── data/
    │   ├── raw/
    │   │   ├── BasicMaterials/             # 10 ticker CSVs (AXTA, MP, RGLD, SSD, etc.)
    │   │   ├── Energy/                     # 10 ticker CSVs (CTRA, DKL, KNTK, MTDR, etc.)
    │   │   ├── Healthcare/                 # 10 ticker CSVs (BIIB, CNC, GMAB, ICLR, etc.)
    │   │   ├── Industrials/                # 10 ticker CSVs (ABM, ALG, HRI, HUBG, etc.)
    │   │   ├── Technology/                 # 10 ticker CSVs (AMKR, ARW, HPE, PTC, etc.)
    │   │   ├── all_close.csv               # Aggregated closing prices (unused by source)
    │   │   ├── cluster_results.csv         # 50 stocks with Hierarchical & Trend cluster labels
    │   │   └── stock_order.npy             # Ordered array of 50 stock tickers
    │   └── processed/
    │       ├── normalized_adj_matrix.npy   # Precomputed 50x50 symmetric normalized graph
    │       └── stock_dataset.npz           # Processed tensor dataset (train/val splits & norms)
    ├── img/                                # Architectural & experimental result PDF figures
    ├── models/
    │   ├── __init__.py
    │   ├── st_grad.py                      # Core architecture: MVCA, BDGC, STBlock, ST_GRAD
    │   └── saved/
    │       ├── best_model_ours.pth         # Provided best checkpoint (author-reported weights)
    │       └── st_grad_best_model.pth      # Default checkpoint saved during training
    └── utils/
        ├── __init__.py
        ├── data_processor.py               # Feature extraction, label generation, data splits
        └── graph_builder.py                # Graph adjacency generator from cluster_results.csv
```

---

## 2. File Inventory & Module Analysis

| File Path | Size (Bytes) | Lines | Primary Responsibility | Key Inputs / Outputs |
|---|---:|---:|---|---|
| `main.py` | 4,245 | 110 | Unified CLI orchestrator. Parses arguments, handles execution modes (`graph`, `data`, `train`, `predict`, `all`). | Calls `build_adjacency_matrix`, `process_and_save_data`, `train_model`, `test_model`. |
| `train.py` | 5,667 | 141 | Training loop, evaluation, checkpoint saving, metric calculation (Acc, Pre, Rec, F1, AUC). | Inputs: `stock_dataset.npz`, `normalized_adj_matrix.npy`. Outputs: `models/saved/st_grad_best_model.pth`. |
| `models/st_grad.py` | 8,662 | 209 | Core PyTorch model definitions: `AdaptiveAdjacency`, `UDConv`, `CrossAttentionView`, `TemporalAttention`, `DiffusionConvolution`, `STBlock`, `ST_GRAD`. | Input: `[B, 24, 30, 50]`, `A: [50, 50]`. Output: `[B, 50]` logits. Total parameters: 216,429. |
| `utils/data_processor.py` | 7,471 | 163 | Technical indicator extraction (24 features), target label construction, chronological split (70/30), normalization. | Inputs: Raw CSV files in `data/raw/*/*.csv`. Output: `data/processed/stock_dataset.npz`. |
| `utils/graph_builder.py` | 1,250 | 35 | Parses `data/raw/cluster_results.csv` and builds symmetrically normalized prior graph. | Inputs: `cluster_results.csv`, `stock_order.npy`. Output: `normalized_adj_matrix.npy`. |
| `data/raw/*/*.csv` | ~45-55 KB each | 1,256 | 50 stock OHLCV historical time series (1,255 daily observations, 2020-08-19 to 2025-08-18). | Columns: `date`, `open`, `high`, `low`, `close`, `volume`. |
| `data/raw/cluster_results.csv` | 460 | 51 | Precomputed cluster assignments for the 50 stocks across two columns: `Hierarchical` and `Trend`. | Only the `Trend` column is used by `graph_builder.py`. |
| `data/raw/stock_order.npy` | 817 | 50 elements | Serialized numpy array with the fixed ordering of the 50 stock tickers. | Order: Energy (10), Healthcare (10), Technology (10), BasicMaterials (10), Industrials (10). |
| `models/saved/best_model_ours.pth` | 893,266 | State Dict | Saved PyTorch weights achieving paper-level performance on the validation set. | Evaluates to Acc: 0.8317, Pre: 0.7651, Rec: 0.8911, F1: 0.8233, AUC: 0.9041. |
| `models/saved/st_grad_best_model.pth` | 892,474 | State Dict | Reference training checkpoint. | Evaluates to Acc: 0.8269, Pre: 0.7596, Rec: 0.8875, F1: 0.8186, AUC: 0.9038. |

---

## 3. Data Pipeline & Schemas

### 3.1 Universe & Ordering
- **50 Stocks across 5 Sectors (10 stocks/sector):**
  - **Energy:** `CTRA`, `DKL`, `KNTK`, `MTDR`, `MUR`, `NEXT`, `NOG`, `NOV`, `PAGP`, `USAC`
  - **Healthcare:** `BIIB`, `CNC`, `GMAB`, `ICLR`, `INCY`, `SAUHY`, `SNN`, `TEVA`, `WAT`, `ZBH`
  - **Technology:** `AMKR`, `ARW`, `CAJPY`, `HPE`, `JKHY`, `LFUS`, `NTAP`, `PTC`, `STM`, `VRSN`
  - **Basic Materials:** `AXTA`, `IVPAF`, `LYSCF`, `LYSDY`, `MP`, `RGLD`, `SOUHY`, `SSD`, `SUZ`, `WFG`
  - **Industrials:** `ABM`, `ALG`, `HRI`, `HUBG`, `ODFL`, `SBLK`, `TEX`, `TNET`, `TRN`, `VWDRY`
- Raw data row count: Exactly 1,255 trading days per stock, spanning August 19, 2020 to August 18, 2025.

### 3.2 Processed Arrays (`data/processed/stock_dataset.npz`)
- `X_train`: shape `(856, 30, 50, 24)`, dtype `float32` (856 samples, 30 lookback days, 50 stocks, 24 features).
- `y_train`: shape `(856, 50, 1)`, dtype `float32` (binary label for next-day movement).
- `X_val`: shape `(368, 30, 50, 24)`, dtype `float32` (368 samples, 30 lookback days, 50 stocks, 24 features).
- `y_val`: shape `(368, 50, 1)`, dtype `float32`.
- `train_mean`: shape `(1, 1, 50, 1)`, dtype `float32` (mean per stock pooled across time and features).
- `train_std`: shape `(1, 1, 50, 1)`, dtype `float32` (standard deviation per stock pooled across time and features).
- `stock_order`: shape `(50,)`, dtype `object`.
- `features`: shape `(24,)`, list of 24 indicator names.
- `all_dates`: shape `(1255,)`, array of strings representing trading dates.

---

## 4. Architectural Component Breakdown

1. **Input Normalization & Projection:**
   - Input: $X \in \mathbb{R}^{B \times 24 \times 30 \times 50}$.
   - Trainable `nn.BatchNorm2d(24)` followed by `nn.Dropout(0.3)`.
   - `init_conv`: `nn.Conv2d(24, 32, kernel_size=1)`. Output: $[B, 32, 30, 50]$.

2. **Temporal Attention (`TemporalAttention`):**
   - Reshaped to $[(B \times N), T, D] = [(B \times 50), 30, 32]$.
   - `nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)`.
   - Residual connection + `nn.LayerNorm(32)`.
   - Output reshaped back to $[B, 32, 30, 50]$.
   - In `STBlock.forward`: added again to block input: `x = x + x_temp`.

3. **Multi-View Cross-Attention (`CrossAttentionView`):**
   - Three views: ST (`num_cav=0`), DS (`num_cav=1`), DT (`num_cav=2`).
   - Readout pooling: Concatenates max, min, mean across the selected axis.
   - Bottleneck U-D Conv: `Conv2d(in_ch -> 32) -> ReLU -> Conv2d(32 -> 1) -> ReLU`.
   - Signal gating: Multiplies gated attention `sigmoid(attention)` with `tanh(x_conv)`.
   - *Note on code implementation:* `x_conv` is computed using `F.conv2d` with a constant filter of ones (`torch.ones`), not a learned parameter.
   - Outputs: ST $[B, 1, T, N]$, DS $[B, D, 1, N]$, DT $[B, D, T, 1]$.
   - Fused via dynamic softmax scalar weights (`alpha_st`, `alpha_fs`, `alpha_ft`) with implicit broadcasting sum to produce $[B, 32, 30, 50]$.

4. **Bidirectional Diffusion Graph Convolution (`DiffusionConvolution`):**
   - Adaptive Adjacency: $E_1, E_2 \in \mathbb{R}^{50 \times 10}$, $A_{adp} = \text{softmax}(\text{ReLU}(E_1 E_2^T) + 10^{-6}, \text{dim}=1)$.
   - Prior Adjacency: Symmetric row-normalized cluster graph $A \in \mathbb{R}^{50 \times 50}$.
   - Propagation: For $k \in \{0, \dots, K_{diff}-1\}$:
     - Forward propagation: $A^k X W_k$.
     - Backward propagation: $(A^T)^k X W_k$.
     - Adaptive propagation (at $k=0$ only): $A_{adp} X W_0$.
   - Output: Sum of diffused representations $[B, 32, 30, 50]$.
   - *Note on code implementation:* `gcn_weights` allocates 3 tensors (for $K_{diff}=2$), but only indices 0 and 1 are ever accessed in the forward loop; index 2 receives no gradient.

5. **Residual & STBlock Output:**
   - Output of BDGC passes through `residual_conv`: `nn.Conv2d(32, 32, kernel_size=1)`.
   - Residual connection: $X_{res} = X_{init} + \text{residual\_conv}(X_{gcn})$.
   - Normalization & activation: $\text{ReLU}(\text{LayerNorm}([32, 30, 50])(X_{res}))$.

6. **Hierarchical Fusion & Output Head:**
   - Stacks $L=2$ ST-Blocks.
   - Code executes: `x_out = x + sum(layer_outputs)` where `layer_outputs = [block1_out, block2_out]`. Since `x` is already `block2_out`, this computes `block1_out + 2 * block2_out` and omits $X^{(0)}$.
   - `output_conv`: `Conv2d(32, 32, 1) -> ReLU -> Dropout(0.5) -> Conv2d(32, 1, 1)`.
   - Slices final time step: `output[:, :, -1, :].squeeze(1)` producing $[B, 50]$ logits.

---

## 5. Summary of Present vs Missing Repository Elements

| Component | Status | Location / Note |
|---|---|---|
| **Raw 50-ticker OHLCV data** | **Present** | `data/raw/*/*.csv` (all 1,255 rows clean and present). |
| **Cluster labels** | **Present (Static)** | `data/raw/cluster_results.csv` (precomputed). |
| **Cluster generation code (CNN-LSTM + K-Means)** | **Missing** | The deep behavioral clustering pipeline described in Section 2.3 is not implemented. |
| **Graph builder from cluster labels** | **Present** | `utils/graph_builder.py`. |
| **Feature engineering (24 features)** | **Present** | `utils/data_processor.py`. |
| **Pretrained weights** | **Present** | `models/saved/best_model_ours.pth` (216,429 params). |
| **Training loop (1 run, seed 43)** | **Present** | `train.py`. |
| **Five-run evaluation harness** | **Missing** | Paper reports 5-seed statistics; repo only runs a single seed (43). |
| **Clean validation vs stress test split** | **Missing** | Repo evaluates on all 368 validation samples; no 307 clean / 61 stress partition. |
| **Baselines (RF, SVM, LSTM, AASTHGCN, etc.)** | **Missing** | None of the 11 baseline models from Table 8 are present in the repository. |
| **Ablation suite (Table 9)** | **Missing** | Ablation code is not present; only the complete model is implemented. |
| **Sensitivity & Orthogonality scripts** | **Missing** | MI / Cosine similarity (Table 11) and (L, K_diff) sensitivity scripts are missing. |
| **Yahoo Finance data downloader** | **Missing** | Raw CSVs are provided, but no automated fetcher is included. |
