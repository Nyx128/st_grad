# Reproduction Runbook: ST-GRAD (Phase 0)

**Project:** Explainable ST-GRAD Reproduction  
**Date:** October 2026  
**Audited Platform:** Windows 11, Python 3.11.9, NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)  
**Virtual Environment:** project-local `.venv` (created per README section 2)  
**PyTorch Version:** `2.14.0+cu132` (CUDA enabled)

---

## 1. Environment Verification

Before executing any commands, verify the Python interpreter, CUDA availability, and key package versions:

```powershell
# Verify Python version
python -V
# Output: Python 3.11.9

# Verify CUDA and GPU device
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
# Output: CUDA available: True | Device: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## 2. Software Compatibility Finding & Shim

### The Pandas 3.0+ Issue
The bundled code in `ST-GRAD-Stock-Prediction-6E7D/utils/data_processor.py` was written for Pandas 2.0.3 and invokes:
```python
df = df.fillna(method='ffill').fillna(0)
```
In **Pandas 3.0+** (installed in the project venv as `pandas 3.0.6`), the `method` parameter has been permanently removed from `NDFrame.fillna()`, raising:
```text
TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'
```

### The Solution: Non-Destructive Runner
To adhere to the core directive—**never modify raw code or data destructively**—we implement a non-destructive runtime compatibility shim in `src/run_repo_compat.py`. It dynamically wraps `.fillna()` to redirect `method='ffill'` calls to `.ffill()`, enabling clean execution without altering repository files.

---

## 3. Step-by-Step Execution of the Bundled Repository

All commands below should be executed from the project root (the directory containing `README.md`).

### Step 1: Adjacency Matrix Construction
Constructs the symmetrically normalized prior graph from `data/raw/cluster_results.csv`:

```powershell
python src/run_repo_compat.py --mode graph
```
- **Execution Time:** ~0.05 seconds
- **Output Artifacts:**
  - `ST-GRAD-Stock-Prediction-6E7D/data/processed/normalized_adj_matrix.npy` (shape: `[50, 50]`, symmetric, row sums $\approx 1.0$)
  - `ST-GRAD-Stock-Prediction-6E7D/data/raw/stock_order.npy` (50 stock tickers)

---

### Step 2: Data Preprocessing & Feature Engineering
Extracts 24 technical indicators from all 50 stock CSVs, computes relative-close targets, applies 70/30 chronological splitting, and normalizes features:

```powershell
python src/run_repo_compat.py --mode data
```
- **Execution Time:** ~9.8 seconds
- **Output Artifacts:**
  - `ST-GRAD-Stock-Prediction-6E7D/data/processed/stock_dataset.npz` containing:
    - `X_train`: shape `(856, 30, 50, 24)`
    - `y_train`: shape `(856, 50, 1)`
    - `X_val`: shape `(368, 30, 50, 24)`
    - `y_val`: shape `(368, 50, 1)`
    - `train_mean`: shape `(1, 1, 50, 1)`
    - `train_std`: shape `(1, 1, 50, 1)`

---

### Step 3: Checkpoint Evaluation (Direct Inference)

Evaluate the author-supplied reference checkpoint `best_model_ours.pth`:

```powershell
python src/run_repo_compat.py --mode predict --load_path ST-GRAD-Stock-Prediction-6E7D/models/saved/best_model_ours.pth
```
**Obtained Execution Output:**
```text
Loading weights from ST-GRAD-Stock-Prediction-6E7D/models/saved/best_model_ours.pth...

=== Model Performance ===
Accuracy  : 0.8317
Precision : 0.7651
Recall    : 0.8911
F1        : 0.8233
Auc       : 0.9041
```

Evaluate the default checkpoint `st_grad_best_model.pth`:

```powershell
python src/run_repo_compat.py --mode predict --load_path ST-GRAD-Stock-Prediction-6E7D/models/saved/st_grad_best_model.pth
```
**Obtained Execution Output:**
```text
Loading weights from ST-GRAD-Stock-Prediction-6E7D/models/saved/st_grad_best_model.pth...

=== Model Performance ===
Accuracy  : 0.8269
Precision : 0.7596
Recall    : 0.8875
F1        : 0.8186
Auc       : 0.9038
```

---

### Step 4: Metric Comparison with Paper

| Metric | `best_model_ours.pth` (Reproduced) | `st_grad_best_model.pth` (Reproduced) | Published Paper (Table 8) | Status |
|---|---:|---:|---:|---|
| **Accuracy** | **0.8317** | 0.8269 | $0.8313 \pm 0.0018$ | **Exact match within $0.2\sigma$** |
| **Precision** | **0.7651** | 0.7596 | $0.7705 \pm 0.0063$ | **Within $0.9\sigma$** |
| **Recall** | **0.8911** | 0.8875 | $0.8799 \pm 0.0078$ | **Within $1.4\sigma$** |
| **F1-Score** | **0.8233** | 0.8186 | $0.8215 \pm 0.0009$ | **Within $2.0\sigma$** |
| **ROC-AUC** | **0.9041** | 0.9038 | $0.9046 \pm 0.0003$ | **Within $1.7\sigma$** |

**Conclusion:** The supplied checkpoint `best_model_ours.pth` reproduces the paper's reported headline results on the validation partition with high precision.

---

### Step 5: Model Training Smoke Test
Run a fast 2-epoch training run to verify GPU gradient flow, optimizer step, learning rate scheduler, and checkpoint writing:

```powershell
python src/run_repo_compat.py --mode train --epochs 2 --batch_size 32
```
- **Execution Time:** 6.37 seconds on NVIDIA RTX 4060 Laptop GPU (~3.1s/epoch).
- **Peak Training GPU Memory:** ~1.85 GB VRAM.
- **Result:** Successfully trained, computed validation loss, updated best checkpoint, and wrote `st_grad_best_model.pth`.

---

## 4. Prioritized List of Unresolved Questions for Phase 1

1. **Closing Price Window for Benchmark Price ($\bar{P}_{i,t}$):**
   - *Question:* In Eq. (29), the paper defines $\bar{P}_{i,t}$ as the "average closing price over the past $t$ days." Does $t$ denote a rolling 30-day window (matching the lookback $T=30$) or a cumulative expanding window?
   - *Proposed Resolution:* Implement rolling 30-day average (`MA30(Close)`) as a clearly labeled assumption in `paper_aligned`.

2. **VWAP Feature Formula Discrepancy:**
   - *Question:* Table 5 defines VWAP with an additional current-day term:
     $$\text{VWAP} = \frac{\sum_{i=0}^{29} TP_{t-i} + V_t \cdot P_t}{\sum_{i=0}^{29} V_{t-i} + V_t}$$
     Does this formula intentionally add $V_t \cdot P_t$ twice (since $i=0$ is day $t$), or was it a typographical error in Table 5?
   - *Proposed Resolution:* Provide both `paper_table_literal` and `conventional_vwap` as configurable options in `paper_aligned`.

3. **Stochastic Label Smoothing Partitioning:**
   - *Question:* Eq. (30) specifies a 10% stochastic dropout on positive labels. Was this applied exclusively to training data, or also to validation and stress test sets?
   - *Proposed Resolution:* Implement `paper_literal` (seed-fixed across all samples) and `clean_eval` (stochastic dropout on training positives only).

4. **Exact Calendar Boundaries for the 61-Day Stress Window:**
   - *Question:* Section 6.8 specifies a 61-trading-day stress window from November 2024 to January 2025 within the 368-sample out-of-sample period (2024-02-27 to 2025-08-15). Which exact trading dates demarcate this window?
   - *Proposed Resolution:* Pin the 61 trading dates between 2024-11-01 and 2025-01-31 from the aligned calendar into a versioned split manifest.

---

## 5. Explainability Layer Operational Commands (Phase 1–5)

### Unified CLI Orchestrator (`src/cli.py`)

1. **Instance-Level Model Explanations:**
   ```bash
   # Terminal report card
   python src/cli.py explain --ticker HPE --split val_clean --index 0

   # Export Markdown card, responsive HTML card, and JSON case record
   python src/cli.py explain --ticker HPE --split val_clean --index 0 \
     --save_card artifacts/explainability/hpe_card.md \
     --save_html artifacts/explainability/hpe_card.html \
     --save_json artifacts/explainability/hpe_case.json
   ```

2. **Phase 5 Dual-Track & Regime Shift Benchmark (Hypothesis H6):**
   ```bash
   python src/cli.py benchmark-regimes --samples-clean 25 --samples-stress 25 --output_dir artifacts/explainability
   ```
   Generates `dual_track_regime_benchmark.json`, SVG charts, and `regime_benchmark_dashboard.html`.

3. **Controlled Hypothesis Experiments (H1–H4):**
   ```bash
   python src/cli.py evaluate-explanations --track paper_aligned --samples 15
   python src/cli.py evaluate-explanations --track repo_compat --samples 15
   ```

4. **Model Performance Evaluation on Partitions:**
   ```bash
   python src/cli.py evaluate --track paper_aligned --split clean_val
   python src/cli.py evaluate --track paper_aligned --split stress
   python src/cli.py evaluate --track repo_compat --split clean_val
   ```

5. **Full Unit Test Suite Execution (42 Tests):**
   ```bash
   python -m unittest discover -s src -p "test_*.py"
   ```
