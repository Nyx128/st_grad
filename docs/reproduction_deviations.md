# Reproduction Deviations & Track Isolation: ST-GRAD

**Project:** Explainable ST-GRAD Reproduction  
**Purpose:** Formal specification of two isolated implementation tracks and explicit recording of all design deviations, assumptions, and mathematical corrections.

---

## 1. Dual-Track Architecture

To guarantee scientific rigor, the codebase maintains two strictly isolated execution tracks:

```text
                        ┌──────────────────────────────────────────────────┐
                        │              ST-GRAD Research Suite              │
                        └─────────────────────────┬────────────────────────┘
                                                  │
                  ┌───────────────────────────────┴───────────────────────────────┐
                  ▼                                                               ▼
    ┌───────────────────────────┐                                   ┌───────────────────────────┐
    │   Track A: repo_compat    │                                   │  Track B: paper_aligned   │
    ├───────────────────────────┤                                   ├───────────────────────────┤
    │ • Preserves bundled repo  │                                   │ • Strict paper equations  │
    │   logic and quirks        │                                   │ • Raw close for labels    │
    │ • Loads author weights    │                                   │ • Masked invalid targets  │
    │ • Matches Table 8 metrics │                                   │ • Learnable MVCA convs    │
    │ • Benchmark verification  │                                   │ • 5-seed blind stress test│
    └───────────────────────────┘                                   └───────────────────────────┘
```

### Track A: `repo_compat`
- **Objective:** Mirror the exact behavior of `ST-GRAD-Stock-Prediction-6E7D` to verify checkpoint compatibility and reproduce published numerical outputs.
- **Rules:**
  - Never alter `repo_compat` to make it look like the paper.
  - Preserve relative-close label calculation, `tanh(8 * vol)` scaling, pooled `[1, 1, 50, 1]` normalization, `torch.ones` convolution in MVCA, dead diffusion parameter, and skip summation formula.
  - Evaluate on the 368-sample validation set.

### Track B: `paper_aligned`
- **Objective:** Implement the published method as specified by the text, equations, and tables in Fu et al. (2026).
- **Rules:**
  - Compute targets strictly from raw closing prices (`Close`).
  - Use literal Eq. (28) threshold formula without the factor of 8.
  - Implement boolean validity masks to exclude inactive node-days from loss and evaluation.
  - Use per-feature Z-score normalization fitted strictly on training data.
  - Implement trainable $1 \times 1$ convolutions for MVCA view signal transformations.
  - Allocate only active diffusion parameters ($K_{\text{diff}}=2$).
  - Implement Algorithm 2 hierarchical skip summation with $X^{(0)}$.
  - Physically isolate the clean validation set (307 samples) and blind stress-test set (61 samples).
  - Run across 5 independent random seeds with reported mean and sample standard deviation.

---

## 2. Controlled Deviation Catalogue

Each entry is assigned an authoritative classification tag:
- **`[PAPER]`**: Grounded directly in the published article text, equations, or tables.
- **`[REPO]`**: Grounded directly in the inspected source code.
- **`[INFERRED]`**: Necessary engineering choice where the paper is underspecified.
- **`[DEVIATION]`**: Explicitly chosen divergence for experimental comparison.
- **`[UNRESOLVED]`**: Open question with competing interpretations.

---

### DEV-01: Target Return Calculation
- **Tags:** `[PAPER]` vs `[REPO]`
- **Paper State:** Eq. (29) defines $R_{i,t+1} = (P_{i,t+1} - \bar{P}_{i,t}) / \bar{P}_{i,t}$, where $P$ is the stock closing price.
- **Repo State:** Computes return using feature index 0 (`Relative_Close = close / MA30(close)`), not the closing price.
- **Resolution:**
  - `repo_compat`: Retain `Relative_Close` calculation.
  - `paper_aligned`: Maintain a dedicated raw closing price sidecar array to compute $R_{i,t+1}$ faithfully from prices.

---

### DEV-02: Volatility Threshold Scaling
- **Tags:** `[PAPER]` vs `[REPO]`
- **Paper State:** Eq. (28) defines $T_{i,t} = T_{\text{base}} \cdot (1 + k \cdot \tanh(|\text{vol}_{i,t}|))$ with $T_{\text{base}}=0.0003, k=3$.
- **Repo State:** Multiplies volume percentage by 8: $\tanh(8 \cdot |\text{volume\_pct}|)$.
- **Resolution:**
  - `repo_compat`: Retain factor of 8.
  - `paper_aligned`: Implement literal Eq. (28) without the factor of 8.

---

### DEV-03: Inactive Sample Filtering
- **Tags:** `[PAPER]` vs `[REPO]`
- **Paper State:** Section 6.1.1 explicitly requires excluding inactive samples with volume change $< 0.3\%$.
- **Repo State:** Inactive samples are assigned label 0 and included in the dataset, effectively treating inactive days as stock declines.
- **Resolution:**
  - `repo_compat`: Retain zero-assignment behavior.
  - `paper_aligned`: Generate a boolean mask `valid_target_mask` where entries with volume change $< 0.3\%$ or missing values are `False`. Exclude `False` entries from loss and metrics.

---

### DEV-04: Normalization Axis
- **Tags:** `[PAPER]` vs `[REPO]`
- **Paper State:** Section 3 states Z-score normalization $X_{in} = (X - \mu) / \sigma_{\text{std}}$ to standardize features.
- **Repo State:** Computes mean/std over axes `(0, 1, 3)`, yielding shape `[1, 1, 50, 1]`. Pools time and all 24 heterogeneous features per stock into a single mean and standard deviation.
- **Resolution:**
  - `repo_compat`: Retain pooled shape `[1, 1, 50, 1]`.
  - `paper_aligned`: Compute Z-score normalization per feature channel across training samples: shape `[1, 24, 1, 1]`.

---

### DEV-05: MVCA Signal Convolution Operators
- **Tags:** `[PAPER]` vs `[REPO]`
- **Paper State:** Section 4 defines $\text{Conv2d}_{1 \times 1}$ as a trainable feature transformation.
- **Repo State:** Uses `F.conv2d` with static `torch.ones` kernels (fixed non-trainable sum).
- **Resolution:**
  - `repo_compat`: Retain `torch.ones` fixed convolution.
  - `paper_aligned`: Implement `nn.Conv2d(in_channels, out_channels, kernel_size=1)` with learned weights and proper Xavier initialization.

---

### DEV-06: MVCA View Tensor Shapes & Broadcasting
- **Tags:** `[PAPER]` vs `[REPO]`
- **Paper State:** All three views produce full representations in $\mathbb{R}^{B \times D \times T \times N}$ before weighted summation.
- **Repo State:** View tensors have collapsed single dimensions: ST $[B, 1, T, N]$, DS $[B, D, 1, N]$, DT $[B, D, T, 1]$, fused via implicit broadcasting.
- **Resolution:**
  - `repo_compat`: Retain broadcasting behavior.
  - `paper_aligned`: Verify and assert explicit tensor shapes $[B, D, T, N]$ for all views prior to fusion.

---

### DEV-07: Unused Diffusion Parameter
- **Tags:** `[REPO]`
- **Repo State:** Allocates 3 weight matrices for $K_{\text{diff}}=2$, but the forward loop only iterates $k \in [0, 1]$. Parameter 2 is dead (`grad is None`).
- **Resolution:**
  - `repo_compat`: Retain 3 allocated weights to preserve exact 216,429 parameter count for loading author weights.
  - `paper_aligned`: Allocate exactly $K_{\text{diff}}$ weight matrices ($2 \times 32 \times 32$), eliminating the dead weight.

---

### DEV-08: Hierarchical Skip Summation
- **Tags:** `[PAPER]` vs `[REPO]`
- **Paper State:** Eq. (27) & Algorithm 2 define $\hat{Y} = \text{OutputConv}(\sum_{k=0}^{L} X^{(k)})$ including $X^{(0)} = X_{\text{init}}$.
- **Repo State:** Computes `x + sum(layer_outputs)` where `layer_outputs = [X_1, X_2]` and `x = X_2`, resulting in $X_1 + 2 X_2$ (omits $X_0$).
- **Resolution:**
  - `repo_compat`: Retain `x + sum(layer_outputs)`.
  - `paper_aligned`: Implement explicit sum $X^{(0)} + X^{(1)} + X^{(2)}$.

---

### DEV-09: Prior Graph Provenance
- **Tags:** `[REPO]` vs `[UNRESOLVED]`
- **Paper State:** Section 2.3 details CNN-LSTM extraction + 5 macro statistics + K-Means ($K=5$).
- **Repo State:** The clustering code is missing from the repository; only precomputed `cluster_results.csv` is provided.
- **Resolution:**
  - `repo_compat`: Load `cluster_results.csv`.
  - `paper_aligned`: Maintain `cluster_results.csv` as a provided static prior while documenting its missing provenance; in Phase 1, implement an independent, leakage-free clustering pipeline fitted on training data only.

---

### DEV-10: Loss Function Weighting
- **Tags:** `[PAPER]` vs `[REPO]`
- **Paper State:** Standard optimization with AdamW; no loss weighting described in Table 7.
- **Repo State:** Uses `BCEWithLogitsLoss` with `pos_weight = (1 - p) / p`.
- **Resolution:**
  - `repo_compat`: Retain `pos_weight`.
  - `paper_aligned`: Test both unweighted standard BCE and weighted BCE as a named configuration ablation.

---

### DEV-11: Evaluation Partition & Stress Isolation
- **Tags:** `[PAPER]` vs `[REPO]`
- **Paper State:** Section 6.8 defines a 307-day clean validation set and an isolated 61-day stress testing window (Nov 2024 - Jan 2025).
- **Repo State:** Evaluates and reports metrics on all 368 validation samples without isolation.
- **Resolution:**
  - `repo_compat`: Evaluate on all 368 samples.
  - `paper_aligned`: Enforce physical isolation between the 307 clean validation samples (for checkpoint selection) and the 61 blind stress test samples (evaluated only once after model freezing).
