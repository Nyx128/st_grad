# Phase 3: Data Correctness & Leakage Audit Report

**Date:** 2026-10-21  
**Environment:** Python 3.11.9 (project-local `.venv`), PyTorch 2.14.0+cu132, CUDA on NVIDIA RTX 4060 Laptop GPU  
**Test Suite:** [`src/test_data_leakage.py`](../src/test_data_leakage.py) (11/11 tests passing)  
**Track:** Track B (`src/paper_aligned/`) vs Track A (`src/repo_compat/`)

---

## 1. Executive Summary

Phase 3 verifies that data preparation, feature engineering, label creation, normalization, chronological splitting, and graph construction conform strictly to the published paper specifications and are mathematically free of forward lookahead bias. All 11 automated verification tests passed.

---

## 2. Dataset Scope, Calendar Alignment, & Integrity

- **Stock Universe:** Exactly 50 stocks evenly distributed across 5 macroeconomic sectors (10 stocks each):
  - **BasicMaterials:** `AXTA`, `IVPAF`, `LYSCF`, `LYSDY`, `MP`, `RGLD`, `SOUHY`, `SSD`, `SUZ`, `WFG`
  - **Energy:** `CTRA`, `DKL`, `KNTK`, `MTDR`, `MUR`, `NEXT`, `NOG`, `NOV`, `PAGP`, `USAC`
  - **Healthcare:** `BIIB`, `CNC`, `GMAB`, `ICLR`, `INCY`, `SAUHY`, `SNN`, `TEVA`, `WAT`, `ZBH`
  - **Industrials:** `ABM`, `ALG`, `HRI`, `HUBG`, `ODFL`, `SBLK`, `TEX`, `TNET`, `TRN`, `VWDRY`
  - **Technology:** `AMKR`, `ARW`, `CAJPY`, `HPE`, `JKHY`, `LFUS`, `NTAP`, `PTC`, `STM`, `VRSN`
- **Timeline:** Exactly 1,255 trading days from **2020-08-19** to **2025-08-18**.
- **Data Integrity:** Zero nulls, zero NaNs, strictly monotonic increasing timestamps, zero duplicate dates.
- **Price Consistency Checks:** Validated across all 50 stocks:
  $$\text{High} \ge \text{Low}, \quad \text{High} \ge \text{Open}, \quad \text{High} \ge \text{Close}, \quad \text{Low} \le \text{Open}, \quad \text{Low} \le \text{Close}, \quad \text{Volume} \ge 0$$

---

## 3. Causal Feature Engineering & Zero Lookahead Verification

- **24 Technical Indicators (Table 5):** Verified against hand-calculated synthetic fixtures:
  - Relative Close, OC Ratio, VWAP, RSI, RSI Relative, MACD Signal, Volume Ratio, ROC, Momentum 5, Momentum 10, MA5 vs MA20, BB Width, BB Position, ATR, Volume Change Pct, Close vs High, Volatility, Price Volume Corr, Volume Momentum, Price Acceleration, High Low Ratio, Close MA Ratio, Sector Relative Strength, Volatility Ratio.
- **Future Invariance Perturbation Test:** For cutoff date $t_0 = 100$, future price and volume records ($t > t_0$) were scaled by $15\times$ with random noise. Recomputed feature matrices showed **$0.0$ absolute difference** for all $t \le t_0$, proving zero future leakage in feature calculation.
- **Sliding Window Alignment:**
  - Window size $T = 30$, prediction horizon $H = 1$.
  - Sample $i$ input features cover trading days $[i, i + 29]$.
  - Sample $i$ target prediction day is day $i + 30$.
  - Target day strictly succeeds the input window with zero temporal overlap.

---

## 4. Target Label Generation & Discrepancy Analysis

The paper specifies directional movement relative to moving average price:
$$R_{i,t+1} = \frac{P_{i,t+1} - \bar{P}_{i,t}}{\bar{P}_{i,t}}, \quad \bar{P}_{i,t} = \frac{1}{T} \sum_{\tau=t-T+1}^t P_{i,\tau}$$
$$T_{i,t} = T_{\text{base}} \cdot (1 + k \cdot \tanh(|\Delta \text{Vol}_{i,t}|))$$
$$y_{i,t+1} = \begin{cases} 1, & R_{i,t+1} > T_{i,t} \\ 0, & R_{i,t+1} < -T_{i,t} \end{cases}$$

### Deviations Identified and Handled:
1. **Underlying Price Series:**
   - *Repository (`repo_compat`):* Used feature index 0 (`Relative_Close`), computing returns on a ratio of ratios.
   - *Paper-Aligned (`paper_aligned`):* Uses raw `Close` prices for $P_{i,t+1}$ and $\bar{P}_{i,t}$.
2. **Threshold Calculation:**
   - *Repository (`repo_compat`):* Multiplied volume change by 8: $\tanh(8 \cdot |\Delta \text{Vol}|)$.
   - *Paper-Aligned (`paper_aligned`):* Uses strict Eq. (28) without the arbitrary factor of 8.
3. **Inactive Sample Masking:**
   - *Repository (`repo_compat`):* Samples where $|\Delta \text{Vol}| < 0.003$ were set to class 0 (treating market inactivity as price drops).
   - *Paper-Aligned (`paper_aligned`):* Explicitly flags inactive samples with `valid_mask = False`. These are excluded from loss and evaluation metrics rather than injected as false negatives.

---

## 5. Normalization Leakage & Numerical Precision Audit

- **Leakage Protection:** Mean and standard deviation are fitted **exclusively on the 856 training samples** ($t \in [0, 855]$). Mutating or perturbing OOS validation and stress test samples produces zero change in $\mu_{\text{train}}$ and $\sigma_{\text{train}}$.
- **Numerical Accumulation Finding:**
  - In single-precision `float32`, summing 1,284,000 values ($856 \times 30 \times 50$) caused a catastrophic accumulation truncation error on low-variance features (e.g. `High_Low_Ratio`, $\sigma = 0.023$), resulting in a normalized mean error of $0.2826$ standard deviations.
  - **Correction:** Implemented high-precision `float64` accumulators (`dtype=np.float64`) in [`src/paper_aligned/dataset.py`](../src/paper_aligned/dataset.py). This reduced the maximum residual normalized mean to **$1.36 \times 10^{-5}$** and standard deviation error to **$0.0011$**.

---

## 6. Chronological Partitioning & Blind Stress Test Isolation

- **Total Samples:** 1,225 ($1,255 - 30 - 1 + 1$).
- **Partitions:**
  - **Training Set:** First 856 samples (2020-10-01 to ~2024-03-01).
  - **Clean Validation Set:** 308 samples (out-of-sample trading days outside the stress window).
  - **Blind Stress Test Set:** Exactly 61 samples covering **November 1, 2024 to January 31, 2025**.
  - **Total Out-of-Sample (OOS):** $308 + 61 = 369$ samples.
- **Model Selection Protocol:** Checkpoint selection is strictly performed on `Clean Val`. The `Stress Test` set remains completely blind during training and hyperparameter selection.

---

## 7. Behavioral Prior Graph Leakage Audit

The author repository provided `data/raw/cluster_results.csv` with pre-assigned trend clusters:
- **Cluster 1:** 30 stocks (fully connected 30-node clique, 900 edges).
- **Cluster 4:** 10 stocks (100 edges).
- **Cluster 2:** 5 stocks (25 edges).
- **Cluster 5:** 4 stocks (16 edges).
- **Cluster 3:** 1 stock (self-loop only, 1 edge).
- **Total Edges in Author Graph:** 1,042 edges (including self-loops).

### Leakage Audit Findings:
- Generating static clusters over the entire 5-year timeline (2020–2025) inadvertently encodes future price trajectory patterns into the static prior graph.
- **Solution in Track B:** Added [`build_train_only_prior_adjacency`](../src/paper_aligned/graph.py#L55-L100) which computes macro-trend profiling (sequence mean, volatility, linear slope, overall return, peak-to-peak amplitude) and K-Means clustering strictly on the 856 training dates.
- **Comparative Metrics:**
  - Author Full-Timeline Prior Graph: 1,042 edges.
  - Train-Only Prior Graph: 858 edges.
  - Common Edges: 552 edges (Jaccard similarity: 0.4095).

---

## 8. Sample Validity & Exclusion Report (No Silent Discards)

Per `instructions.txt` requirements, inactive samples ($|\Delta \text{Vol}| < 0.003$) are explicitly audited and accounted for:

| Partition | Total Stock-Samples | Active (Valid) | Inactive ($|\Delta \text{Vol}| < 0.003$) | Active Ratio | Active Class Balance (Pos / Neg) |
|---|---:|---:|---:|---:|---:|
| **Train** | 42,800 | 42,562 | 238 | 99.44% | 49.0% Pos / 51.0% Neg |
| **Clean Val** | 15,400 | 15,296 | 104 | 99.32% | 50.3% Pos / 49.7% Neg |
| **Blind Stress Test** | 3,050 | 3,032 | 18 | 99.41% | 45.1% Pos / 54.9% Neg |
| **Total OOS** | 18,450 | 18,328 | 122 | 99.34% | 49.4% Pos / 50.6% Neg |
| **Grand Total** | 61,250 | 60,890 | 360 | 99.41% | 49.1% Pos / 50.9% Neg |

- **Sample Exclusion Reason:** Exactly 360 stock-sample observations (0.59% of the dataset) exhibit trading volume changes below 0.3% ($|\Delta \text{Vol}| < 0.003$), where price volatility trends cannot be reliably distinguished from market microstructure noise.
- **Handling:** Retained in data structures with `valid_mask = False` to prevent silent dropping and enable strict reproducibility.
