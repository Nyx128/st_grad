# Data Protocol Specification: ST-GRAD (Phase 1)

**Reference Paper:** Fu, Huang, & Yu (2026), *ST-GRAD: Multi-view cross-attention and bidirectional diffusion for stock trend prediction*, *Expert Systems With Applications* 330, 133046.  
**Scope:** Raw data schema, cross-ticker calendar alignment, 24 technical indicator formulas, target generation, validity masks, chronological splits, and leakage-safe normalization.

---

## 1. Raw Data Specification & Stock Universe

### 1.1 Universe Definition (Table 4)
The universe consists of exactly $N = 50$ stocks across 5 sectors (10 stocks per sector):

| Sector | Ticker Symbols |
|---|---|
| **Energy** | `CTRA`, `DKL`, `KNTK`, `MTDR`, `MUR`, `NEXT`, `NOG`, `NOV`, `PAGP`, `USAC` |
| **Healthcare** | `BIIB`, `CNC`, `GMAB`, `ICLR`, `INCY`, `SAUHY`, `SNN`, `TEVA`, `WAT`, `ZBH` |
| **Technology** | `AMKR`, `ARW`, `CAJPY`, `HPE`, `JKHY`, `LFUS`, `NTAP`, `PTC`, `STM`, `VRSN` |
| **Basic Materials** | `AXTA`, `IVPAF`, `LYSCF`, `LYSDY`, `MP`, `RGLD`, `SOUHY`, `SSD`, `SUZ`, `WFG` |
| **Industrials** | `ABM`, `ALG`, `HRI`, `HUBG`, `ODFL`, `SBLK`, `TEX`, `TNET`, `TRN`, `VWDRY` |

- **Date Range:** 2020-08-19 to 2025-08-18 (exactly 1,255 trading days per ticker).
- **Raw Fields:** `date` (YYYY-MM-DD), `open`, `high`, `low`, `close`, `volume`.
- **Integrity Rule:** Raw files in `data/raw/` are treated as strictly immutable.

### 1.2 Cross-Stock Calendar Alignment
- A single master calendar of 1,255 trading dates is formed by taking the intersection of dates across all 50 tickers.
- Any missing observations within a series are forward-filled (`ffill`) only from past dates; backward-filling from future dates is strictly prohibited.
- For prices, if any invalid/infinite values occur, they are set to `NaN` and forward-filled. For volume, zero-trading periods are tracked in a validity mask.

---

## 2. Feature Engineering Protocol (Table 5)

Twenty-four technical indicators $F=24$ are extracted causally. For any date $t$, indicator calculations use only historical data up to and including day $t$.

Let $O_t, H_t, L_t, C_t, V_t$ denote open, high, low, close, and volume at day $t$.  
Let $r_t = \frac{C_t - C_{t-1}}{C_{t-1}}$ denote the 1-day percentage return.  
Let $TP_t = \frac{H_t + L_t + C_t}{3}$ denote typical price.  
Let $\text{MA}_k(X)_t = \frac{1}{k}\sum_{i=0}^{k-1} X_{t-i}$ denote the $k$-day simple moving average.  
Let $\sigma_k(X)_t = \sqrt{\frac{1}{k}\sum_{i=0}^{k-1} (X_{t-i} - \text{MA}_k(X)_t)^2}$ denote the $k$-day standard deviation.  
Let $\text{EMA}_k(X)_t$ denote exponential moving average with smoothing factor $\alpha = \frac{2}{k+1}$.

### 2.1 The 24 Engineered Indicators

| Idx | Feature Name | Paper Formula (Table 5) | Implementation Specification |
|---:|---|---|---|
| 0 | `Relative_Close` | $C_t / \text{MA}_{30}(C)_t$ | Ratio of current close to its 30-day moving average. Guard with $\epsilon=10^{-8}$. |
| 1 | `OC_Ratio` | $(C_t - O_t) / O_t$ | Intraday close-to-open return ratio. |
| 2 | `VWAP` | $\frac{\sum_{i=0}^{29} TP_{t-i} + V_t C_t}{\sum_{i=0}^{29} V_{t-i} + V_t}$ | **Track A (`repo_compat`):** Implements Table 5 literal formula.<br>**Track B (`paper_aligned`):** Supports both Table 5 literal and conventional VWAP: $\frac{\sum_{i=0}^{29} TP_{t-i} V_{t-i}}{\sum_{i=0}^{29} V_{t-i}}$. |
| 3 | `RSI` | $100 - \frac{100}{1 + \text{RS}_{t, 21}}$ | $\text{RS}_{t, 21} = \frac{\text{MA}_{21}(\text{Gain})}{\text{MA}_{21}(\text{Loss}) + \epsilon}$. Clipped to $[0, 100]$. |
| 4 | `RSI_Relative` | $\text{RSI}_t / 50 - 1$ | Centered and scaled RSI into $[-1, 1]$. |
| 5 | `MACD_Signal` | $\frac{\text{MACD}_t - \text{EMA}_9(\text{MACD})_t}{C_t + \epsilon}$ | Normalized MACD histogram where $\text{MACD}_t = \text{EMA}_{12}(C)_t - \text{EMA}_{26}(C)_t$. |
| 6 | `Volume_Ratio` | $V_t / \text{MA}_{30}(V)_t$ | Volume relative to its 30-day baseline. |
| 7 | `ROC` | $C_t / C_{t-12} - 1$ | 12-day price rate of change. |
| 8 | `Momentum_5` | $C_t / C_{t-5} - 1$ | 5-day price momentum. |
| 9 | `Momentum_10` | $C_t / C_{t-10} - 1$ | 10-day price momentum. |
| 10 | `MA5_vs_MA20` | $(\text{MA}_5(C)_t - \text{MA}_{20}(C)_t) / \text{MA}_{20}(C)_t$ | Percentage gap between short and long moving averages. |
| 11 | `BB_Width` | $4 \cdot \sigma_{20}(C)_t / \text{MA}_{20}(C)_t$ | Bollinger band width relative to middle band. |
| 12 | `BB_Position` | $\frac{C_t - (\text{MA}_{20}(C)_t - 2\sigma_{20}(C)_t)}{4\sigma_{20}(C)_t + \epsilon}$ | Relative position of close price within lower-to-upper Bollinger span. |
| 13 | `ATR` | $\text{MA}_{14}(TR)_t$ | 14-day average true range: $TR_t = \max(H_t - L_t, \|H_t - C_{t-1}\|, \|L_t - C_{t-1}\|)$. |
| 14 | `Volume_Change_Pct` | $V_t / V_{t-1} - 1$ | Daily percentage change in trading volume ($\text{vol}_{i,t}$). |
| 15 | `Close_vs_High` | $(C_t - L_t) / (H_t - L_t + \epsilon)$ | Position of closing price within the daily high-low range. |
| 16 | `Volatility` | $\sigma_5(C)_t / \text{MA}_5(C)_t$ | 5-day coefficient of variation of close price. |
| 17 | `Price_Volume_Corr` | $\text{Corr}_{10}(r, \text{Volume\_Change\_Pct})_t$ | 10-day rolling Pearson correlation between return and volume change. |
| 18 | `Volume_Momentum` | $V_t / V_{t-5} - 1$ | 5-day volume momentum. |
| 19 | `Price_Acceleration`| $r_t - r_{t-1}$ | First difference of 1-day returns (second difference of price log-proxy). |
| 20 | `High_Low_Ratio` | $H_t / (L_t + \epsilon)$ | Daily high-to-low price ratio. |
| 21 | `Close_MA_Ratio` | $C_t / \text{MA}_{20}(C)_t$ | Ratio of current close to 20-day moving average. |
| 22 | `Sector_Relative_Strength` | $r_{t, 5} - \text{MA}_{10}(r_{t, 5})$ | Asset momentum relative to its own 10-day mean (proxy defined in paper). |
| 23 | `Volatility_Ratio` | $\sigma_5(C)_t / (\sigma_{20}(C)_t + \epsilon)$ | Short-term to medium-term volatility ratio. |

---

## 3. Target Generation & Validity Mask Protocol

### 3.1 Mathematical Specification (Eqs. 28–30)
For each stock $i$ and prediction day $t$ (predicting trend on $t+1$):

1. **Benchmark Fair-Value Price $\bar{P}_{i,t}$:**
   $$\bar{P}_{i,t} = \frac{1}{T_{\text{lookback}}} \sum_{\tau=0}^{T_{\text{lookback}}-1} P_{i, t-\tau}$$
   - **Track A (`repo_compat`):** Uses mean of `Relative_Close` ($C / \text{MA}_{30}(C)$).
   - **Track B (`paper_aligned`):** Uses rolling 30-day mean of raw closing prices ($P = \text{Close}$).

2. **Next-Day Return $R_{i,t+1}$ (Eq. 29):**
   $$R_{i,t+1} = \frac{P_{i,t+1} - \bar{P}_{i,t}}{\bar{P}_{i,t}}$$

3. **Volatility-Adaptive Threshold $T_{i,t}$ (Eq. 28):**
   $$T_{i,t} = T_{\text{base}} \cdot \left(1 + k \cdot \tanh(|\text{vol}_{i,t}|)\right)$$
   where $T_{\text{base}} = 0.0003$, $k = 3$, and $\text{vol}_{i,t} = \frac{V_{i,t} - V_{i,t-1}}{V_{i,t-1}}$.
   - **Track A (`repo_compat`):** Uses $\tanh(8 \cdot |\text{vol}_{i,t}|)$.
   - **Track B (`paper_aligned`):** Uses $\tanh(|\text{vol}_{i,t}|)$.

4. **Stochastic Label Smoothing (Eq. 30):**
   $$y_{i,t} = \begin{cases} 1, & \text{if } R_{i,t+1} > T_{i,t} \text{ and } u_{i,t} > 0.10 \\ 0, & \text{otherwise} \end{cases}$$
   where $u_{i,t} \sim \text{Uniform}(0, 1)$.
   - **Track A (`repo_compat`):** Fixed NumPy random seed 43 applied at generation time.
   - **Track B (`paper_aligned`):**
     - Mode `paper_literal`: Seeded 10% dropout generated once and stored in dataset manifest.
     - Mode `clean_eval`: 10% dropout applied strictly to training positives; validation and stress-test targets remain 100% deterministic.

5. **Sample Validity Mask (Section 6.1.1):**
   A sample at $(i, t)$ is valid if and only if:
   $$\text{Mask}_{i,t} = \left(|\text{vol}_{i,t}| \ge 0.003\right) \land \left(P_{i,t} > 0\right) \land \left(P_{i,t+1} > 0\right) \land \neg \text{isnan}(R_{i,t+1})$$
   - **Track A (`repo_compat`):** Inactive samples are set to $y=0$ without masking.
   - **Track B (`paper_aligned`):** Samples with $\text{Mask}_{i,t} = \text{False}$ are masked with weight 0 in loss calculation and excluded from metric computations.

---

## 4. Chronological Splitting Protocol

Given 1,255 raw trading dates:
- Initial lookback warm-up: 30 trading days (dates 0 through 29).
- Horizon: 1 trading day (date 1,254 is the target for window ending on 1,253).
- **Total Sequential Windowed Samples:** Exactly $1,224$ samples ($1255 - 30 - 1 = 1224$).

```text
  Dates: 0              30                          886                1193        1254
         |--- Warm-up --|--------- Train -----------|--- Clean Val ----|-- Stress -|
                        |      856 samples          |   307 samples    | 61 samples|
                        |  2020-09-30 to 2024-02-26 |  2024-02-27 to   | Nov 2024  |
                        |                           |   Oct 2024 &     |    to     |
                        |                           | Feb-Aug 2025     | Jan 2025  |
```

### 4.1 Partition Definitions
1. **Training Partition (`train`):**
   - Samples 0 to 855 (first 70%, 856 samples).
   - Target Dates: September 30, 2020 through February 26, 2024.
2. **Out-of-Sample Partition (`oos_all`):**
   - Samples 856 to 1,223 (latter 30%, 368 samples).
   - Target Dates: February 27, 2024 through August 15, 2025.
3. **Clean Validation Partition (`validation_clean`):**
   - Exactly 307 samples within the 368-sample out-of-sample partition, excluding the 61 stress dates.
   - Used strictly for early stopping and best-checkpoint selection based on Validation F1.
4. **Blind Stress Test Partition (`stress_test`):**
   - Exactly 61 samples spanning the market turbulence window of November 2024 to January 2025.
   - Completely held out; no model weights, hyperparameters, or decision thresholds are tuned against this partition.

---

## 5. Leakage-Safe Normalization Protocol

1. **Training-Only Statistics:**
   Mean $\mu_F$ and standard deviation $\sigma_F$ are computed strictly on the 856 training samples:
   $$\mu_f = \frac{1}{856 \cdot 30 \cdot 50} \sum_{s=0}^{855} \sum_{t=0}^{29} \sum_{i=0}^{49} X_{s, t, i, f}$$
   $$\sigma_f = \sqrt{\frac{1}{856 \cdot 30 \cdot 50} \sum_{s=0}^{855} \sum_{t=0}^{29} \sum_{i=0}^{49} (X_{s, t, i, f} - \mu_f)^2}$$
2. **Standardization Forward Pass:**
   $$X_{\text{norm}, f} = \frac{X_f - \mu_f}{\max(\sigma_f, 10^{-6})}$$
   Frozen $\mu_f$ and $\sigma_f$ are applied to validation, stress-test, and inference tensors.
3. **Storage & Manifest:**
   All normalization statistics, feature names, stock ordering, date indices, and target masks are serialized into `stock_dataset_manifest.json` with SHA-256 hashes.
