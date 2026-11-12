# Known Limitations & Critical Diagnostics

**Project:** ST-GRAD Reproduction & Explainability Research  
**Document Type:** Scientific Integrity & Limitations Analysis  
**Date:** 2026-10-21  

---

## 1. Prior Graph Lookahead Leakage in Static Clusters

- **The Issue:** The author's provided `cluster_results.csv` groups stocks into 5 clusters based on macro-trend profiling (sequence mean, volatility, linear slope, overall return, peak-to-peak amplitude) computed across the entire 5-year period (2020-08-19 to 2025-08-18).
- **Consequence:** Stocks in the validation and stress testing periods (2024–2025) are connected via static prior edges that were informed by their future 2024–2025 trajectories.
- **Remediation in Track B:** We implemented [`build_train_only_prior_adjacency`](../src/paper_aligned/graph.py#L55-L100), which restricts clustering profiling strictly to the first 856 training trading days.
- **Takeaway:** Any real-world deployment of ST-GRAD must re-cluster dynamically using rolling trailing windows rather than a single static 5-year graph.

---

## 2. Parameter Randomization Sanity Check in Residual Architectures

- **The Finding:** In Phase 7, randomizing the output head and top ST-Block parameters resulted in a post-randomization Spearman correlation of **$\rho = 0.594$** (higher than the ideal $< 0.50$ threshold).
- **Root Cause Analysis:** ST-GRAD implements explicit hierarchical residual fusion:
  $$X_{\text{sum}} = X_{\text{init}} + X_1 + X_2$$
  Because $X_{\text{init}} = \text{Conv2d}_{\text{init}}(\text{BatchNorm}(X))$ is added directly to the decoder input, the gradient of the logit with respect to $X$ contains a direct additive path:
  $$\frac{\partial \hat{y}}{\partial X} = \frac{\partial \hat{y}}{\partial X_{\text{sum}}} \left( \frac{\partial X_{\text{init}}}{\partial X} + \frac{\partial X_1}{\partial X} + \frac{\partial X_2}{\partial X} \right)$$
  Even when top layers ($X_2$, head) are randomized, the initial projection and normalization scale gradients retain strong sensitivity to raw input feature magnitudes.
- **Scientific Caveat:** Users must recognize that in deeply residual networks, gradient-based attributions partially reflect input feature variance and projection scaling, alongside learned higher-order interactions.

---

## 3. Post-Hoc Attribution is Non-Causal

- **Attribution vs Causality:** Integrated Gradients highlights which technical indicators and historical days strongly correlate with the model's logit output. It **does not prove** that intervening in the real-world market on that indicator would alter the stock's future return.
- **Confounding:** Financial indicators exhibit severe multicollinearity (e.g., `RSI` and `RSI_Relative`, or `Close_MA_Ratio` and `BB_Position`). Attributions may distribute credit across collinear proxies rather than identifying unique economic drivers.

---

## 4. Universe Scope & Survivorship Bias

- **Universe Limitation:** The study is restricted to 50 large-cap US equities across 5 sectors selected in 2025.
- **Survivorship Bias:** Selecting stocks that survived and traded continuously throughout 2020–2025 eliminates delisted, acquired, or bankrupt firms, skewing long-term trend statistics upward.
- **Generalization:** Findings on this 50-stock universe cannot be assumed to transfer directly to small-cap equities, emerging markets, or non-equity asset classes (crypto, commodities, forex) without separate empirical validation.

---

## 5. Sensitivity to Inactive Sample Masking

- **Label Noise:** Around $0.59\%$ of the dataset observations have volume change $|\Delta \text{Vol}| < 0.003$.
- **Discrepancy:** The author's original code mapped these inactive samples to class 0 (price drop). This artificial negative bias deflated precision and F1 in Track A.
- **Caution:** While masking inactive samples in Track B properly filters microstructure noise, practitioners must define an explicit operational policy for how an algorithmic trading strategy handles "inactive" prediction days (e.g. cash holding or position freeze).
