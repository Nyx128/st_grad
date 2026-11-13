# ST-GRAD Prediction Explanation: HPE
**Target Date:** 2024-02-28 | **Track:** `paper_aligned` | **Model Hash:** `a00bad84ef`

## 1. Directional Prediction Summary
- **Predicted Trend:** **`DOWN`** (Class `0`)
- **Probability:** `0.0207` (Confidence: `95.9%`)
- **Pre-Sigmoid Logit:** `-3.8553`

**Ground Truth Label:** `DOWN` (Class `0`) | **Outcome:** **`CORRECT`**

## 2. Key Technical Indicator Drivers (Integrated Gradients)
*Convergence:* `PASSED` (Delta: `0.24%` over `50` gauss_legendre steps)
*Intrinsic vs Peer Reliance:* `54.3%` of attribution originates from HPE's own technicals.

| Rank | Technical Indicator | Absolute Score | Directional Impact |
|:---:|---|:---:|:---:|
| 1 | `BB_Position` | 2.373 | 🔴 Bearish (-) (-1.527) |
| 2 | `RSI` | 1.913 | 🔴 Bearish (-) (-1.197) |
| 3 | `Close_MA_Ratio` | 1.018 | 🔴 Bearish (-) (-0.645) |
| 4 | `RSI_Relative` | 0.999 | 🔴 Bearish (-) (-0.564) |
| 5 | `Volatility` | 0.752 | 🟢 Bullish (+) (+0.080) |

## 3. Multi-View Cross-Attention (MVCA) Hierarchy
- **View Impact Order:** **DS** > **ST** > **DT**
  - `ST` View Ablation Drop: `Δlogit = -0.1759` (`Δprob = -0.0039`)
  - `DS` View Ablation Drop: `Δlogit = -0.7607` (`Δprob = -0.0226`)
  - `DT` View Ablation Drop: `Δlogit = 0.1389` (`Δprob = +0.0026`)

## 4. Spatio-Temporal Graph & Peer Influence
- **Static Sector Prior Channel Drop:** `Δlogit = 0.1605`
- **Adaptive Dynamic Graph Channel Drop:** `Δlogit = 0.0618`
- **Top Influential Peer Stocks:** `ICLR` (Δ0.058), `CTRA` (Δ0.050), `CAJPY` (Δ0.042)

## 5. Scientific Caveats & Disclaimers
> ⚠️ *Model attribution identifies features correlated with predictions, NOT real-world economic causality.*
> ⚠️ *Technical indicators exhibit multicollinearity; credit may be split among correlated proxies.*
> ⚠️ *Interventions measure model sensitivity under synthetic ablation and do not simulate market mechanics.*