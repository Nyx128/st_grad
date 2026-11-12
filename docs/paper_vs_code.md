# Paper vs. Code Discrepancy Analysis: ST-GRAD (Phase 0)

**Reference Paper:** Fu, Huang, & Yu (2026), *ST-GRAD: Multi-view cross-attention and bidirectional diffusion for stock trend prediction*, *Expert Systems With Applications* 330, 133046.  
**Audited Codebase:** `ST-GRAD-Stock-Prediction-6E7D`

---

## 1. Executive Summary of Discrepancies

While the released repository successfully instantiates a neural architecture with the exact reported parameter count of **216,429** and the bundled weights `best_model_ours.pth` produce metrics closely matching Table 8, a rigorous mathematical and structural audit reveals **ten fundamental discrepancies** between the printed paper and the source code.

These discrepancies span the data pipeline, target generation, graph construction, core neural operators, and experimental protocols. They are detailed below.

---

## 2. Comprehensive Discrepancy Matrix

| Area | Paper Specification | Repository Implementation | Severity | Reproducibility Impact |
|---|---|---|---|---|
| **Target Label: Price Source** | Eq. (29): $R_{i,t+1} = \frac{P_{i,t+1} - \bar{P}_{i,t}}{\bar{P}_{i,t}}$ where $P$ is the raw closing price. | `create_labels()` computes returns using `data[..., 0]`, which is the `Relative_Close` feature ($C_t / \text{MA}_{30}(C)$), NOT the raw closing price. | **Critical** | Target definitions differ fundamentally; relative-close return is compared to a mean of relative-close ratios. |
| **Target Label: Volatility Scaling** | Eq. (28): $T_{i,t} = T_{\text{base}} \cdot (1 + k \cdot \tanh(\|\text{vol}_{i,t}\|))$ with $k=3, T_{\text{base}}=0.0003$. | `data_processor.py:98`: `volatility = np.tanh(np.abs(volume_pct) * 8)`. Introduces an unmentioned factor of 8. | **High** | Changes threshold distribution, causing `tanh` to saturate much faster on small volume changes. |
| **Target Label: Inactive Masking** | Section 6.1.1: Exclude inactive samples with $\|\text{vol}_{i,t}\| < 0.3\%$. | Inactive samples are initialized to label 0 and included in training/loss/metrics as negative examples. No validity mask is stored. | **High** | Corrupts class balance by counting filtered samples as genuine price drops. |
| **Feature Normalization** | Section 3, Eq. (3): $X_{in} = \frac{X - \mu}{\sigma_{\text{std}}}$ across feature dimension. | `data_processor.py:146`: `np.nanmean(X_train, axis=(0,1,3))` computes mean/std of shape `[1, 1, 50, 1]`, pooling all 24 features and 30 time steps per ticker into a single scalar. | **High** | Mixes completely incompatible feature scales (e.g. RSI [0,100] with Return [-0.1, 0.1]). |
| **VWAP Indicator Formula** | Table 5: $\frac{\sum_{i=0}^{29} TP_{t-i} + V_t \cdot P_t}{\sum_{i=0}^{29} V_{t-i} + V_t}$. | `data_processor.py:41`: `(typical_price.rolling(30).sum() + volume * close) / (volume.rolling(30).sum() + volume)`. Matches the unusual Table 5 formula, but both diverge from standard VWAP. | **Medium** | Literature divergence; Table 5 double-counts day $t$. |
| **Prior Graph Generation** | Section 2.3: CNN-LSTM feature extraction + 5 macro indicators + Z-score + K-Means ($K=5$). | No deep clustering code is provided. Precomputed assignments in `cluster_results.csv` are simply loaded. Adjacency is symmetric. | **Medium** | Prior graph is non-reproducible from raw OHLCV alone; must be treated as a precomputed asset. |
| **MVCA: Signal Projection** | Eq. (12, 15, 18): Learned $1 \times 1$ conv: $X_{ST} = \tanh(\text{Conv2d}_{1\times 1}(X)) \odot \sigma(\Psi)$. | `models/st_grad.py:73, 76, 79`: Uses `F.conv2d` with a constant filter of ones (`torch.ones`). No learnable parameters in signal branches. | **Critical** | Replaces learned feature projection with a static sum; alters model representation capacity. |
| **MVCA: View Shapes & Fusion** | Section 4: All views produce $[B, D, T, N]$ feature representations before elementwise adaptive fusion. | View outputs have collapsed dimensions: ST $[B, 1, T, N]$, DS $[B, D, 1, N]$, DT $[B, D, T, 1]$. Fused via implicit broadcasting. | **High** | Fused tensor is not a weighted mixture of complete representations, but an outer sum of orthogonal projections. |
| **BDGC: Dead Diffusion Parameter** | Eq. (24): Sums $j=0 \dots K_{\text{diff}}-1$. Code sets $K_{\text{diff}}=2$. | `DiffusionConvolution` allocates $K_{\text{diff}}+1 = 3$ weights. The forward loop runs `range(2)`. The 3rd parameter receives no gradient. | **Medium** | Dead parameter wastes 1,024 weights; gradient is perpetually `None`. |
| **Hierarchical Skip Fusion** | Eq. (27) & Algorithm 2: $\hat{Y} = \text{OutputConv}\left(\sum_{k=0}^{L} X^{(k)}\right)$ where $X^{(0)} = X_{\text{init}}$. | Code: `x_out = x + sum(layer_outputs)`. Since `x` is the last block output, this omits $X^{(0)}$ and adds $X^{(L)}$ twice. | **Medium** | Alters skip connection weighting; early features are not directly forwarded to the head. |
| **Loss Function Weighting** | Section 6.1.4 & Table 7: AdamW, standard parameters. No loss weighting mentioned. | `train.py:74-75`: `pos_weight = (1 - mean(y)) / mean(y)`. Employs weighted `BCEWithLogitsLoss`. | **Medium** | Shifts precision-recall trade-off; unweighted BCE would yield different recall. |
| **Validation & Test Split** | Section 6.8: Physical isolation of 307 clean validation days and 61 stress test days. | `train.py:126`: Evaluates and reports metrics on the exact same 368 validation samples used for checkpoint selection. | **High** | No blind out-of-sample evaluation; risk of optimistic bias on the validation partition. |

---

## 3. Deep-Dive Mathematical Audits

### 3.1 Target Label Discrepancy (Eqs. 28–30)
In the paper:
$$R_{i,t+1} = \frac{P_{i,t+1} - \bar{P}_{i,t}}{\bar{P}_{i,t}}$$
where $P_{i,t+1}$ is the closing price on day $t+1$, and $\bar{P}_{i,t}$ is the average closing price over a historical lookback window.
In the repository (`utils/data_processor.py:86-100`):
```python
current = data[i + window_size - 1, :, 0]    # Relative_Close feature!
next_day = data[i + window_size, :, 0]       # Relative_Close feature!
avg_price = np.mean(data[i:i + window_size, :, 0], axis=0) # Mean of Relative_Close!
returns = (next_day - avg_price) / (avg_price + 1e-8)
```
Because feature index 0 is `Relative_Close = close / MA30(close)`, the codebase is computing the percentage change of a normalized ratio relative to its 30-day mean, rather than the raw asset return relative to its historical price. Furthermore, the volatility threshold in code is:
$$\text{volatility} = \tanh(8 \cdot |\text{volume\_pct}|)$$
whereas Eq. (28) specifies $\tanh(|\text{volume\_pct}|)$. The factor of 8 causes the hyperbolic tangent to saturate at 1.0 even for minor volume movements ($> 0.25$).

### 3.2 MVCA Structural Discrepancy (Eqs. 11–18)
The paper describes each cross-attention view as:
$$X^{(k)}_{ST} = \tanh(\text{Conv2d}_{1\times 1}(X^{(k-1)})) \odot \sigma(\Psi^{(k)}_{S \rightleftharpoons T})$$
where $\text{Conv2d}_{1\times 1}$ is a trainable transformation that maintains channel dimension $D=32$.
In `models/st_grad.py`:
```python
if self.num_cav == 0:
    x_conv = F.conv2d(x.permute(0, 1, 3, 2), torch.ones(1, self.latent_channels, 1, 1).to(device)).permute(0, 1, 3, 2)
    return torch.tanh(x_conv) * torch.sigmoid(attention)
```
1. `torch.ones` is a fixed, non-trainable filter of ones that performs an unweighted sum across channels.
2. The output of `F.conv2d` collapses the channel dimension to 1, producing tensor shape $[B, 1, T, N]$ for ST, $[B, D, 1, N]$ for DS, and $[B, D, T, 1]$ for DT.
3. In `STBlock.forward`:
```python
x_cross = weights[0] * x_st + weights[1] * x_fs + weights[2] * x_ft
```
These tensors only have compatible dimensions because PyTorch implicitly broadcasts $[B, 1, T, N] + [B, D, 1, N] + [B, D, T, 1] \rightarrow [B, D, T, N]$.

### 3.3 BDGC Diffusion Weights (Eq. 24)
Eq. (24) defines:
$$X^{(k)}_{gcn} = \sum_{j=0}^{K_{\text{diff}}-1} \left( (A^j + (A^T)^j) \hat{X}^{(j)} W_j \right) + A_{adp} \hat{X}^{(j)} W_0$$
In `DiffusionConvolution.__init__`:
```python
self.gcn_weights = nn.ParameterList([
    nn.Parameter(torch.Tensor(feature_dim, feature_dim))
    for _ in range(max_diffusion_step + 1)
])
```
For `max_diffusion_step = 2`, 3 matrices ($32 \times 32 = 1,024$ params each) are initialized. But `forward()` executes:
```python
for k in range(self.max_diffusion_step): # k in [0, 1]
    ...
    weighted = diffused @ self.gcn_weights[k]
```
`self.gcn_weights[2]` is never used in the computation graph. Its gradient is permanently `None`.

### 3.4 Hierarchical Skip Fusion (Eq. 27 & Algorithm 2)
Algorithm 2 states:
$$\hat{Y} = \text{OutputConv}\left(X^{(0)} + X^{(1)} + \dots + X^{(L)}\right)$$
In `ST_GRAD.forward`:
```python
layer_outputs = []
for block in self.st_blocks:
    x = block(x, A)
    layer_outputs.append(x)
x_out = x + sum(layer_outputs)
```
Here:
- `layer_outputs = [X_1, X_2]`
- After the loop, `x = X_2`
- `x_out = X_2 + (X_1 + X_2) = X_1 + 2 \cdot X_2`
- $X^{(0)}$ (`init_conv(x)`) is omitted completely.
