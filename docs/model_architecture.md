# Model Architecture Specification: ST-GRAD (Phase 1)

**Reference Paper:** Fu, Huang, & Yu (2026), *ST-GRAD: Multi-view cross-attention and bidirectional diffusion for stock trend prediction*, *Expert Systems With Applications* 330, 133046.  
**Scope:** Tensor dimensions, neural modules, mathematical operations, activations, normalizations, parameter counting, and complete forward algorithms.

---

## 1. Tensor Dimension Conventions

All modules strictly adhere to the standard 4D spatio-temporal financial tensor format:

| Symbol | Description | Standard Dimension |
|---|---|---:|
| $B$ | Batch size (date-windows) | 32 (configurable) |
| $F$ | Input technical feature channels | 24 |
| $D$ | Latent hidden feature channels | 32 |
| $T$ | Backtracking lookback window (days) | 30 |
| $N$ | Number of stock nodes in graph | 50 |
| $d$ | Adaptive node embedding dimension | 10 |
| $L$ | Number of stacked Spatio-Temporal Blocks | 2 |
| $K_{\text{diff}}$ | Maximum graph diffusion steps | 2 |

- **Input Tensor Shape:** $X_{\text{in}} \in \mathbb{R}^{B \times F \times T \times N}$
- **Latent Hidden State Shape:** $X \in \mathbb{R}^{B \times D \times T \times N}$
- **Graph Adjacency Shape:** $A \in \mathbb{R}^{N \times N}$
- **Output Prediction Logits:** $\hat{Y} \in \mathbb{R}^{B \times N}$

---

## 2. Module Specifications

### 2.1 Input Standardization & Feature Projection (Eqs. 3–5)
1. **Batch Normalization (Eq. 4):**
   $$X_{\text{norm}} = \text{BatchNorm2d}(X_{\text{in}}) \in \mathbb{R}^{B \times F \times T \times N}$$
   - Trainable parameters: $\gamma \in \mathbb{R}^F$, $\beta \in \mathbb{R}^F$ (48 parameters).
   - Followed by `nn.Dropout(p=0.3)`.
2. **Initial 2D Convolution (Eq. 5):**
   $$X_{\text{init}} = \text{Conv2d}_{\text{init}}(X_{\text{norm}}) \in \mathbb{R}^{B \times D \times T \times N}$$
   - Kernel size: $1 \times 1$, mapping $F=24 \to D=32$.
   - Trainable parameters: Weight $[32, 24, 1, 1]$ (768 params) + Bias $[32]$ (32 params) = 800 parameters.

---

### 2.2 Temporal Attention Module (Eqs. 6–9)
Models historical sequence dependencies independently for each stock across the batch.

1. **Reshape (Eq. 6):**
   $$X_{\text{reshape}} = \text{permute}(0, 3, 2, 1)(X) \to \text{reshape}((B \cdot N), T, D) \in \mathbb{R}^{(B \cdot 50) \times 30 \times 32}$$
2. **Multi-Head Self-Attention (Eq. 7):**
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   - Heads: $h = 4$, $d_k = D / h = 32 / 4 = 8$.
   - Trainable parameters: In-proj $[3 \cdot 32, 32]$ + Out-proj $[32, 32]$ + biases = 4,224 parameters.
3. **Residual Connection & Layer Normalization (Eq. 8):**
   $$X_{\text{attn}} = \text{LayerNorm}_D(X_{\text{reshape}} + \text{Attention}(Q, K, V))$$
   - Trainable parameters: $\gamma \in \mathbb{R}^{32}$, $\beta \in \mathbb{R}^{32}$ (64 parameters).
4. **Restore Dimensions (Eq. 9):**
   $$X_{\text{temp}} = \text{reshape}(B, N, T, D) \to \text{permute}(0, 3, 2, 1) \in \mathbb{R}^{B \times D \times T \times N}$$
5. **Block Input Update:**
   $$X \leftarrow X + X_{\text{temp}}, \quad X_{\text{init}} \leftarrow X$$

---

### 2.3 Multi-View Cross-Attention (MVCA) (Eqs. 10–18)
Decouples market dynamics into three orthogonal interaction views using the "Aggregate-Transform-Gate" paradigm:

```text
  X in [B, D, T, N]
         │
         ├───► ST View (Ψ_ST): Pool over D ───► U-D Conv ───► Gate [B, D, T, N]
         ├───► DS View (Ψ_DS): Pool over T ───► U-D Conv ───► Gate [B, D, T, N]
         └───► DT View (Ψ_DT): Pool over N ───► U-D Conv ───► Gate [B, D, T, N]
                                                                     │
                                             Learned Softmax Fusion ◄┘
                                                                     │
                                                                     ▼
                                                          Fused X_fused in [B, D, T, N]
```

#### A. Spatio-Temporal View ($\Psi_{S \rightleftharpoons T}$) (Eqs. 10–12)
- **Aggregation:** Max, mean, and min pooling along feature dimension $D$:
  $$A_D^{(k)} = [\max_D(X), \text{mean}_D(X), \min_D(X)] \in \mathbb{R}^{B \times 3 \times T \times N}$$
- **U-D Conv Bottleneck (Eq. 11):**
  $$\Psi_{S \rightleftharpoons T}^{(k)} = \text{ReLU}\left(\text{Conv2d}_{\text{down}}(\text{ReLU}(\text{Conv2d}_{\text{up}}(A_D^{(k)})))\right) \in \mathbb{R}^{B \times 1 \times T \times N}$$
  - $\text{Conv2d}_{\text{up}}$: Kernel $1 \times 1$, mapping $3 \to C_{\text{latent}}=32$ (128 params).
  - $\text{Conv2d}_{\text{down}}$: Kernel $1 \times 1$, mapping $32 \to 1$ (33 params).
- **Signal Gating (Eq. 12):**
  $$X_{ST}^{(k)} = \tanh\left(\text{Conv2d}_{1\times 1}(X)\right) \odot \sigma\left(\Psi_{S \rightleftharpoons T}^{(k)}\right)$$
  - *Track A (`repo_compat`):* Uses unweighted constant $F.\text{conv2d}$ with `torch.ones` (output shape $[B, 1, T, N]$).
  - *Track B (`paper_aligned`):* Uses learned $\text{Conv2d}(D \to D, 1 \times 1)$ (output shape $[B, D, T, N]$).

#### B. Feature-Spatio View ($\Psi_{D \rightleftharpoons S}$) (Eqs. 13–15)
- **Permutation:** Permute to $[B, T, D, N]$.
- **Sequence Patching:** Partition $T=30$ into $p = 15$ non-overlapping patches of length 2.
- **Aggregation (Eq. 13):** Max, mean, min pooling within each patch along time, concatenated:
  $$A_T^{(k)} \in \mathbb{R}^{B \times (3p) \times D \times N} = \mathbb{R}^{B \times 45 \times D \times N}$$
- **U-D Conv Bottleneck (Eq. 14):**
  $$\Psi_{D \rightleftharpoons S}^{(k)} = \text{ReLU}\left(\text{Conv2d}_{\text{down}}(\text{ReLU}(\text{Conv2d}_{\text{up}}(A_T^{(k)})))\right) \in \mathbb{R}^{B \times 1 \times D \times N}$$
  - $\text{Conv2d}_{\text{up}}$: Kernel $1 \times 1$, mapping $45 \to 32$ (1,472 params).
  - $\text{Conv2d}_{\text{down}}$: Kernel $1 \times 1$, mapping $32 \to 1$ (33 params).
- **Signal Gating (Eq. 15):**
  $$X'_{DS} = \tanh\left(\text{Conv2d}_{1\times 1}(X')\right) \odot \sigma\left(\Psi_{D \rightleftharpoons S}^{(k)}\right)$$
  Permuted back to $[B, D, T, N]$.

#### C. Feature-Temporal View ($\Psi_{D \rightleftharpoons T}$) (Eqs. 16–18)
- **Permutation:** Permute to $[B, N, D, T]$.
- **Aggregation (Eq. 16):** Max, mean, min pooling across all $N=50$ stocks:
  $$A_S^{(k)} \in \mathbb{R}^{B \times 3 \times D \times T}$$
- **U-D Conv Bottleneck (Eq. 17):**
  $$\Psi_{D \rightleftharpoons T}^{(k)} = \text{ReLU}\left(\text{Conv2d}_{\text{down}}(\text{ReLU}(\text{Conv2d}_{\text{up}}(A_S^{(k)})))\right) \in \mathbb{R}^{B \times 1 \times D \times T}$$
  - $\text{Conv2d}_{\text{up}}$: Kernel $1 \times 1$, mapping $3 \to 32$ (128 params).
  - $\text{Conv2d}_{\text{down}}$: Kernel $1 \times 1$, mapping $32 \to 1$ (33 params).
- **Signal Gating (Eq. 18):**
  $$X''_{DT} = \tanh\left(\text{Conv2d}_{1\times 1}(X'')\right) \odot \sigma\left(\Psi_{D \rightleftharpoons T}^{(k)}\right)$$
  Permuted back to $[B, D, T, N]$.

#### D. Adaptive View Fusion (Eqs. 21–22)
Three learnable scalar parameters $\alpha_{ST}, \alpha_{DS}, \alpha_{DT} \in \mathbb{R}$:
$$[w_{ST}, w_{DS}, w_{DT}] = \text{softmax}([\alpha_{ST}, \alpha_{DS}, \alpha_{DT}])$$
$$\hat{X}^{(k)} = w_{ST} X_{ST}^{(k)} + w_{DS} X_{DS}^{(k)} + w_{DT} X_{DT}^{(k)} \in \mathbb{R}^{B \times D \times T \times N}$$

---

### 2.4 Bidirectional Diffusion Graph Convolution (BDGC) (Eqs. 23–24)

1. **Adaptive Adjacency Matrix (Eq. 23):**
   $$A_{\text{adp}} = \text{softmax}\left(\text{ReLU}(E_1 E_2^T) + \epsilon\right) \in \mathbb{R}^{N \times N}$$
   - Learnable node embeddings: $E_1, E_2 \in \mathbb{R}^{50 \times 10}$ ($2 \times 500 = 1,000$ params), $\epsilon = 10^{-6}$.
   - Softmax is applied row-wise ($\text{dim}=1$).
2. **Prior Adjacency Matrix:**
   $A_{\text{norm}} \in \mathbb{R}^{50 \times 50}$, precomputed from behavioral clustering, symmetric and row-normalized.
3. **Graph Diffusion Propagation (Eq. 24):**
   For each hop $k \in \{0, \dots, K_{\text{diff}}-1\}$ (with $K_{\text{diff}}=2$):
   - Forward propagation: $A^k \hat{X} W_k$
   - Backward propagation: $(A^T)^k \hat{X} W_k$
   - Adaptive propagation ($k=0$ only): $A_{\text{adp}} \hat{X} W_0$
   - Channel transformation weights: $W_k \in \mathbb{R}^{D \times D} = \mathbb{R}^{32 \times 32}$ ($1,024$ params each).
   - Tensor contraction: $\text{einsum}('ij, bjc \to bic', A, X_{\text{reshaped}})$.

---

### 2.5 ST-Block Residual Connection & LayerNorm (Eqs. 25–26)
1. **Residual Projection (Eq. 25):**
   $$X_{\text{res}} = X_{\text{init}} + \text{Conv2d}_{1\times 1}(X_{\text{gcn}}) \in \mathbb{R}^{B \times D \times T \times N}$$
   - `residual_conv`: $\text{Conv2d}(32 \to 32, 1 \times 1)$ (1,056 params).
2. **Layer Normalization & ReLU (Eq. 26):**
   $$X_{\text{out}} = \text{ReLU}\left(\text{LayerNorm}_{[D, T, N]}(X_{\text{res}})\right)$$
   - Trainable parameters: $2 \times (32 \cdot 30 \cdot 50) = 96,000$ parameters.

---

### 2.6 Hierarchical Skip Fusion & Output Decoder (Eq. 27)
1. **Dense Skip Aggregation:**
   - **Track A (`repo_compat`):** $X_{\text{sum}} = X_2 + (X_1 + X_2) = X_1 + 2 X_2$.
   - **Track B (`paper_aligned`):** $X_{\text{sum}} = X_{\text{init}} + X_1 + X_2$.
2. **Pointwise Output Decoder Network:**
   $$\hat{Y}_{\text{all}} = \text{Conv2d}(32 \to 1, 1 \times 1)\left(\text{Dropout}_{0.5}\left(\text{ReLU}\left(\text{Conv2d}(32 \to 32, 1 \times 1)(X_{\text{sum}})\right)\right)\right)$$
   - Parameters: $(32 \times 32 + 32) + (1 \times 32 + 1) = 1,056 + 33 = 1,089$ parameters.
3. **Temporal Slicing:**
   $$\hat{Y} = \hat{Y}_{\text{all}}[:, 0, -1, :] \in \mathbb{R}^{B \times N}$$
   Returns logits for binary prediction.

---

## 3. Total Parameter Count Verification

Matching Table 7 reported parameter count of **216,429**:

| Component | Sub-module | Calculation | Parameters |
|---|---|---|---:|
| **Initial Projection** | `BatchNorm2d(24)` | $2 \times 24$ | 48 |
| | `Conv2d_init(24 -> 32, 1x1)` | $24 \times 32 + 32$ | 800 |
| **ST-Block 1** | `TemporalAttention` (MHA 32, 4 heads + LN) | $4,224 + 64$ | 4,288 |
| | MVCA View `Psi_ST` (U-D Conv: $3 \to 32 \to 1$) | $(3 \times 32 + 32) + (32 \times 1 + 1)$ | 161 |
| | MVCA View `Psi_DS` (U-D Conv: $45 \to 32 \to 1$) | $(45 \times 32 + 32) + (32 \times 1 + 1)$ | 1,505 |
| | MVCA View `Psi_DT` (U-D Conv: $3 \to 32 \to 1$) | $(3 \times 32 + 32) + (32 \times 1 + 1)$ | 161 |
| | View fusion scalars ($\alpha_{ST}, \alpha_{DS}, \alpha_{DT}$) | $1 + 1 + 1$ | 3 |
| | `AdaptiveAdjacency` ($E_1, E_2 \in \mathbb{R}^{50 \times 10}$) | $50 \times 10 \times 2$ | 1,000 |
| | `DiffusionConvolution` ($3 \times [32, 32]$ weights) | $3 \times 32 \times 32$ | 3,072 |
| | `residual_conv(32 -> 32, 1x1)` | $32 \times 32 + 32$ | 1,056 |
| | `LayerNorm([32, 30, 50])` | $2 \times (32 \times 30 \times 50)$ | 96,000 |
| **ST-Block 2** | Identical structure to ST-Block 1 | Duplicate of above | 107,246 |
| **Output Head** | `Conv2d(32 -> 32, 1x1)` + `Conv2d(32 -> 1, 1x1)` | $(32 \times 32 + 32) + (32 \times 1 + 1)$ | 1,089 |
| **Total Parameters** | | | **216,429** |

---

## 4. End-to-End Forward Algorithm

### Algorithm 1: The ST-Block Forward Pass
```text
Input: Hidden state X^(k-1) in R^(B x D x T x N), Graph A_norm in R^(N x N)
Output: Updated hidden state X^(k) in R^(B x D x T x N)

1.  X_temp <- TemporalAttention(X^(k-1))
2.  X <- X^(k-1) + X_temp
3.  X_init <- X
4.  X_ST <- ST_View(X)
5.  X_DS <- DS_View(X)
6.  X_DT <- DT_View(X)
7.  w <- softmax([alpha_ST, alpha_DS, alpha_DT])
8.  X_fused <- w_ST * X_ST + w_DS * X_DS + w_DT * X_DT
9.  A_adp <- softmax(ReLU(E1 @ E2.T) + eps)
10. X_gcn <- 0
11. For j = 0 to K_diff - 1:
        X_gcn <- X_gcn + (A_norm^j + (A_norm.T)^j) @ X_fused @ W_j
        if j == 0:
            X_gcn <- X_gcn + A_adp @ X_fused @ W_0
12. X_res <- X_init + Conv2d_1x1(X_gcn)
13. X^(k) <- ReLU(LayerNorm(X_res))
14. Return X^(k)
```

### Algorithm 2: The ST-GRAD End-to-End Network Forward Pass
```text
Input: Input tensor X_in in R^(B x F x T x N), Graph A_norm in R^(N x N)
Output: Prediction logits Y_hat in R^(B x N)

1.  X_norm <- BatchNorm2d(X_in)
2.  X_init <- Conv2d_init(X_norm)
3.  layer_outputs <- [X_init]
4.  X <- X_init
5.  For k = 1 to L:
        X <- ST_Block_k(X, A_norm)
        layer_outputs.append(X)
6.  X_sum <- sum(layer_outputs)  // X_init + X_1 + ... + X_L
7.  Y_all <- OutputConv(X_sum)
8.  Y_hat <- Y_all[:, 0, -1, :]  // slice final time position
9.  Return Y_hat
```
