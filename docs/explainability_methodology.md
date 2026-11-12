# Explainability Methodology & Experimental Design

**Module:** [`src/explainability/`](../src/explainability/)  
**Evaluation Script:** [`src/explainability/experiments.py`](../src/explainability/experiments.py)  
**Test Suite:** [`src/test_explainability.py`](../src/test_explainability.py) (5/5 tests passing)  
**Status:** Completed and Validated

---

## 1. Core Principles & Design Stance

In accordance with Phase 6 requirements:
1. **Separation from Inference:** All explainability methods are implemented as post-hoc diagnostic modules that **do not alter the model's native forward pass**.
2. **Beyond Attention Weights:** Raw attention weights or fusion coefficients ($\alpha_{\text{ST}}, \alpha_{\text{DS}}, \alpha_{\text{DT}}$) are treated purely as model parameters, not faithful explanations of individual predictions. Instance-level importance is quantified strictly through **axiomatic path integrals (Integrated Gradients)** and **counterfactual interventions**.
3. **No Causal Overclaims:** Attribution scores indicate model sensitivity and feature reliance, not economic causality or market drivers.

---

## 2. Explanation Family A: Integrated Gradients Attribution

### 2.1 Formulation
Implemented in [`src/explainability/attribution.py`](../src/explainability/attribution.py). For an input tensor $X \in \mathbb{R}^{1 \times F \times T \times N}$ and static adjacency $A \in \mathbb{R}^{N \times N}$, we attribute the pre-sigmoid logit $\hat{y}_i$ of target stock $i$ along a straight-line interpolation from a baseline $X_0$:

$$\text{IG}_{f, t, n}(X) = (X_{f, t, n} - X_{0, f, t, n}) \times \int_{0}^1 \frac{\partial F_i(X_0 + \alpha (X - X_0), A)}{\partial X_{f, t, n}} d\alpha$$

### 2.2 Numerical Implementation
- **Baseline ($X_0$):** Zero tensor in normalized feature space ($\mu = 0, \sigma = 1$).
- **Riemann Sum Discretization:** Approximated over $m = 30$ interpolation steps:
  $$\text{IG}_{f, t, n} \approx (X_{f, t, n} - X_{0, f, t, n}) \times \frac{1}{m} \sum_{k=1}^m \left. \frac{\partial F_i(\tilde{X}_k, A)}{\partial X_{f, t, n}} \right|_{\tilde{X}_k = X_0 + \frac{k}{m}(X - X_0)}$$
- **Completeness Axiom Verification:**
  $$\sum_{f=1}^F \sum_{t=1}^T \sum_{n=1}^N \text{IG}_{f, t, n} \approx F_i(X, A) - F_i(X_0, A) = \Delta \text{logit}$$
  Every attribution call checks $\Delta_{\text{conv}} = |\sum \text{IG} - \Delta \text{logit}|$, guaranteeing mathematical conservatism.

### 2.3 Attribution Aggregations
- **Feature Importance:** $I(f) = \sum_{t=1}^T \sum_{n=1}^N |\text{IG}_{f, t, n}|$ (and signed $\sum \text{IG}$).
- **Temporal Saliency:** $T(t) = \sum_{f=1}^F \sum_{n=1}^N |\text{IG}_{f, t, n}|$ (capturing recency bias over trading days $0 \dots 29$).
- **Spatial Peer Influence:** $S(n) = \sum_{f=1}^F \sum_{t=1}^T |\text{IG}_{f, t, n}|$ (relative attribution of peer stocks $n \ne i$ vs target $i$).

---

## 3. Explanation Family B: MVCA Counterfactual View Interventions

Implemented in [`src/explainability/interventions.py`](../src/explainability/interventions.py).

The paper's MVCA module pools representations across three orthogonal planes:
1. Spatio-Temporal View ($\Psi_{\text{ST}}$)
2. Feature-Spatial View ($\Psi_{\text{DS}}$)
3. Feature-Temporal View ($\Psi_{\text{DT}}$)

### 3.1 Intervention Protocol
Using non-destructive forward hooks on `block.st_view`, `block.ds_view`, and `block.dt_view`:
- **Ablation Operator:** Intervene on view $V \in \{\text{ST}, \text{DS}, \text{DT}\}$ by setting $\Psi_V \to 0$ (neutral baseline) during inference.
- **Logit Sensitivity:** Measure the instance-level impact on target stock $i$:
  $$\Delta \text{logit}_V = \text{logit}_{\text{clean}} - \text{logit}_{\text{ablated}(V)}$$
- **Distinguishability Test:** Compare rank order of $\{\Delta \text{logit}_{\text{ST}}, \Delta \text{logit}_{\text{DS}}, \Delta \text{logit}_{\text{DT}}\}$ across different samples and market regimes to verify whether views provide distinct information.

---

## 4. Explanation Family C: Graph Channel & Neighbor Interventions

Implemented in [`src/explainability/interventions.py`](../src/explainability/interventions.py).

The Bidirectional Diffusion Graph Convolution (BDGC) fuses:
1. **Static Prior Graph ($A_{\text{norm}}$):** Derived from historical macro-trend clustering.
2. **Adaptive Learned Graph ($A_{\text{adp}}$):** Parameterized by dynamic node embeddings $E_1 E_2^T$.

### 4.1 Channel Disentanglement Interventions
- **Static Prior Ablation:** Replace $A_{\text{norm}}$ with the identity matrix $I_N$ (eliminating cross-stock static diffusion while preserving self-loops):
  $$\Delta \text{logit}_{\text{static}} = \text{logit}(X, A_{\text{norm}}) - \text{logit}(X, I_N)$$
- **Adaptive Graph Ablation:** Register a hook on `block.diffusion_conv.adaptive_adj` to force $A_{\text{adp}} \to 0$:
  $$\Delta \text{logit}_{\text{adapt}} = \text{logit}(X, A_{\text{norm}}) - \text{logit}(X, A_{\text{norm}} \mid A_{\text{adp}}=0)$$
- **Isolated Time Series Baseline:** Both static prior and adaptive graph ablated simultaneously.

### 4.2 Peer Stock Node Masking
For target stock $i$ and each peer $j \ne i$:
- Mask peer stock $j$'s input features: $X_{:, :, :, j} \to 0$.
- Mask graph edges: $A_{i, j} \to 0, A_{j, i} \to 0$.
- Measure peer influence: $\Delta \text{logit}_{i \leftarrow j} = |\text{logit}_i - \text{logit}_{i \mid j\text{-masked}}|$.

---

## 5. Quantitative Faithfulness Testing Protocols

Implemented in [`src/explainability/faithfulness.py`](../src/explainability/faithfulness.py).

### 5.1 Deletion & Insertion Curves (AOPC)
- **Top-$k$ Deletion:** Progressively zero out features in descending order of IG attribution ($k \in \{1, 2, 4, 8, 12, 16, 20\}$). A faithful explanation causes a rapid drop in model confidence.
- **Random Deletion Control:** Average of 5 random feature subsets of size $k$.
- **Bottom-$k$ Deletion Control:** Zero out lowest-attributed features.
- **Faithfulness Ratio:**
  $$\text{Ratio} = \frac{\text{AUC}_{\text{Top Deletion}}}{\text{AUC}_{\text{Random Deletion}}}$$
  $\text{Ratio} > 1.0$ indicates that the explanation correctly identifies features that drive model predictions.

### 5.2 Model Parameter Randomization Sanity Check (Adebayo et al., 2018)
- Progressively randomize weights of the output head and top ST-Block.
- Recompute Integrated Gradients attribution on the randomized model.
- Measure Spearman rank correlation and cosine similarity between true attributions and randomized attributions.
- **Criteria:** If the attribution relies on learned weights, the rank correlation must collapse towards zero ($\rho < 0.50$). If correlation remains high ($> 0.8$), the method acts merely as an input edge detector.

---

## 6. Stability Evaluation Protocols

Implemented in [`src/explainability/stability.py`](../src/explainability/stability.py).

### 6.1 Cross-Seed Stability
- Evaluate attribution consistency across models independently trained on seeds `42, 43, 44, 45, 46`.
- Compute pairwise Spearman rank correlations and Top-5 feature set Jaccard overlap:
  $$J(\mathcal{S}_a, \mathcal{S}_b) = \frac{|\mathcal{S}_a \cap \mathcal{S}_b|}{|\mathcal{S}_a \cup \mathcal{S}_b|}$$

### 6.2 Input Perturbation Robustness
- Inject Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.02$ into the input $X$.
- Filter for prediction-preserving trials ($\text{sign}(\hat{y}_{\text{noisy}}) == \text{sign}(\hat{y}_{\text{orig}})$).
- Measure cosine similarity and rank correlation between clean and noisy attributions.
