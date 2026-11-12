# ST-GRAD: Multi-view cross-attention and bidirectional diffusion for stock trend prediction

**Authors:** Yu Fu, Zhengming Huang, Bin Yu  
**Journal:** *Expert Systems With Applications*, 330 (2026), article 133046  
**DOI:** https://doi.org/10.1016/j.eswa.2026.133046  
**Source:** User-supplied 21-page PDF  
**Document type:** Source-faithful Markdown transcription for reading and agent-assisted implementation

## How to use this Markdown

This file preserves the article's organization, prose, equation numbering, algorithm descriptions, table data, table/figure captions, and references. The PDF's raster figure artwork is not embedded; figure captions and page references are retained. Tables have been reconstructed as Markdown tables to improve machine readability. Equations are transcribed into plain mathematical notation where needed; when a symbol or expression is ambiguous in the PDF extraction, do not treat this file as more authoritative than the original PDF page image. For reproduction, cite the original paper section/equation/table and record any interpretation as an explicit implementation decision. No new scientific claims have been added to the paper text.

## Abstract

Stock market prediction is a complex and challenging task in quantitative finance due to the highly noisy, non-stationary, and multi-dimensional nature of financial data. While existing spatio-temporal graph neural networks (GNNs) have made progress in capturing spatial correlations between assets, they typically rely on a single relational perspective. This limitation prevents them from fully capturing the complex, intertwined dependencies among time, assets, and market features, thereby restricting their predictive power in dynamic market environments. To overcome these limitations, we propose ST-GRAD (Spatio-Temporal Graph Adaptive Diffusion). This model introduces two key innovations. First, a Multi-View Cross-Attention (MVCA) mechanism explicitly disentangles multi-dimensional dependencies into three orthogonal views: spatio-temporal, feature-spatio, and feature-temporal. Second, a Bidirectional Diffusion Graph Convolution (BDGC) architecture fuses a static prior graph (derived from deep behavioral clustering) with a dynamically learned adaptive graph. This hybrid design enables the model to capture stable long-term sector structures and short-term market dynamics. The authors report experiments across five industry sectors and claim improvements over several baseline models. The paper presents ST-GRAD as a framework for stock-trend prediction and decision support.

**Keywords:** Stock trend forecasting; spatio-temporal graph neural network; cross-attention; ST-Block; adaptive graph convolution.

---

## Full paper text

## 1. Introduction

Stock market price prediction has long been a key focus in both academia and industry. However, the high noise, non-stationarity, and complex multicollinearity of financial data pose significant forecasting challenges. Essentially, market price formation is a complex spatiotemporal process.

It is driven simultaneously by the historical evolution of individual assets over time, and the network propagation of shocks across different assets in space. The evolution of predictive methodologies reflects an increasingly deep understanding of market complexity. Early research primarily focused on the statistical modeling of univariate time series.

Box et al. (1970) introduced the ARIMA model to capture the autocorrelation and moving average components of time series, establishing the standard paradigm of using historical prices to forecast future returns. To account for “volatility clustering” in financial data, Engle (1982) developed the ARCH model, which Bollerslev (1986) later refined into the GARCH model by incorporating a lagged conditional variance term.

Because these early models failed to capture the market’s “leverage effect”, Nelson (1991) proposed the EGARCH model. By using a logarithmic variance equation and an asymmetric response function, EGARCH successfully captures the differing impacts of positive and negative shocks, opening a new direction in volatility modeling. Despite these advancements in temporal modeling, early research typically treated each stock as an independent time series, relying solely on the historical data of the target asset.

While this univariate approach laid the foundation for forecasting, its assumption of independence fundamentally contradicts market reality. As established by the Capital Asset Pricing Model (CAPM) (Sharpe, 1964) and Arbitrage Pricing Theory (APT) (Ross, 1976), asset returns are driven by common risk factors. Consequently, stocks form a highly interconnected network, bound by shared industry affiliations and supply chain relationships.

Ignoring these spatial dependencies in cross-sectional data can lead to the omission of critical predictive information, significantly reducing the accuracy of the model’s forecasts. To address these limitations, predictive methodologies have evolved in two main directions. First, to capture non-linear relationships, researchers introduced machine learning models.

For instance, the

Support Vector Machines (SVM) (Tay & Cao, 2001) uses kernel functions to map data to higher-dimensional feature space, effectively solving nonlinear separability issues. Random Forest (RF) (Ballings et al., 2015; Breiman, 2001) mitigates overfitting by integrating multiple decision trees via Bootstrap Aggregating, while also providing feature importance rankings based on the reduction in Gini impurity. On this basis, Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs) further enhance the modeling ability of machine learning models for time series.

RNNs capture historical information through hidden states but struggle with long-term dependencies due to vanishing gradients (Hochreiter & Schmidhuber, 1997). To address this, the LSTM network (Hochreiter & Schmidhuber, 1997) controls information flow using specialized gating mechanisms, significantly improving long-term dependency modeling. Second, while machine learning models excel in univariate forecasting, they have not yet fully addressed the challenge of modeling the “spatio dimension”.

Most studies continue to independently model individual stocks, using various technical indicators as input features, without effectively utilizing the cross-sectional correlations existing in the market. Here, Graph Neural Networks (GNNs) (Kipf, 2016) and their applications in finance (Cheng et al., 2022) offer a key breakthrough. GNNs represent financial markets as graph structures, where nodes represent assets and edges define relationships (such as industry classification, yield correlation).

By aggregating information from neighboring nodes, they explicitly model cross-sectional dependencies between assets. Ultimately, the fusion of time series deep learning and graph neural networks has given rise to cutting-edge technology in stock prediction: Spatial Temporal Graph Neural Networks (STGNNs). The STGNN framework aims to unify the joint learning of temporal dynamics and spatial correlations.

Temporal modules (e.g., LSTM, TCN) capture the historical evolution patterns of individual nodes (e.g., the price trends of individual stocks). Concurrently, spatial modules (e.g., GCN, GAT) capture the mutual influence and dynamic diffusion between nodes such as volatility transmission across sectors. For example, the DCRNN model (Yu et al., 2017) innovatively combines diffusion convolution with recurrent neural networks.

Wu et al. (2020b) successfully applied this architecture to financial markets. Their experiments demonstrated a more than 20% improvement in prediction accuracy over traditional univariate models, strongly validating the potential of STGNNs.

STGNNs represent a significant leap forward in capturing spatiotemporal dependencies. Nevertheless, applying them to complex financial markets reveals three major challenges. (1) Model bulkiness and inefficiency.

Many advanced models incur high computational costs while pursuing performance benchmarks. For example, STGODE (Fang et al., 2021) introduces a complex ordinary differential equation (ODE) solver, while models like AASTHGCN (Su et al., 2024) rely on computationally expensive attention mechanisms. These computationally heavy methods lead to parameter explosion, as shown in the experiments of Section 6.6, making them challenging to deploy in real financial scenarios that demand quick signal generation and market analysis.

(2) Singleperspective modeling. While recent studies explore multiple relations (He et al., 2024; Li et al., 2025; Qian et al., 2024), most models still predominantly focus on a single perspective: spatio-temporal interaction. They often overlook the explicit decoupling of orthogonal relationships, such as feature-asset interactions (feature-spatio) or market consensus behaviors (feature-temporal).

This limitation omits key predictive signals. (3) Rigid graph structures. Spatial modeling often falls into two traps.

Predefined static graphs are too rigid to capture unforeseen market dynamics. Conversely, purely data-driven adaptive graphs are vulnerable to noise and ignore stable structural priors (Cui et al., 2022; Wu et al., 2020b, 2019). Even with recent 2024 and 2025 advancements in sophisticated dynamic graph adaptation (Lu et al., 2025; Zheng & Xie, 2025), purely data-driven architectures still struggle with extreme market non-stationarity.

Therefore, a robust solution must seamlessly combine the strengths of both approaches, providing dynamic adaptability without sacrificing structural stability. To address the above challenges, we propose ST-GRAD, a novel spatio-temporal graph network. At its core, the model employs a Multi- View Cross-Attention (MVCA) mechanism to extract market information across three hierarchical levels: micro (individual assets), meso (featureasset interactions), and macro (market consensus).

Subsequently, a dual graph convolution structure integrates static structural priors with dynamically learned graphs to capture complex asset correlations. Ultimately, the model generates robust predictions through hierarchical feature fusion. The main contributions of this paper are summarized as follows: (1) We propose the MVCA mechanism, a computationally efficient paradigm transcending single spatio-temporal modeling.

By explicitly decoupling financial data into three orthogonal views (spatio-temporal, feature-spatio, and feature-temporal), MVCA achieves a fine-grained decomposition of market dynamics with minimal overhead. This directly addresses the inherent parameter complexity and the lack of explicit feature decoupling found in recent high-capacity models such as AASTHGCN and MDGNN. (2) We introduce the BDGC mechanism.

It synergistically fuses a Deep Behavioral Trend Clustering prior with a dynamically learned adaptive graph. This hybrid approach provides a robust structural anchor, overcoming the structural rigidity of traditional static-graph methods and the noise sensitivity of purely adaptive architectures, such as STGODE and GraphWaveNet, respectively. (3) By stacking the designed ST-Block and employing a hierarchical fusion strategy, our model integrates multi-scale features from low-level local patterns to high-level global trends.

This ensures robust predictions based on a comprehensive feature set, which is crucial for navigating the inherently low signal-to-noise ratio of financial markets. The structure of this paper is organized as follows: Section 2 introduces preliminary concepts. Section 3 outlines the overall framework of ST-GRAD.

Section 4 details the MVCA mechanism. Section 5 presents the spatio-temporal propagation, encompassing the BDGC architecture and hierarchical fusion strategy. Finally, Section 6 provides comprehensive empirical evaluations and concludes with an in-depth discussion on model interpretability and practical implications.

## 2. Preliminaries

This section introduces the core concepts that form the foundation of our model, including graph neural networks (GNNs) and attention mechanisms. Based on these concepts, we will formally define the graph construction method and the stock price trend prediction problem.

### 2.1. Graph neural networks

Graph neural networks (GNNs) are a class of deep learning models specifically designed for processing graph-structured data. The core idea is to learn node representations through a message-passing mechanism. At each layer, a node aggregates the feature information of its neighboring nodes and combines it with its original features to update its representation through a non-linear transformation. By stacking multiple layers of GNNs, nodes can capture information from increasingly distant neighbors, enabling the model to learn both the graph’s topology and the contextual features of the nodes (Kipf, 2016; Wu et al., 2020a).

### 2.2. Attention mechanism

The attention mechanism is inspired by human cognitive attention, allowing the model to dynamically and selectively focus on more important parts when processing input sequences or feature sets. The core idea is to generate attention weights by calculating the similarity between a query and a set of keys, then apply these weights to the corresponding

values (Vaswani et al., 2017). Since its introduction in the Transformer architecture, the attention mechanism has become a powerful tool for processing sequence data and capturing long-range dependencies.

### 2.3. Graph construction

To capture the complex interrelations among assets in the stock market, we model a set of stocks and their relationships as a graph G = (V , E, A), where V = {v1, v2, ... , vN } represents the nodes corresponding to N stocks, E denotes the edge set representing the relationships between stocks, and A ∈ RN×N is the adjacency matrix describing the connection strength between nodes. Our model utilizes both static graphs based on prior knowledge and data-driven dynamic graphs. Traditional static adjacency matrices are typically based on fixed industry classifications, but this approach may fail to capture correlations among stocks that exhibit similar behavioral patterns across industries.

To construct a prior graph structure that better reflects the intrinsic dynamic similarity of stocks, we draw on the concept of time series clustering based on deep features. The core idea is to first use deep learning models to extract high-dimensional features from each stock’s time series, then cluster the stocks based on the similarity of these features, and finally define the connection relationships between stocks according to the clustering results. The specific steps are as follows:

- Macro-Trend Profiling: Rather than relying solely on raw prices or rigid industry labels, we comprehensively profile the macroscopic market dynamics of each asset. To achieve this, we extract a combined CNN-LSTM representation and explicitly concatenate it with five statistical trend indicators: sequence mean, volatility (standard deviation), linear trend slope, overall period return, and peak-topeak price amplitude.

- Behavioral Trend Clustering: To prevent higher-dimensional deep features from dominating the distance metrics, the combined feature vectors are scaled using Z-score standardization. Subsequently, we apply the K-Means clustering algorithm. We set the number of clusters to K = 5 to naturally align with the structural baseline of the five predefined macroeconomic sectors. This dynamic partitioning strictly maps assets based on their behavioral trajectory and volatility coherence, effectively identifying latent “concept sectors”.

- Adjacency Matrix Construction and Normalization: Based on the trend clustering results, we construct the initial adjacency matrix A. For any two stocks i and j, Aij = 1 if they belong to the same cluster (including i = j), and Aij = 0 otherwise. To ensure stable gradient propagation during graph convolution, we apply symmetric normalization to derive the final prior adjacency matrix Anorm: Anorm = D− 1 2 AD− 1 2 (1) where D is the degree matrix of A. This data-driven Anorm serves as the structural anchor for the subsequent BDGC module. In this way, we construct an adjacency matrix based on data-driven clustering results. After normalization, the matrix is obtained as the prior adjacency matrix Anorm, which is then provided as input to the subsequent graph convolutional network.

### 2.4. Problem formulation

Based on the graph structure, we formalize the stock price trend prediction task as follows. All stock attributes over a period of time are uniformly represented as a three-dimensional tensor X ∈ RF×T ×N , where F represents the number of node attribute features, including stock closing price, trading volume, RSI, VWAP, and other parameters. T represents the length of the historical stock time series used for model training, i.e., the size of the backtracking window, which determines the range of historical information the model can capture.

N represents the number of nodes in the stock network, which is the total number of stocks. In this case, the cross-attention view used to connect the feature dimension F with the time dimension T can be precisely defined as: ΨD⇌T = f(X) ∈ RD×T , (2) where f(X) is a mapping function that characterizes the bidirectional dependency relationship between the feature dimension D and the time dimension T in the 3D stock tensor X. Furthermore, the same form of mapping function can be applied to construct the feature-spatio view ΨD⇌S and the spatio-temporal view ΨS⇌T .

Through this mechanism, the ST-GRAD model effectively reveals the nonlinear synergy between features, time series, and nodes, allowing information across different dimensions to interact and be weighted in parallel. This significantly improves the accuracy of stock price trend prediction. To ensure clarity and reproducibility of the methodology, the symbols for all vectors, constants, and functions used throughout the text are summarized in Table 1.

## 3. The ST-GRAD framework: overall architecture

ST-GRAD is a graph neural network model specifically designed for spatio-temporal data to predict stock price trends. Its overall architecture is shown in Fig. 1, adopting an end-to-end design composed of multiple stacked ST-Blocks. This modular design allows the model to progressively and hierarchically extract increasingly complex features from the data.

Before entering the core of the ST-Block, the input data first passes through an initial feature transformation layer. This layer is responsible for data standardization, dimensional mapping, and capturing preliminary time patterns, laying a solid foundation for subsequent deeper multi-dimensional interaction modeling. The data then flows through L ST-Blocks, each refining and integrating information from different dimensions.

Ultimately, all outputs of the ST-Block are integrated through a hierarchical fusion mechanism and sent to an output module to generate the final prediction. This section focuses on the preprocessing process applied to the original input data, including standardization, feature dimension mapping, and capturing temporal sequence dependencies of each stock through the initial time attention mechanism. These steps prepare the data for the core multi-dimensional interaction modeling detailed in Section 4.

To unify the scale of the original data, we standardize the input data X. The standardization formula is as follows: Xin = X − μ σstd , (3) where μ and σstd are the mean and standard deviation of the input data X. The standardization process in Eq.

(3) effectively avoids information leakage in time series data. To further improve the robustness of the model, we introduce the BatchNorm layer in the architecture, with its calculation process shown in Eq. (4): Xnorm = BatchNorm(Xin).

(4) This design has three advantages: (1) it stabilizes the parameter distribution during training, (2) it enhances the generalization ability of the model, and (3) it improves the gradient flow characteristics, thereby accelerating convergence. As shown in Eq. (5), the standardized data undergoes feature projection through a 2D convolution: Xinit = Conv2dinit(Xnorm) ∈ RB×D×T ×N .

(5) Here, Conv2dinit is a 2D convolution with a kernel size of 1 × 1, which linearly maps the original F-dimensional features to a D-dimensional latent space, generating Xinit as the input to the spatio-temporal block of the ST-GRAD core component. Given the strong temporal dependence of stock data, we apply a temporal attention mechanism to the input data Xinit in the spatio-temporal

### Table 1. Notation summary

| Symbol | Description |
|---|---|
| `G` | Graph structure of the stock market |
| `N`, `S` | Number of stocks (`S` denotes the spatial dimension) |
| `E` | Edge set representing associations between stocks |
| `V` | Set of nodes corresponding to stocks |
| `F` | Number of original input characteristics per stock |
| `D` | Feature dimension of the latent space |
| `T` | Length of the historical time-series window |
| `B` | Batch size (32 in this study) |
| `A` | General adjacency matrix |
| `K` | Number of clusters |
| `A_norm` | Predefined and normalized prior matrix |
| `A_adp` | Adaptive adjacency matrix |
| `X` | Input feature tensor for all stocks |
| `X^(k)` | Intermediate hidden feature tensor output from the k-th ST-Block |
| `mu`, `sigma_std` | Mean and standard deviation of input data `X` |
| `sigma(.)` | Sigmoid activation function |
| `f(.)` | Core mapping function |
| `Psi_ST` | Spatio-temporal cross-attention view |
| `Psi_DS` | Feature-spatio cross-attention view |
| `Psi_DT` | Feature-temporal cross-attention view |
| `alpha_ST`, `alpha_DS`, `alpha_DT` | Learnable scale weights for view fusion |
| `w_ST`, `w_DS`, `w_DT` | Dynamic integration weights |
| `epsilon` | Small positive value |
| `E1`, `E2` | Learnable node-embedding matrices for generating `A_adp` |
| `d` | Node-embedding dimension (10 in this study) |
| `K_diff` | Number of diffusion steps in graph convolution |
| `L` | Number of ST-Blocks |
| `Y_hat` | Final forecast output |
| `T_base` | Basic threshold for stock price rise/fall |
| `k` | Regulatory/adjustment factor |
| `P_(i,t+1)` | Closing price of stock `i` on day `t+1` |
| `P_bar_(i,t)` | Average closing price of stock `i` over the past `t` days, as written in the paper |
| `R_(i,t+1)` | Next one-day yield of stock `i` relative to the average price |
| `y_(i,t)` | Rise/fall label for stock `i` |

Fig. 1. The ST-GRAD framework. (Xin ∈ RB×F×T ×N , X(k−1) ∈ RB×D×T ×N ,whereB = batch size, F = input features, T = time steps, N = nodes, D = hidden dimension. Ψi represents the generated cross-attention view, wi denotes the learned adaptive fusion weight, and ̂ X(k) is the fused multi-view feature tensor fed into the diffusion graph convolution network.) block to capture the key patterns along the time dimension.

Specifically, the Temporal Attention (TA) mechanism is realized by Eqs. (6)–(9). Xreshape = Re(Xinit) ∈ R(B×N)×T ×D , (6) where B represents the batch size during training, and Re(⋅) denotes the reshape operation.

This reshaping operation reorganizes the multidimensional input tensor so that the model can independently process the time series data of each stock in the batch. The reshaped tensor structure provides an ideal format for subsequent multi-head self-attention calculation. The multi-head self-attention mechanism computes attention weights over the time dimension T , capturing dependency relationships between different time steps while maintaining independent modeling for each stock.

Each attention head focuses on learning specific temporal patterns in the stock time series, and the results from each head are integrated through concatenation and linear transformation. The multihead attention mechanism is defined as: Attention(Q, κ, ν) = softmax ( Qκ⊤ √ dk ) ν. (7) Here, Q is a query vector, κ is a key vector, and ν is a value vector derived from the reshaped input tensor Xreshape, with dk representing the dimension of each attention head.

To enhance gradient flow and improve training stability, we introduce residual connections and layer normalization to the output of the multi-head self-attention module, as shown in Eq. (8): Xattn = LayerNorm ( Xreshape + Attention(Q, κ, ν) ) . (8) The residual connection alleviates the problem of vanishing gradients in deep networks, while layer normalization standardizes the feature dimensions, further enhancing convergence and generalization.

Finally, to restore the processed tensor to the original input structure for subsequent module calculations (e.g., feedforward neural networks), we perform a reshape operation on Xattn: Xtemp = Re(Xattn) ∈ RB×N×T ×D . (9) This operation adjusts the tensor from (B × N) × T × F to B × N × T × F, ensuring that the time series information for each stock is realigned in the batch dimension, providing structural consistency for subsequent cross-stock relationship modeling or prediction output. The result of the TA mechanism, Xtemp, is then connected with Xinit to form the input X for the subsequent attention view, while the original input Xinit is preserved for later use. This sequence of operations constitutes the essential preprocessing steps before the data enters the core ST-Block, ensuring that the input tensor contains rich preliminary time-dependent information for each stock.

## 4. Multi-view attention mechanism

This section details the core conceptual innovation of this framework–the MVCA mechanism. Traditional ST-GNNs typically process the time and spatial dimensions alternately, or model them together, which often fails to capture the more refined and complex interactions in financial data. The novelty of ST-GRAD lies in transforming feature dimensions from passive inputs into active participants.

It introduces three mutually orthogonal cross-attention views: ST, DS, and DT. The core concept behind this design is to explicitly disentangle the multi-dimensional dependencies embedded in financial data. Each view follows the “aggregation-transform-gate” design philosophy, analyzing data from a unique perspective that complements the others, together constructing a more comprehensive representation of market dynamics than any single perspective could.

### 4.1. The core concept of MVCA: structured decomposition of market dynamics

Traditional ST-GNNs often treat financial data as an integrated, entangled information block for modeling, which can lead to confusion among different market dynamics signals. Even recent advanced baselines like AASTHGCN (Su et al., 2024) and MDGNN (Qian et al., 2024), despite utilizing complex spatio-temporal or multi-relational attention, lack explicit orthogonal decoupling when handling feature interactions. This inevitably leads to increased parameter complexity.

To address these limitations, ST-GRAD introduces the MVCA mechanism to structurally decompose the input financial data. Rather than attempting to learn all relationships at once, the MVCA mechanism explores and answers different questions about the market from three fundamental and complementary perspectives: (1) Spatio-Temporal View: Focuses on horizontal comparisons among assets. It answers: “At a specific moment, how does the state of a particular stock compare to its peers?

Is there an inter-linkage effect within the sector?” (2) Feature-Spatio View: Focuses on the uniqueness of a single asset. It answers: “For this specific stock, which technical features (such as momentum, volatility) have shown stronger predictive power consistently over time?” (3) Feature- Temporal View: Focuses on the vertical consensus across the market. It answers: “At a specific point in time, is there a general consensus in the market regarding a certain type of feature (such as volume)?” Traditional ST-GNNs (right) usually adopt a sequential processing method, while the MVCA mechanism (left) collaboratively processes information via orthogonal parallel deconstruction (aggregating the D, T, and S dimensions, respectively).

The architecture comparison is shown in

Fig. 2. The MVCA mechanism formalizes this multi-angle exploration through its orthogonal design. Technically, this is achieved by forcing each view to aggregate information along a unique, mutually exclusive dimension.

This structural constraint ensures that each view captures a distinct, non-redundant stream of market dynamics information, thus avoiding overlap. To better understand the function of each view, the following table maps its core functions to corresponding financial scenarios: Table 2 clearly shows that the MVCA mechanism is not a “black-box operation”, but a meaningful and structured deconstruction of complex market dynamics. To illustrate this more vividly, imagine a scenario where an “industry leader’s favorable financial report” impacts the market.

Traditional ST-GNNs may only observe the synchronized rise of asset regions (spatio-temporal view ΨS⇌T ). In contrast, ST-GRAD’s MVCA performs a deeper multi-dimensional deconstruction: First, the featuretemporal view (ΨD⇌T ) detects that “volume” and “momentum” have become consensus features across the market, capturing the impact of macro events. Meanwhile, the feature-spatio view (ΨD⇌S ) has learned from historical data that “momentum” is a core predictive factor for stocks in this sector.

Finally, the spatio-temporal view (ΨS⇌T ) confirms sector linkage behavior between the stock and its leading counterpart. Therefore, ST-GRAD captures not just simple synchronization but a complex interaction that activates individual stock DNA (DS) through market consensus (DT), and is verified by sectoral linkage (ST), leading to more robust predictions. The following sections will detail how each view achieves these functions through technical methods.

### 4.2. The cross-attention view ΨS⇌T

Stock price fluctuations arise from complex interactions in both time and space. In space, the interlinkage effects among industry sectors are significant; in time, stock prices exhibit multi-scale dynamic patterns. To capture these intertwined dependencies, we introduce the ST crossattention view (ΨS⇌T ).

This view constructs an integrated state by aggregating all financial features at each spatio-temporal point, treating the stock dimension as the spatial dimension. For any spatio-temporal point, it is represented by a highdimensional feature. To efficiently extract its core statistical attributes without losing the spatio-temporal structure, we first perform a feature dimension reading operation.

The process can be formulated as follows: A(k) D = concat(max D (X(k−1) ), mean D (X(k−1) ), min D (X(k−1) )). (10) Here, the tensor X(k−1) ∈ RB×D×T ×N represents the output from the (k − 1)th layer. The operations max D (⋅), mean D (⋅), and min D (⋅) denote max pooling, mean pooling, and min pooling along the feature dimension D, respectively.

By concatenating these three statistics, we obtain the aggregated feature tensor A(k) D ∈ RB×3×T ×N . Next, we map the aggregated feature tensor A(k) D to an attention view ΨS⇌T that reflects the spatio-temporal interaction relationship. To achieve this, we employ a parameter-efficient “Up-Down Conv” (U-D Conv) convolutional structure.

Its mathematical definition is as follows: Ψ(k) S⇌T = ReLU(Conv2ddown(ReLU(Conv2dup(A(k) D )))). (11) The U-D Conv in Eq. (11) consists of an upsampling convolution (Conv2dup) followed by a downsampling convolution (Conv2ddown).

The mechanism works as follows: First, the three-channel aggregated feature A(k) D is mapped to a high-dimensional latent space via a 1 × 1 convolutional operation Conv2dup, expanding the output channels to a hyperparameter Clatent to enhance the model’s ability to capture complex nonlinear relationships among input statistics. Next, the latent features are compressed from Clatent dimensions to a single-channel output using a 1 × 1 convolution Conv2ddown, generating the attention weights for each spatio-temporal position. This bottleneck architecture– expanding and then compressing–enables the model to capture complex patterns in high-dimensional space and refine them into a concise, single-dimensional output.

The entire process uses ReLU as the activation function. The key idea here is to allow the model to learn an unbounded, non-negative attention score. A value greater than zero in ΨS⇌T indicates the importance of the corresponding spatio-temporal point, while a value of zero implies irrelevance.

Unlike traditional attention mechanisms that directly

Fig. 2. Comparison between the proposed multi-view cross-attention (MVCA) mechanism and the traditional ST-GNNs attention mechanism. The MVCA extracts spatio-temporal correlations via orthogonal parallel deconstruction, whereas traditional ST-GNNs rely on sequential processing. (X(k−1) ∈ RB×D×T ×N , where B = batch size, D = hidden dimension, T = time steps, and N = nodes. X′ DS and X.′′ DT denote the intermediate tensors after dimension rearrangement. Ψ represents the generated cross-attention view, w is the adaptive fusion weight, and ̂ X(k) is the final fused feature tensor.).

### Table 2. MVCA view functions and financial scenarios

| View | Aggregation dimension | Core function / scenario |
|---|---|---|
| `Psi_ST` | `D` | Identify plate/sector linkage and asset impacts |
| `Psi_DS` | `T` | Uncover predictive factors for individual stocks |
| `Psi_DT` | `S` | Detect market consensus and macro events |

generate normalized weights, this method maintains a wider dynamic range, making it more flexible in capturing subtle differences in signal intensity. Finally, after generating the attention view ΨS⇌T , we use it to guide the filtering of the original information stream. This is achieved through a two-step process: first, a refined extraction of the original input features, followed by gated modulation.

The formula is as follows: X(k) ST = tanh(Conv2d1×1(X(k−1) ) ⊙ σ(Ψ(k) S⇌T )). (12) Before gating, a 1 × 1 convolution operation is applied to compute weighted linear combinations along the feature dimension D. The results are compressed using the tanh(⋅) function to the range (−1, 1), forming a standardized score representing the strength of “bullish” or “bearish” sentiment.

Simultaneously, the unbounded attention score Ψ(k) S⇌T is fed into a Sigmoid activation function σ(⋅), which adaptively converts the attention score into gate weights within the (0, 1) interval. Points with higher scores receive weights closer to 1, while those with lower scores are weighted closer to 0. This gating mechanism, based on the separation of identification and application strategy—where ReLU first identifies “what is important” and Sigmoid then determines “how to use this importance” — gives the model significant flexibility.

The final gated output X(k) ST is a feature representation with higher information density, where key signals are amplified, and noise is suppressed. This refined representation provides a strong foundation for subsequent layers to make precise bullish/bearish predictions.

### 4.3. The cross-attention view ΨD⇌S

In the financial market, the relationship between different technological features and assets is complex and intricate. To explicitly model feature-spatio interactions, we design a cross-attention view (DS). The core idea is to extract the overall performance summary of each feature on each asset by aggregating information along the time dimension T , and then generate attention signals based on this summary.

To derive an overview of the feature-asset relationship, the model must analyze their behavior across the entire time horizon. However, a single global aggregation operation may smooth out key local dynamic information, such as short-term trend reversals or intensified volatility. To address this limitation, we draw inspiration from advanced time series analysis concepts: Multi-scale Modeling (Ismail Fawaz et al., 2020) and Sequence Patching (Zerveas et al., 2021).

These studies show that decomposing long sequences into multiple segments and analyzing them separately is an effective way to construct rich temporal representations. Based on this, we design a Blocked Temporal Readout Operation. The operation first rearranges the dimensions of the input tensor X(k−1) ∈ RB×D×T ×S to obtain X′(k−1) ∈ RB×T ×D×S , prioritizing the time dimension for analysis.

The time series is then uniformly partitioned into t non-overlapping patches, denoted as split(X′(k−1)) = [X′ , X′ , ... , X′ t ]. We apply a statistical transformation operation ΦS (⋅) to each subsequence, and concatenate the results along a new dimension: A(k) T = concat(ΦR(X′ ), ΦR(X′ ), ... , ΦR(X′ t )). (13) Here, for any X′ i , we have ΦR(X′ i ) = concat(max T (X′ i ), mean T (X′ i ), min T (X′ i )), performing pooling operations along the time dimension, similar to Eq.

(10). We then obtain an aggregated feature tensor A(k) T ∈ RB×(3t)×D×S . This blocked aggregation mechanism enables the model to attend to both local and global information, similar to how hierarchical convolutional networks capture multi-scale receptive fields (Wu et al., 2019).

It not

only summarizes the overall performance of each feature-asset pair across the entire time period but also captures its dynamic evolution across different stages. This mechanism provides a richer and more detailed foundation for generating high-fidelity attention views compared to a single global aggregation. Next, we map the aggregated feature tensor A(k) T to the feature space attention view Ψ(k) D⇌S .

We use a U-D Conv, similar to that in Eq. (11), as follows: Ψ(k) D⇌S = ReLU ( Conv2ddown ( ReLU ( Conv2dup ( A(k) T )))) . (14) Notably, the Conv2dup layer’s channel count is 3t to directly match the output dimension of the blocked readout operation.

The structure fuses and facilitates interactions among the statistical properties from various time stages in a high-dimensional latent space, ultimately extracting and outputting an unbounded attention score representing the intrinsic importance of each feature-asset pair. Finally, the generated attention view Ψ(k) D⇌S is used to filter the input information, which has undergone dimension rearrangement: X′(k) DS = tanh ( Conv2d1×1 ( X′(k−1) ) ⊙ σ ( Ψ(k) D⇌S )) . (15) This equation is similar to the gating mechanism of the spatio-temporal view.

The resulting gated tensor X′(k) DS is then reordered back into the initial (B × D × T × S) dimensional sequence, generating an enhanced feature representation that emphasizes key feature-asset interactions.

### 4.4. The cross-attention view ΨD⇌T

In financial markets, beyond the spatial correlation among assets and interactions among features, there is a strong market consensus effect. Specifically, at a given time, certain features across many assets exhibit similar behavioral patterns. To capture this cross-asset, time-evolving feature-temporal relationship, we design the DT view.

The core idea is to aggregate information from the spatial dimension (S) to extract the overall market state of each feature at each time point. To evaluate the “market consensus” state of each feature at a specific time, we examine its collective performance across all assets. We achieve this through a Global Readout Operation over the spatial dimension.

The operation starts by rearranging the dimensions of the input tensor X(k−1) to obtain X′′(k−1) ∈ RB×S×D×T , which prioritizes the asset dimension. Next, we apply a statistical transformation operation ΦR(⋅), similar to Eq. (10), over all assets.

The transformation is as follows: A(k) S = ΦR(X′′(k−1) ) = concat ( max S ( X′′(k−1) ) , mean S ( X′′(k−1) ) , min S ( X′′(k−1) )) . (16) This yields an aggregated feature tensor A(k) S , which represents the “market consensus”. Next, we map this aggregated tensor to the Feature-Temporal Attention View Ψ(k) D⇌T using the same U-D Conv structure as in Eq.

(11): Ψ(k) D⇌T = ReLU ( Conv2ddown ( ReLU ( Conv2dup ( A(k) S )))) . (17) X′′(k) DT = tanh ( Conv2d1×1 ( X′′(k−1) ) ⊙ σ ( Ψ(k) D⇌T )) . (18) Finally, the attention view Ψ(k) D⇌T modulates the input tensor that has undergone corresponding dimension rearrangement via a gating mechanism, generating an enhanced feature representation that highlights key market consensus moments.

### 4.5. Formal characterization of view orthogonality and complementarity

A primary objective of the MVCA mechanism is to minimize information redundancy by capturing data dependencies across distinct dimensions. This structural decoupling is intended to encourage the model to learn complementary feature representations. Architecturally, the three views are structurally orthogonal because each aggregates a different dimension: ΨS⇌T focuses on the Spatio- Temporal plane by aggregating the Feature dimension (D), ΨD⇌S focuses on the Feature-spatio plane by aggregating the Time dimension (T ), and ΨD⇌T focuses on the Feature-Temporal plane by aggregating the Spatial dimension (S).

Since the aggregation dimensions are mutually orthogonal, the information source for each view is fundamentally distinct, ensuring complementary learning. To quantify this property, we define two metrics evaluated on the flattened output feature tensors Zv1 and Zv2 from any two distinct views v1, v2 ∈ {ΨS⇌T , ΨD⇌S , ΨD⇌T }: 1. Cosine Similarity: We assess the linear independence between views using the Cosine Similarity (cos): cos(Zv1 , Zv2 ) = |Zv1 ⋅ Zv2 | ‖Zv1 ‖2‖Zv2 ‖2 (19) where cos → 0 indicates that the representations learned by the views are nearly mathematically orthogonal in the linear domain. 2.

Mutual Information: From an information-theoretic perspective, ideal orthogonality dictates that the Mutual Information (MI) between any two views should be minimal. To evaluate non-linear redundancy, we utilize the Estimated Mutual Information () (Sanchez et al., 2020): (Zv1 ; Zv2 ) = ∑ z1∈Zv1 ∑ z2∈Zv2 p(z1, z2) log ( p(z1, z2) p(z1)p(z2) ) (20) where p(z1, z2) is the joint probability distribution of the feature representations, and p(z1) and p(z2) denote their respective marginal distributions (estimated via discretization of the feature space). A low  indicates that the information contained in one view provides minimal predictive power regarding another, suggesting the capture of different latent factors.

This formal framework is empirically evaluated in Section 6.7.

### 4.6. Adaptive view fusion

This section explains how the model adaptively weights and fuses the outputs from the three cross-attention views by introducing learnable parameters, allowing it to dynamically adjust the contribution of different dimensions to the final prediction. The three orthogonal views capture the market’s complex dynamics from different but complementary perspectives. However, their contribution to the final prediction is not fixed and can change depending on the market environment.

For instance, in a stable trending market, the spatio-temporal patterns of individual stocks (ΨS⇌T ) may dominate. Conversely, on trading days when macroeconomic data is released, the importance of market consensus behavior (ΨD⇌T ) may increase significantly. To allow the model to learn the appropriate weights for these views, we introduce an adaptive fusion mechanism.

Specifically, we define three learnable scalar parameters, αST , αDS , and αDT , which represent the initial importance of each view. These parameters are normalized using the Softmax function to derive dynamic fusion weights, w. The outputs of the three views are then combined according to these weights to form the final fused feature tensor ̂ X(k): w = softmax([αST , αDS , αDT ]), (21) ̂ X(k) = wST X(k) ST + wDS X(k) DS + wDT X(k) DT .

(22) This mechanism enables the model to automatically emphasize views with richer information during training, based on feedback from the loss function, while reducing the influence of views with more noise or redundancy. This results in the most efficient utilization of multi-source information.

## 5. Spatio-temporal propagation and integration

After generating informative feature representations through the MVCA mechanisms, the model must explicitly model the spatial correlations between assets and integrate all operations into an efficient,

stackable module. This chapter introduces the components responsible for spatio-temporal information propagation and integration in the ST- GRAD framework, including BDGC, the architecture of the ST-Block, and the hierarchical fusion strategy used for final prediction.

### 5.1. Bidirectional diffusion graph convolution

This section explains how the model explicitly models and propagates spatial correlation information among assets using BDGC, which operates over a composite graph structure that combines a static prior graph with a dynamic adaptive graph. The feature tensor ̂ X(k) after multi-view fusion contains rich multidimensional information but has not yet explicitly modeled the spatial correlation between assets. Instead of relying on the rigid predefined graphs used in STGODE (Fang et al., 2021) or the purely implicit adaptive graphs in GraphWaveNet (Wu et al., 2019), we address this by employing BDGC to capture information propagation across multi-hop neighborhoods.

The core idea behind BDGC is that information diffuses not only from a node to its neighbors (forward diffusion) but also back from the neighbors (backward diffusion). The model iteratively models the influence of neighbors at varying distances (k-hop). This process is based on powers of the pre-defined adjacency matrix A, its transpose AT , and a datadriven adaptive adjacency matrix Aadp.

The adaptive adjacency matrix Aadp is generated by the inner product of two learnable node embedding matrices, E1, E2 ∈ RN×d, defined as follows: Aadp = softmax(ReLU(E1ET ) + ε), E1, E2 ∈ RN×d (23) Here, d is the embedding dimension, and ε is a small positive constant. During the forward pass of graph convolution, the model iterates for Kdiff steps. At each step j, the model performs forward diffusion (using Aj), backward diffusion (using (AT )j), and adaptive diffusion (using Aadp, but only when j = 0).

All diffusion results are multiplied by a shared weight matrix Wj and accumulated into the final output. This can be mathematically expressed as: X(k) gcn = Kdiff −1 ∑ j=0 ( (Aj + (AT )j ) ̂ X(j) Wj ) + Aadp ̂ X(j) W0 (24) Here, A0 and (AT )0 are identity matrices. This multi-step, bidirectional diffusion mechanism enables the model to capture long-range and complex inter-asset propagation effects, thereby significantly enhancing the depth and breadth of spatial modeling.

### 5.2. The ST-Block: an integrated module

The time attention, MVCA, adaptive fusion, and BDGC components described earlier form the core of our designed ST-Block. To efficiently integrate these components into a stackable and repeatable module while ensuring the stability of deep network training, the final stage of the ST-Block introduces a key residual connection and normalization mechanism. This architectural design, which encapsulates complex operations into reusable modules, enables the construction of deeper and more hierarchical models.

The residual connection within the module does more than just improve gradient flow; it plays a crucial functional role. It directly adds the module’s input Xinit to the output after a series of transformations, implying that the core task of each ST-Block is to “incrementally refine” or “polish” the input information, rather than learning a completely new representation from scratch. Specifically, the output X(k) gcn from the graph convolution undergoes a 1 × 1 convolution for channel transformation and feature integration, after which it is element-wise added to the original input of the module (Xinit, which is the input after initial feature projection): X(k) res = Xinit + Conv2d1×1(X(k) gcn) (25) The residual connection ensures that, even in deep networks, the fundamental information from the original input is preserved and not lost in multiple nonlinear transformations.

To further accelerate model convergence and reduce internal covariate shift, we apply layer normalization to the output of the residual connection. The normalized result is then passed through a ReLU activation function to obtain the final output X(k) out of the ST-Block. The formula is as follows: X(k) out = ReLU(LayerNorm(X(k) res)) (26) At this point, we have fully described the internal architecture of the ST-Block.

To provide a clear overview, we summarize the complete forward propagation process of the ST-Block, from receiving input tensors to generating the final output, in pseudocode form in Algorithm 1. This algorithm illustrates the synergistic effects of time attention, multi-view generation, adaptive fusion, graph convolution, and residual normalization. Algorithm 1: The ST-block method.

Input: Feature tensor from previous layer X(k−1) ∈ RB×D×T ×N ; Pre-defined adjacency matrix Anorm Output: ST-Block output tensor X(k) ∈ RB×D×T ×N 1 Initial Temporal Attention 2 Xtemp ← TemporalAttention(X(k−1)) 3 X ← X(k−1) + Xtemp 4 Xinit ← X // Save for final residual connection 5 Multi-View Cross-Attention Generation 6 X(k) ST ← ST-View(X) // Generate Spatio-Temporal View 7 X(k) DS ← DS-View(X) // Generate Feature-Spatio View 8 X(k) DT ← DT-View(X) // Generate Feature-Temporal View 9 Adaptive View Fusion & Graph Convolution 10 w(k) ← softmax([αST , αDS , αDT ]) // Fusion weights 11 ̂ X(k) ← w(k) ST ⋅ X(k) ST + w(k) DS ⋅ X(k) DS + w(k) DT ⋅ X(k) DT // Fused represetation 12 A(k) adp ← softmax(ReLU(E1ET )) // Adaptive adjacency 13 A(k) ← Anorm + A(k) adp // Composite adjacency 14 X(k) gcn ← 0 // Initialize output 15 for j ← 1; j ≤ Kdiff ; j + + do 16 X(k) gcn ← X(k) gcn + ((A(k))j + (A(k)T )j) ̂ X(k)Wj 17 Final Residual Connection and Normalization 18 X(k) res ← Xinit + Conv2d1×1(X(k) gcn) 19 X(k) ← ReLU(LayerNorm(X(k) res)) 20 return X(k)

### 5.3. Hierarchical fusion and final prediction

This section describes the final prediction generation process, explaining how multiple ST-Blocks are stacked to hierarchically extract features, and how a hierarchical fusion strategy is employed to integrate the outputs of all intermediate layers, thus forming a comprehensive, multi-scale feature representation for the final prediction. In the ST-GRAD architecture, we stack L ST-Blocks to perform a hierarchical, staged extraction of spatio-temporal features. This deep architecture allows the model to progressively build feature representations, ranging from local and simple to global and abstract.

However, in deep networks, information may be diluted or lost as it is transmitted layer by layer. For tasks with low signal-to-noise ratios, such as financial forecasting, the basic but potentially strong signals extracted at early layers are just as important as the abstract, more global signals from deeper layers. To ensure that the final prediction utilizes all levels of feature information, we adopt a hierarchical fusion strategy.

The hierarchical fusion strategy is achieved by summing up the outputs from all ST-Blocks. This dense skip-connection mechanism across all layers ensures that features from every level, from the lowest to the highest, directly contribute to the final prediction module: ̂ Y = OutputConv (L−1 ∑ k=0 X(k) out ) . (27) Here, X(0) out represents the initial input to the stacked ST-Blocks, Xinit, while the OutputConv is a decoder network composed of 1 × 1 convolutions, responsible for mapping the fused high-dimensional feature into the final prediction result. This strategy ensures that the model’s prediction is based on a comprehensive feature set, integrating patterns from different levels of abstraction and temporal scales, significantly enhancing both prediction accuracy and robustness. To illustrate the complete pipeline from initial data processing to final prediction generation, we summarize the forward pass of the ST- GRAD model in pseudocode in Algorithm 2.

**Algorithm 2. ST-GRAD method (transcribed pseudocode)**

```text
Input: X_in in R^(B x F x T x N); prior adjacency A_norm
Output: predictions Y_hat in R^(B x N)

1. X_norm <- BatchNorm2d(X_in) 2. X_init <- Conv2d_1x1(X_norm) // initial feature projection 3.

X^(0) <- X_init 4. layer_outputs <- [X^(0)] 5. For k = 1..L: X^(k) <- ST-Block_k(X^(k-1), A_norm) append X^(k) to layer_outputs 6. X_sum <- sum(layer_outputs[0..L]) 7.

X_out <- X^(L) + X_sum 8. Y_hat <- OutputConv(X_out) 9. return Y_hat ```

### 5.4. Complexity and convergence analysis

#### 5.4.1. Computational complexity analysis

The main computational overhead of the ST-GRAD model arises from the MVCA module and the BDGC module. (1) MVCA: The generation of each view involves pooling along one dimension and two 1 × 1 convolutions. For an input tensor of shape B × D × T × N, the complexity of this process is approximately O(B ⋅ D ⋅ T ⋅ N).

(2) BDGC: The computational complexity of the BDGC module is primarily determined by the graph convolution operations and the generation of the adaptive adjacency matrix. The adaptive adjacency matrix Aadp is calculated via the inner product E1ET , which requires O(N2 d) complexity. The graph convolution operation involves multiplying the adjacency matrix by the feature matrix, with complexity O(N2D).

Since this operation is executed at each time step and iterates for Kdiff steps, the total complexity becomes O(Kdiff ⋅ T ⋅ N2 ⋅ D). In summary, the computational complexity of ST-GRAD is divided into two main parts:

- The MVCA module, with a complexity of O(B ⋅ D ⋅ T ⋅ N).

- The graph convolution module, where the generation of the adaptive adjacency matrix Aadp introduces an O(N2 d) complexity. It should be noted that the O(N2) complexity is a common limitation shared by all advanced STGNNs that utilize adaptive adjacency matrices, not unique to ST-GRAD. The core efficiency innovation of ST-GRAD lies in its “component efficiency”: for a given number of nodes N, the MVCA mechanism is computationally lightweight. This contrasts sharply with the complex and expensive attention calculations in models like AASTHGCN, as quantitatively verified in the experiments.

#### 5.4.2. Convergence analysis

Providing rigorous theoretical convergence proofs for complex deep graph neural networks remains an open research problem. However, ST- GRAD ensures training stability and convergence in practice through the integration of several well-established architectural designs.

- Normalization Layers: The model extensively uses batch normalization and layer normalization. These techniques mitigate the issue of internal covariate shift by adjusting and scaling the distribution of activation values during training. This results in a smoother loss surface and accelerates the convergence of gradient descent.

- Residual Connections: Residual connections are employed both internally and in the final hierarchical fusion of the ST-Block. This design provides a “highway” for gradient backpropagation in deep networks, significantly reducing the problem of gradient vanishing and making it feasible to train deeper and more complex models.

- Learning Rate Scheduling: Learning rate schedulers, such as ReduceLROnPlateau, are used to automatically reduce the learning rate when the model reaches a plateau. This helps the model escape local optima and ensures finer convergence. The synergy of these components allows the ST-GRAD model to exhibit excellent stability during optimization, enabling it to converge effectively and learn complex patterns in the data. Fig. 3 illustrates the convergence curves for both the loss function and the F1 score on the training set. The smooth decrease in the loss function and the stable increase in the F1 score, without any sharp oscillations, directly demonstrate that the model’s design promotes smooth and reliable convergence.

### 5.5. Comparison with related work

To clearly identify the academic contributions of ST-GRAD and highlight its innovations, this section first outlines several representative design paradigms within the field of STGNNs and then deeply analyzes the core advantages of ST-GRAD in comparison to these paradigms.

#### 5.5.1. Overview of related model mechanisms

In the field of STGNNs, several advanced works have attempted to capture complex spatio-temporal dependencies from different perspectives. These works can be broadly classified into the following design paradigms:

- Continuous-Time Modeling Paradigm: Represented by models such as STGODE (Fang et al., 2021), this paradigm treats spatiotemporal dynamics as a continuous function described by Ordinary Differential Equations (ODEs). This approach is particularly suitable for modeling systems with smooth temporal changes.

- Implicit Spatio-Temporal Modeling Paradigm: Models like Graph- WaveNet (Wu et al., 2019) belong to this paradigm. These models use stacked dilated causal convolutions to capture temporal dependencies and implicitly learn spatial dependencies using an adaptive adjacency matrix, without relying on explicit attention mechanisms.

- Explicit Attention Modeling Paradigm: Represented by AASTHGCN (Su et al., 2024), this paradigm uses spatio-temporal attention mechanisms to explicitly capture dynamic spatio-temporal correlations. Attention modules help the model focus on relevant features and time steps when learning spatial and temporal dependencies separately.

Fig. 3. Training convergence curves: loss and F1 score evolution.

### Table 3. Comparison of related models

| Model | Attention mechanism | Graph diffusion / convolution mechanism |
|---|---|---|
| **ST-GRAD** | MVCA explicitly decomposes dependencies across three orthogonal views: ST, DS, DT. | BDGC performs multi-step bidirectional propagation on a composite static-prior and dynamic-adaptive graph. |
| **MDGNN (Qian et al., 2024)** | Multi-relational attention captures multifaceted stock relations, but does not explicitly orthogonally decouple feature interactions. | Dynamic relational graph adapts to multiple relations; modeling multiple relations can increase parameter complexity relative to static priors. |
| **AASTHGCN (Su et al., 2024)** | Spatial and temporal attention separately capture dynamic spatio-temporal correlations without decomposing feature dimensions. | Standard graph convolution on a predefined static graph. |
| **GraphWaveNet (Wu et al., 2019)** | No explicit attention; temporal dynamics are captured by stacked dilated causal convolutions, with spatial dependencies learned implicitly. | Graph convolution with an adaptive adjacency matrix learned from node embeddings. |
| **STGODE (Fang et al., 2021)** | No explicit attention; spatio-temporal dynamics are modeled as a continuous function using ODEs parameterized by a neural network. | Continuous-time graph convolution is integrated into the ODE and evolves node states over continuous time. |

- Multi-relational and Dynamic Adaptation Paradigm: Recently, modeling complex graph structures to capture intricate spatiotemporal dependencies has gained increasing attention (He et al., 2024).Works such as MDGNN (Qian et al., 2024), DGDNN (You et al., 2024), and COGRASP (Li et al., 2025) explore multi-relational, cooccurrence, and diffusion-based dependencies, which enhance representation but may increase parameter complexity. Simultaneously, cutting-edge dynamic models (Lu et al., 2025; Zheng & Xie, 2025) emphasize adaptive graph evolution to capture short-term shocks. ST-GRAD builds upon and extends the Explicit Attention Modeling paradigm. Furthermore, to overcome the limitations observed in the recent Multi-relational and Dynamic Adaptation Paradigm, ST-GRAD introduces two targeted innovations. First, to address parameter complexity, it utilizes MVCA for strictly orthogonal decoupling, achieving highly efficient multi-perspective modeling. Second, to prevent dynamic instability, it achieves an optimal balance between structural robustness and short-term sensitivity by hybridizing a deep-behavioral clustering prior (Anorm) with an adaptive diffusion matrix (Aadp). Table 3 summarizes the design principles of these representative models alongside ST-GRAD, focusing on two core components: spatio-temporal dependency modeling and the use of attention mechanisms.

#### 5.5.2. ST-GRAD’s comparative advantages and innovations

The design concept of ST-GRAD significantly differs from other advanced models.

- Compared with different paradigms: Unlike the continuous modeling of STGODE (Fang et al., 2021), ST-GRAD retains the discrete time step framework, avoiding the huge computational cost of the ODE solver in STGODE and significantly improving the inference efficiency. Compared to GraphWaveNet (Wu et al., 2019), which uses stacked dilated causal convolutions to capture temporal dynamics, ST-GRAD’s initial temporal attention and multi-view mechanism is a more efficient feature extraction method, capable of capturing rich dynamic information with fewer layers and parameters. Furthermore, when contrasted with recent multi-relational paradigms like MDGNN (Qian et al., 2024), ST-GRAD elegantly extracts multi-perspective signals through the strict orthogonal decoupling of MVCA, effectively avoiding the parameter inflation inherently associated with modeling multiple separate relation graphs.

- Compared with the same paradigm: The architectural differences between ST-GRAD and AASTHGCN highlight fundamentally different design philosophies. AASTHGCN achieves strong performance through the deep stacking of complex attention modules, which leads to a relatively large parameter count (4.84 million parameters). As

discussed in the restructured Section 6.6 (Table 10), this approach may encounter diminishing returns in efficiency. In contrast, the MVCA mechanism in ST-GRAD prioritizes structural efficiency. With its lightweight orthogonal view design, ST-GRAD captures a richer set of interactions (including feature-spatio and feature-temporal dimensions) with only 216,429 parameters–just 4.5% of AASTHGCN’s size. As confirmed in Table 8, ST-GRAD achieves superior performance, demonstrating that mathematically explicit feature decomposition can be highly effective without relying on sheer model scale. Moreover, ST-GRAD’s BDGC offers more comprehensive spatial modeling by hybridizing static priors with dynamic graphs.

- Enhancement of Graph Robustness: Recent dynamic models such as COGRASP (Li et al., 2025) achieve high efficiency using purely data-driven adaptive graphs. However, unconstrained graph learning is highly susceptible to overfitting in noisy financial data. Rather than relying solely on implicit relations, ST-GRAD restricts its adaptive diffusion (BDGC) using the aforementioned behavioral clustering prior. Grounding the dynamic propagation in this stable topology prevents the model from overreacting to short-term noise, resulting in significantly higher prediction accuracy (an F1 score of 0.8215 compared to 0.7626 for COGRASP).

- AASTHGCN as a contemporary benchmark: AASTHGCN is highly representative of recent trends that prioritize performance through expanded model capacity. Because it effectively establishes the stateof-the-art for its paradigm, it serves as the most rigorous benchmark for measuring ST-GRAD’s efficiency innovations. Therefore, we selected it as the core high-level baseline for quantitative performance and complexity comparisons in Section 6.

## 6. Experiments

To thoroughly evaluate the performance and internal mechanisms of the proposed ST-GRAD model, this study addresses the following key research questions:

- RQ1: Does the prior adjacency matrix construction based on deep features outperform the traditional adjacency matrix based on industry sectors and returns?

- RQ2: In the stock trend prediction task, does ST-GRAD significantly outperform various baseline models in terms of overall performance?

- RQ3: How much performance do the model’s key components contribute individually, and is there a synergistic relationship between them?

- RQ4: How sensitive is ST-GRAD’s performance to its key hyperparameters – the number of stacked ST-Blocks (L) and the number of diffusion steps (Kdiff )? What is the impact of different combinations of these parameters on the model’s predictions, and is there an optimal configuration?

- RQ5: How computationally efficient is ST-GRAD? How does its performance (e.g., F1) compare with baseline models in relation to computational complexity (e.g., Parameters)? The experimental design follows a structured, multi-step approach: first, we verify the effectiveness of the model’s input (RQ1); second, we evaluate ST-GRAD’s overall performance against various baseline models (RQ2); third, we explore the contributions of its core components via ablation experiments (RQ3); next, we investigate the sensitivity of the model to key hyperparameters and identify the optimal configuration (RQ4); finally, we assess the computational efficiency of ST-GRAD, balancing performance and complexity (RQ5). This comprehensive evaluation covers input validation, performance assessment, mechanism analysis, hyperparameter sensitivity, and efficiency, providing robust empirical evidence for the effectiveness of ST-GRAD.

### Table 4. Dataset stock universe

| Energy | Healthcare | Technology | Basic Materials | Industrials |
|---|---|---|---|---|
| CTRA | BIIB | AMKR | AXTA | ABM |
| DKL | CNC | ARW | IVPAF | ALG |
| KNTK | GMAB | CAJPY | LYSCF | HRI |
| MTDR | ICLR | HPE | LYSDY | HUBG |
| MUR | INCY | JKHY | MP | ODFL |
| NEXT | SAUHY | LFUS | RGLD | SBLK |
| NOG | SNN | NTAP | SOUHY | TEX |
| NOV | TEVA | PTC | SSD | TNET |
| PAGP | WAT | STM | SUZ | TRN |
| USAC | ZBH | VRSN | WFG | VWDRY |

### 6.1. Experimental setup

#### 6.1.1. Datasets and data preprocessing

The stock data used in this study was obtained from Yahoo Finance, covering 50 stocks from five sectors: energy, healthcare, technology, basic materials, and industrials. Specifically, we selected 10 stocks from each sector, with each stock containing five basic indicators: open, high, low, close, and volume. The dataset spans from August 19, 2020, to August 18, 2025, with each stock consisting of a total of 1255 data points.

The stock dataset is summarized in Table 4. To ensure the reproducibility and rigor required for a financial forecasting setting, our data preprocessing, feature engineering, and dataset splitting pipeline strictly follow the principles of stationarity and the absolute avoidance of look-ahead bias. The detailed procedures involve four key steps:

- Data Cleaning and Anomaly Neutralization: Financial time series inherently contain noise, such as missing values due to temporary trading suspensions and infinities caused by extreme market shocks. To maintain strict temporal continuity without leaking future information, we apply the forward-filling (ffill) method to impute missing prices. Furthermore, any infinite values (±∞) generated during feature calculations are neutralized to zero, and samples containing pervasive invalid data are masked to ensure gradient stability during deep learning optimization.

- Feature Engineering and Treatment of Non-stationarity: Feeding raw, non-stationary asset prices into neural networks often leads to spurious correlations. To explicitly address non-stationarity, our model strictly prohibits the input of raw absolute prices. Instead, we engineered 24 strictly stationary technical features from the basic data (summarized in Table 5). These encompass proportional metrics , momentum oscillators, and volatility indices, ensuring the fundamental stationarity of the input tensor.

- Chronological Splitting and Leakage-Free Normalization: The dataset is partitioned in strict chronological order without shuffling. Due to the 30-day sliding window requirement for feature construction and the 1-day prediction horizon, the initial 30 trading days serve as a “warm-up” period, and the final trading day is reserved as the ultimate prediction target. Consequently, the effective sample period strictly spans 1224 sequential samples. We split these samples chronologically: the first 70% (September 30, 2020, to February 26, 2024; 856 samples) is exclusively designated for training. The subsequent 30% (February 27, 2024, to August 15, 2025; 368 samples) is reserved for out-of-sample validation and testing (the detailed physical isolation protocol for the extreme stress window within this outof-sample period is discussed in Section 6.8). Crucially, to absolutely prevent look-ahead bias, the Z-score normalization parameters (μ and σstd) are computed exclusively on the 856 historical training samples and applied statically forward.

- Volatility-Adaptive Label Generation & Stochastic Smoothing: We define the prediction task as a binary classification problem.

### Table 5. Engineered technical features

The paper says `P_close_t`, `P_open_t`, `P_high_t`, `P_low_t`, and `V_t` come from the original stock data.

| Feature(s) | Calculation | Description / definitions given in paper |
|---|---|---|
| `f_rel_c` | `P_close_t / MA30(P_close)` | Relative close vs. 30-day simple moving average. `MA_k(.)` is the k-day simple moving average. |
| `f_ocr` | `(P_close_t - P_open_t) / P_open_t` | Open-close return ratio. |
| `f_vwap` | `(sum(i=0..29, TP_(t-i)) + V_t * P_close_t) / (sum(i=0..29, V_(t-i)) + V_t)` | Paper's displayed formula, transcribed literally. `TP_t = (P_high_t + P_low_t + P_close_t)/3`; `V_t` is volume on day t. The printed expression is unusual; do not silently replace it with a conventional VWAP formula. |
| `f_c_ma` | `P_close_t / MA20(P_close)` | Relative close vs. 20-day simple moving average. |
| `f_roc` | `P_close_t / P_close_(t-12) - 1` | 12-period rate of change. |
| `f_mom_k` | `P_close_t / P_close_(t-k) - 1`, `k in {5,10}` | Momentum at k = 5 or 10. |
| `f_p_acc` | `r_t - r_(t-1)` | Price acceleration; `r_t = P_close_t / P_close_(t-1) - 1`. |
| `f_ma_diff` | `(MA5(P_close_t) - MA20(P_close_t)) / MA20(P_close_t)` | Normalized moving-average difference. |
| `f_rsi` | `100 - 100/(1 + RS_(t,21))` | `RS_(t,k)` is ratio of k-day average gains to average losses; paper uses k = 21 here. |
| `f_rsi_rel` | `f_rsi/50 - 1` | RSI centered around 50. |
| `f_macd` | `(MACD_t - EMA9(MACD_t)) / P_close_t` | `MACD_t = EMA12(P_close_t) - EMA26(P_close_t)`; `EMA_k(.)` is k-day exponential moving average. |
| `f_v_rat` | `V_t / MA30(V_t)` | Volume relative to 30-day average. |
| `f_v_pct` | `V_t / V_(t-1) - 1` | Volume percentage change. |
| `f_v_mom` | `V_t / V_(t-5) - 1` | Five-day volume momentum. |
| `f_pv_corr` | `Corr10(r_t, f_v_pct)` | `Corr_k(X,Y)` is k-day rolling Pearson correlation. |
| `f_volat` | `sigma5(P_close_t) / MA5(P_close_t)` | Relative short-term volatility; `sigma_k(.)` is k-day standard deviation. |
| `f_bb_w` | `4 * sigma20(P_close_t) / MA20(P_close_t)` | Bollinger-band width. |
| `f_bb_p` | `(P_close_t - BB_low) / (4 * sigma20(P_close_t))` | Relative close position within the lower-to-upper Bollinger span; `BB_low = MA20(P_close_t) - 2*sigma20(P_close_t)`. |
| `f_atr` | `MA14(TR_t)` | Average true range; `TR_t = max(P_high_t-P_low_t, abs(P_high_t-P_close_(t-1)), abs(P_low_t-P_close_(t-1)))`. |
| `f_hl_rat` | `P_high_t / P_low_t` | High-low price ratio. |
| `f_volat_rat` | `sigma5(P_close_t) / sigma20(P_close_t)` | Short- to medium-term volatility ratio. |
| `f_cvh` | `(P_close_t - P_low_t) / (P_high_t - P_low_t)` | Close position in daily range. |
| `f_srs` | `f_mom_5 - MA10(f_mom_5)` | Called “Sector Relative Strength” in the paper. |

**Important source-faithfulness note:** the table above reformats the typeset Table 5. Keep the formulas as printed, even where naming or formula conventions look unusual; record any alternative interpretation separately rather than changing the source transcription.

To ensure statistical robustness and filter out market microstructure noise, we logically exclude highly inactive samples by requiring a minimum absolute volume change of 0.3%. Given the high idiosyncratic noise in daily closing prices, we employ a multi-day moving average ̄ Pi,t as the benchmark price (representing the “fair value” of the recent window) to capture genuine structural trends. The dynamic threshold Ti,t scales with recent volume-volatility: Ti,t = Tbase ⋅ (1 + k ⋅ tanh(| |voli,t | |)), (28) where Tbase is the baseline threshold (set to 0.0003 to rigorously filter out marginal, statistically insignificant price fluctuations), k is an adjustment factor (empirically set to 3 to dynamically suppress high-variance noise during turbulent periods), and voli,t is the volume change percentage.

The future return Ri,t+1 relative to this average price is calculated as: Ri,t+1 = Pi,t+1 − ̄ Pi,t ̄ Pi,t , (29) To prevent the model from overfitting to marginal upward fluctuations that barely cross the threshold, we introduce a stochastic label smoothing mechanism. The final label yi,t is defined as: yi,t = { 1, if Ri,t+1 > Ti,t and survives a 10% stochastic dropout, 0, otherwise. (30) The 10% dropout rate is deliberately chosen to align with standard label smoothing heuristics (0.1) widely adopted in deep learning.

In low signal-to-noise financial datasets, this specific rate injects optimal stochastic regularization–breaking the deterministic memorization of borderline noise without destroying the underlying distribution of robust momentum signals.

#### 6.1.2. Baselines

To thoroughly evaluate the performance of our proposed ST-GRAD model, our baseline selection follows a three-tier progressive strategy designed to benchmark the model against different levels of complexity: (1) traditional and sequence-based models to establish a performance floor; (2) standard graph-based models to evaluate spatial modeling; (3) the absolute latest spatio-temporal SOTAs to represent the current research frontier. The detailed parameters and configurations of these baseline models are provided in Table 6.

- RF (Ballings et al., 2015; Breiman, 2001): A popular ensemble learning method that constructs a large number of decision trees and outputs the class determined by the majority vote of the individual trees.

- SVM (Tay & Cao, 2001): A powerful classification algorithm that finds the optimal hyperplane to separate data points from different classes.

- ANN (Bishop, 1995): Typically refers to shallow feedforward neural networks with one or two hidden layers, also known as Multilayer Perceptrons (MLP).

- DNN (LeCun et al., 2015): An extension of the ANN, characterized by having multiple hidden layers (usually more than two). DNNs are capable of learning more complex, hierarchical abstract features from data due to their deeper structures.

- LSTM (Graves et al., 2013; Hochreiter & Schmidhuber, 1997): A type of recurrent neural network (RNN) that uses a gated mechanism to effectively learn long-term dependencies and mitigate the gradient vanishing problem common in simple RNNs. We evaluated both 1- layer and 2-layer LSTM architectures.

- BiLSTM (Schuster & Paliwal, 1997): An extension of LSTM that processes sequences in both forward and backward directions, enabling the model to capture information from both the past and the future within the sequence window. We tested both single-layer and double-layer versions.

- CNN (Kim, 2014): Although traditionally used in image processing, 1D CNNs can also effectively extract local patterns and features from time series data by applying convolution filters along the time axis.

- CNN-LSTM (Zhao et al., 2024): A hybrid architecture that first employs CNN to automatically extract high-level, local features from the

input financial time series, and then feeds these feature sequences into an LSTM network to capture their temporal dependencies.

- GCN (Kipf, 2016): A basic graph neural network model that learns node representations by aggregating information from neighboring nodes. It is effective in capturing the static spatial relationships between assets.

- AASTHGCN (Su et al., 2024): An advanced, attention-based spatiotemporal graph convolutional network model. It serves as a sophisticated baseline for current spatio-temporal prediction tasks, designed to capture dynamic spatio-temporal dependencies among assets.

- COGRASP (Li et al., 2025): A state-of-the-art model published in 2025 that utilizes a co-occurrence graph-based approach for stock price forecasting. It serves as our most recent baseline, representing the latest advancements in dynamic and multi-relational graph modeling.

### Table 6. Baseline-model hyperparameter settings

| Method | Parameters | Setting |
|---|---|---|
| RF | Maximum depth | 10 |
| RF | Number of decision trees | 100 |
| SVM | Regularization factor | 1 |
| SVM | Kernel | Linear |
| ANN/DNN | Input length; batch size; max epochs; learning rate; optimizer; loss | 720; 32; 100; 0.01; Adam; binary cross-entropy |
| LSTM | Input shape; batch size; max epochs; learning rate; optimizer; loss | (30, 24); 32; 50; 0.001; Adam; binary cross-entropy |
| BiLSTM | Input shape; batch size; max epochs; learning rate; optimizer; loss | (30, 24); 32; 50; 0.001; Adam; binary cross-entropy |
| CNN | Input length; batch size; max epochs; learning rate; optimizer; loss | (30, 24, 1); 32; 100; 0.01; SGD + momentum; binary cross-entropy |
| CNN-LSTM | Input shape; batch size; max epochs; learning rate; optimizer; loss | (30, 24); 32; 50; 0.001; Adam; BCELoss |
| GCN | Input shape; batch size; max epochs; learning rate; optimizer; loss | (50, 24); 32; 50; 0.001; Adam; BCELoss |
| AASTHGCN | Input shape; batch size; max epochs; learning rate; optimizer; loss | (50, 30, 24); 32; 50; 0.001; Adam; BCELoss |
| COGRASP | Input shape; batch size; max epochs; learning rate; optimizer; loss | (50, 30, 24); 32; 50; 0.001; Adam; BCEWithLogitsLoss |

#### 6.1.3. Evaluation metrics

To evaluate the performance of the model comprehensively, we select five core metrics commonly used in binary classification tasks (Su et al., 2024). These metrics are Accuracy (Acc), Precision (Pre), Recall (Rec), F1-score (F1), and Area Under the ROC Curve (AUC) (Fawcett, 2006). Their mathematical definitions are as follows: Acc = T P + T N T P + T N + FP + FN , (31) Pre = T P T P + FP , (32) Rec = T P T P + FN , (33) F1 = 2 ⋅ Pre ⋅ Rec Pre + Rec .

(34) Here, T P, T N, FP , and FN represent True Positives, True Negatives, False Positives, and False Negatives, respectively. AUC measures the area under the ROC curve, providing a comprehensive measure of the model’s classification performance across all possible thresholds. A higher AUC value (closer to 1) indicates a better discriminatory ability of the model.

#### 6.1.4. Parameter settings

To ensure reproducibility, all experiments were conducted under consistent software and hardware environments. We used the AdamW optimizer with a learning rate of 1e − 3, a batch size of 32, and trained for 50 epochs. For stability, the reported results are the average of 5 runs. The model weights achieving the best F1 score on the validation set during each run were saved and used for the final evaluation. The experimental environment and key parameter settings for the ST-GRAD model are summarized in Table 7.

### 6.2. RQ1: prior graph structure validity analysis

To validate the effectiveness of our proposed graph construction method based on deep feature clustering, we conducted comparative experiments. In these experiments, we fixed the ST-GRAD model structure and only replaced the input prior adjacency matrix Anorm. The three graph construction methods compared were as follows:

- ST-GRAD (Proposed Method,labeled as RIE): This method utilizes the CNN-LSTM representation combined with behavioral trend clustering (K-Means), exactly as detailed in Section 2.3.

- ST-GRAD Industry(labeled as Prior Knowledge): This method adopts an adjacency matrix based on predefined industry sectors.

- ST-GRAD Correlation(labeled as Return Rate): This method uses the Pearson correlation coefficient matrix based on historical returns. The experimental results, shown in Fig. 4, clearly indicate that our proposed method based on behavioral trend clustering achieves the highest F1 score, significantly outperforming the other two methods.

All other evaluation metrics also favor our method, confirming the superiority of our approach. This result strongly validates that our method constructs a more effective graph structure. This superiority fundamentally stems from the fact that while static industry classifications offer useful prior knowledge, they fail to capture dynamic concept sectors formed by companies with similar business models or market focuses, which frequently transcend traditional industry boundaries.

Furthermore, while simple yield correlation mainly measures linear relationships, our method explicitly clusters non-linear deep features alongside macroscopic trend indicators. This enables the model to uncover deeper, more intrinsic dynamic behavioral similarities between stocks. As a result, our data-driven be-

### Table 7. Experimental environment and ST-GRAD settings

| Category | Name | Configuration / value |
|---|---|---|
| Hardware/software | GPU | NVIDIA GeForce RTX 4060 Laptop |
| Hardware/software | GPU memory | 8 GB |
| Hardware/software | Framework | PyTorch 2.4.1, CUDA 12.1 |
| Model architecture | Number of ST-Block layers (`L`) | 2 |
| Model architecture | Hidden channels (`D`) | 32 |
| Model architecture | Node-embedding dimension (`d`) | 10 |
| Model architecture | Diffusion steps (`K_diff`) | 2 |
| Model architecture | Temporal-attention heads | 4 |
| Model architecture | Parameter count | 216,429 |
| Training | Optimizer | AdamW |
| Training | Initial learning rate | 0.001 |
| Training | Weight decay | 0.01 |
| Training | Batch size | 32 |
| Training | Epochs | 50 |
| Training | Number of runs | 5 |
| Training | Training time per run | 51.64 s |

Fig. 4. Adjacency matrix comparison of experimental results. havioral graph provides richer and more accurate spatial relationship information compared to conventional topological priors. To intuitively demonstrate why this behavioral trend clustering graph (Anorm) outperforms static industry classifications, Fig. 5 visualizes the normalized pricing trajectories of a representative subset of stocks assigned to the same cluster.

As illustrated, although these assets belong to entirely different traditional industry sectors (e.g., Energy, Healthcare, and Technology), their historical price dynamics, overall slopes, and volatility patterns are highly synchronized. By linking these structurally coherent assets via the trend clustering prior, the BDGC module can successfully capture and propagate cross-sector momentum signals that rigid industry adjacency matrices inherently ignore, ultimately explaining the superior prediction accuracy previously demonstrated. In conclusion, we affirmatively answer RQ1: Our proposed graph construction method supplies the model with a more effective and informative structural prior, directly leading to significant improvements in final forecasting performance.

### 6.3. RQ2: comprehensive performance evaluation of the model

To address RQ2 – whether our proposed ST-GRAD model offers significant advantages in overall performance compared to various baseline models – we conducted a comprehensive performance comparison and statistical significance test. As shown in Table 8, our proposed ST-GRAD model outperforms all baseline models across four key metrics: Accuracy (Acc: 0.8313), Recall (Rec: 0.8799), F1-score (F1: 0.8215), and AUC (AUC: 0.9046), strongly validating its superior overall performance. The model’s robust classification ability is further demonstrated by the ROC curve for the best model (with an AUC of 0.9053) and its confusion matrix across 5 runs, as shown in

Fig. 6. The confusion matrix also confirms its effectiveness, particularly in accurately identifying positive cases (0.889) while maintaining a high true negative rate (0.785). From the overall results in Table 8, our proposed ST-GRAD model outperforms all baseline models across the four key metrics: Acc, Rec, F1, and AUC, clearly demonstrating its comprehensive performance advantage.

(1) Comparison with Traditional Machine Learning Models: ST- GRAD shows a significant improvement over traditional machine learning models. Specifically, its F1 score is 4.6% higher than that of RF (F1: 0.7852) and 26.6% higher than SVM (F1: 0.6491). This highlights the limitations of traditional models, which struggle to capture the dynamics of time series and the complex, structured relationships between financial assets.

These models are generally not designed to handle the temporal and spatial complexities inherent in financial forecasting tasks. (2) Comparison with Sequential Deep Learning Models: ST-GRAD also outperforms time series-specific models such as LSTM, BiLSTM, and CNN. Among these, CNN-LSTM achieves the highest F1 score (F1: 0.8105), but ST-GRAD achieves an F1 score that is 1.4% higher.

Notably, while CNN-LSTM has a slight edge in Precision (Pre: 0.8019 vs

Fig. 5. Visualization of behavioral trend clustering for the initial 100 time steps. The plotted assets, despite originating from distinct traditional industry sectors, are assigned to the same behavioral cluster due to their highly synchronized macroscopic price trends and volatility profiles.

Fig. 6. ROC and confusion matrix in ST-GRAD.

### Table 8. Performance comparison on stock trend prediction

| Model | ACC | PRE | REC | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| RF | 0.8138 ± 0.0299 | 0.7814 ± 0.0468 | 0.7931 ± 0.0631 | 0.7852 ± 0.0389 | 0.8847 ± 0.0265 |
| SVM | 0.7080 ± 0.0000 | 0.6883 ± 0.0000 | 0.6201 ± 0.0000 | 0.6491 ± 0.0000 | 0.7780 ± 0.0000 |
| ANN | 0.7926 ± 0.0000 | 0.7639 ± 0.0515 | 0.7687 ± 0.0784 | 0.7636 ± 0.0505 | 0.8715 ± 0.0318 |
| DNN | 0.7949 ± 0.0365 | 0.7591 ± 0.0515 | 0.7868 ± 0.0719 | 0.7703 ± 0.0462 | 0.8741 ± 0.0316 |
| CNN | 0.7059 ± 0.0845 | 0.6338 ± 0.1095 | 0.8202 ± 0.2223 | 0.6921 ± 0.1528 | 0.8234 ± 0.0657 |
| 1-layer LSTM | 0.7995 ± 0.0501 | 0.7461 ± 0.1163 | 0.7962 ± 0.1324 | 0.7685 ± 0.1184 | 0.8763 ± 0.0415 |
| 2-layer LSTM | 0.7869 ± 0.0540 | 0.7373 ± 0.1107 | 0.7812 ± 0.1390 | 0.7542 ± 0.1190 | 0.8632 ± 0.0486 |
| 1-layer BiLSTM | 0.8019 ± 0.0500 | 0.7570 ± 0.1185 | 0.7843 ± 0.1314 | 0.7682 ± 0.1179 | 0.8780 ± 0.0436 |
| 2-layer BiLSTM | 0.7904 ± 0.0525 | 0.7374 ± 0.1191 | 0.7878 ± 0.1361 | 0.7589 ± 0.1185 | 0.8686 ± 0.0392 |
| GCN | 0.6029 ± 0.0004 | 0.6008 ± 0.0045 | 0.2989 ± 0.0121 | 0.3990 ± 0.0101 | 0.6303 ± 0.0011 |
| CNN-LSTM (2024) | 0.8309 ± 0.0027 | 0.8019 ± 0.0031 | 0.8194 ± 0.1147 | 0.8105 ± 0.0045 | 0.9040 ± 0.0012 |
| AASTHGCN (2024) | 0.8277 ± 0.0031 | 0.7852 ± 0.0100 | 0.8115 ± 0.0136 | 0.7964 ± 0.0036 | 0.9017 ± 0.0031 |
| COGRASP (2025) | 0.7859 ± 0.0013 | 0.7468 ± 0.0083 | 0.7794 ± 0.0151 | 0.7626 ± 0.0029 | 0.8619 ± 0.0007 |
| **ST-GRAD (ours)** | **0.8313 ± 0.0018** | 0.7705 ± 0.0063 | **0.8799 ± 0.0078** | **0.8215 ± 0.0009** | **0.9046 ± 0.0003** |

The paper marks the ST-GRAD F1 result with `***`, described as extreme statistical significance (`p < 0.001`) via independent two-sample t-tests against top-performing GNN baselines; bold values in the source indicate best performance.

0.7705), its Recall (Rec: 0.8194) is much lower than ST-GRAD’s (Rec: 0.8799). In financial forecasting, a higher Rec is often more important than a higher Pre because it reflects the model’s ability to capture genuine upward structural trends, which are typically rarer and harder to predict. The advantage of ST-GRAD in Rec, along with its superior F1, demonstrates its ability to effectively identify positive momentum signals, a crucial feature for financial forecasting.

(3) Comparison with Graph Neural Network Models: ST-GRAD maintains a clear advantage when evaluated against other graph-based architectures. The classical GCN model achieves an F1 score of only 0.3990, highlighting that simple static graph structures are insufficient for capturing the complex spatio-temporal dynamics of financial data. Furthermore, ST-GRAD outperforms recent advanced spatio-temporal baselines, including the 2024 SOTA AASTHGCN (F1 ∶ 0.7964) and the 2025 SOTA COGRASP (F1 ∶ 0.7626).

This performance gap primarily results from two architectural differences. First, unlike models such as AASTHGCN, the proposed MVCA mechanism explicitly captures the intricate interactions across both feature-spatio and feature-temporal dimensions. Second, recent models like COGRASP depend heavily on purely data-driven adaptive graphs.

These dynamic structures are highly vulnerable to overfitting when exposed to noisy financial fluctuations. To address this limitation, the BDGC module in ST-GRAD explicitly combines the stability of static behavior priors with the flexibility of adaptive graphs. Ultimately, this hybrid spatial propagation and fine-grained feature decomposition allow ST-GRAD to extract genuine structural momentum from volatile market data more effectively.

To rigorously confirm that the observed performance improvements are statistically significant and not artifacts of random initialization, we performed independent two-sample t-tests across the 5 runs. As highlighted by the ∗∗∗ notation in Table 8, ST-GRAD achieves extreme significance (p < 0.001) in the core F1 metric when compared to leading GNN architectures, including the top spatio-temporal model, AASTHGCN (pvalue = 0.0004, 95% CI: [0.0116, 0.0272]) and the most recent 2025 SOTA COGRASP (p-value < 0.0001, 95% CI: [0.0554, 0.0625]). Furthermore, even against the most competitive sequence-based model, CNN-LSTM, ST-GRAD maintains a statistically significant and stable lead (p-value = 0.0106, 95% CI: [0.0058, 0.0163]).

These results, combined with the statistically significant lead in Recall (p-value < 0.0001 against all three models), formally confirm that the ST-GRAD model captures genuine structural momentum rather than overfitting to specific data splits or benign market conditions. In summary, we affirmatively answer RQ2: The ST-GRAD model not only surpasses all baselines in the average values of multiple key metrics, but it also achieves a statistically significant performance advantage over the strongest baseline models in both F1 and Recall. These two metrics are especially critical in imbalanced financial prediction tasks.

This demonstrates the model’s superior ability to effectively identify directional market movements and achieve optimal comprehensive performance, marking a significant advancement in the field.

### 6.4. RQ3: analysis of the effectiveness and synergy of key components

To answer RQ3, we conducted a systematic ablation experiment to assess the individual contributions of the model’s key components and to verify their synergistic relationship. We evaluated different variants by removing or replacing specific modules as follows:

- ST-only (Single View): This variant removes the feature-spatio (DS) and feature-temporal (DT) views, keeping only the standard spatiotemporal view. It validates the necessity of our multi-view orthogonal design compared to traditional single-view modeling.

- wo-Fusion (Mean Fusion): This variant replaces the adaptive view fusion weights (αST , αDS , αDT ) with equal fixed weights, testing the contribution of the dynamic fusion mechanism.

- wo-TA (Without Temporal Attention): This variant removes the temporal attention module from the STBlock, verifying the necessity of applying attention weighting to the time series itself.

- wo-ud_conv (Without U-D Conv): This variant replaces the U-D Conv used to generate attention views with a standard 1 × 1 convolution, aiming to assess the role of the U-D Conv in extracting statistical features.

- wo-st (Without Spatio-Temporal View): This variant removes the ST Cross-Attention View, testing the importance of the ST view, which focuses on individual stock temporal patterns.

- wo-ds (Without Feature-Spatio View): This variant removes the DS Cross-Attention View, testing the contribution of the DS view, which captures the relationship between features and assets.

- wo-dt (Without Feature-Temporal View): This variant removes the DT Cross-Attention View, designed to assess the contribution of the DT view, which focuses on market consensus.

- wo-DGCN (GCN instead of Adaptive GCN): This variant replaces the BDGC mechanism with a simplified single-step adaptive graph convolution, to verify whether BDGC offers an advantage over a basic aggregation method. Table 9 demonstrates that the complete ST-GRAD model achieves the highest performance across the core metrics, securing an F1 score of 0.8215 ± 0.0009 and ranking first in both Accuracy and AUC. Every ablated variant exhibits a clear performance decline. Notably, the removal of the temporal attention module (wo-TA) results in the most substantial F1 drop, confirming the necessity of capturing temporal dependencies prior to the multi-view analysis. We further analyze the parameter scale of each variant to confirm that performance improvements originate from the proposed structural designs rather than simply increased architectural capacity. Regarding the multi-view framework, removing individual views (such as wo-st or wo-dt) reduces the overall parameter count by only 324 parameters. However, this marginal capacity change still leads to a statistically significant decrease in the F1. Furthermore, the ST-only variant degrades the model to a traditional single-view architecture by completely removing both the DS and DT views. Its resulting performance degradation validates that the gains are driven by the structural decoupling of market dynamics into orthogonal perspectives. To evaluate the synergistic relationship among these views, the wo-Fusion variant replaces adaptive weighting with simple average pooling. The subsequent performance drop validates the effectiveness of the dynamic fusion mechanism in capturing view synergy. Additionally, the wo-DGCN variant maintains a parameter scale (212,333) highly comparable to the full model but shows clear performance degradation. This confirms that the improvements are attributed to the specific design of combining static clustering priors with dynamic bidirectional diffusion, rather than relying on a generic graph propagation layer. In conclusion, the ablation study provides a comprehensive answer to RQ3. The individual contributions of each module are quantitatively verified, with the temporal attention and graph diffusion modules providing foundational temporal and spatial processing capabilities. Moreover, a strong synergistic relationship exists among the components. The multi-view cross-attention and adaptive fusion mechanisms do not merely add parameters; they work collaboratively to disentangle and integrate complex market signals, forming the core performance advantage of the ST-GRAD framework.

### 6.5. RQ4: sensitivity analysis of key hyperparameters of the model

To investigate the sensitivity of the ST-GRAD model to key hyperparameters, we focused on two parameters: the number of stacked ST- Blocks (L) and the number of diffusion steps (Kdiff ). A series of parameter comparison experiments were conducted, and the results are displayed in

Fig. 7.

### Table 9. Ablation study results

| Model | ACC | PRE | REC | F1 | AUC | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| **ST-GRAD (ours)** | **0.8313 ± 0.0018** | 0.7705 ± 0.0063 | 0.8799 ± 0.0078 | **0.8215 ± 0.0009** | **0.9046 ± 0.0003** | 216,429 |
| ST-only (single view) | 0.8277 ± 0.0031 | 0.7852 ± 0.0100 | 0.8115 ± 0.0136 | 0.8180 ± 0.0015 | 0.9014 ± 0.0018 | 213,093 |
| wo-Fusion | 0.8270 ± 0.0038 | 0.7623 ± 0.0107 | **0.8840 ± 0.0122** | 0.8185 ± 0.0019 | 0.9028 ± 0.0012 | 216,423 |
| wo-TA | 0.8186 ± 0.0045 | 0.7520 ± 0.0126 | 0.8796 ± 0.0157 | 0.8106 ± 0.0019 | 0.8985 ± 0.0018 | 207,853 |
| wo-ud_conv | 0.8287 ± 0.0027 | **0.8793 ± 0.0122** | 0.7671 ± 0.0096 | 0.8192 ± 0.0010 | 0.9030 ± 0.0012 | 212,883 |
| wo-st | 0.8279 ± 0.0019 | 0.7642 ± 0.0063 | 0.8825 ± 0.0106 | 0.8190 ± 0.0020 | 0.9034 ± 0.0010 | 216,105 |
| wo-ds | 0.8281 ± 0.0031 | 0.7661 ± 0.0107 | 0.8792 ± 0.0132 | 0.8186 ± 0.0012 | 0.9035 ± 0.0008 | 213,417 |
| wo-dt | 0.8285 ± 0.0015 | 0.7669 ± 0.0103 | 0.8790 ± 0.0138 | 0.8190 ± 0.0015 | 0.9027 ± 0.0009 | 216,105 |
| wo-DGCN | 0.8306 ± 0.0019 | 0.7706 ± 0.0072 | 0.8774 ± 0.0097 | 0.8205 ± 0.0009 | 0.9041 ± 0.0006 | 212,333 |

Fig. 7. Hyperparameter sensitivity analysis for number of layers and diffusion steps. Note: L denotes the number of ST-Blocks; Kdiff denotes the number of diffusion steps.

The model’s performance shows a clear sensitivity to both hyperparameters. The F1 Comparison chart (middle) highlights the optimal region: the model achieves the highest F1 (0.8215) when L = 2 and Kdiff = 2. 1. Impact of the Number of Layers (L).

Comparing configurations with Kdiff = 2 (i.e., L = 1, L = 2 and L = 3), we observe that increasing L from 1 to 2 results in a significant performance boost. However, as L increases further to 3, the F1 starts to decline. This suggests that an appropriate increase in model depth (L = 2) helps the model capture more abstract features.

However, excessively deep networks (L ≥ 3) become more prone to overfitting, especially when deal ing with high-noise financial data, leading to performance degradation. 2. Impact of the Number of Diffusion Steps (Kdiff ). For L = 2 (i.e.,Kdiff = 1, Kdiff = 2 and Kdiff = 3), the F1 score improves as Kdiff increases from 1 to 2.

However, when Kdiff reaches 3, the performance drops. This reflects the importance of the graph convolution’s “receptive field” size: Kdiff = 1 may not capture enough neighborhood information, while Kdiff = 3 may cause the “over smoothing” problem, where node features become overly homogeneous, weakening the model’s ability to distinguish between nodes. Additionally, the other subplots in Fig. 7 provide further insights into performance trade-offs.

The Model Ranking Across Metrics heatmap(left) shows that the optimal configuration (L = 2 and Kdiff = 2) provides the most balanced performance, ranking first across three of the five core metrics (Acc, Pre, and F1). On the other hand, the Pre vs. Rec Bubble Chart (right) highlights why some configurations fail to achieve optimal overall performance.

For example, the combination of (L = 2 and Kdiff = 3) yields the highest recall (Rec), but suffers from lower precision (Pre), resulting in a suboptimal F1 score. In summary, our answer to RQ4 is that the ST-GRAD model’s performance is highly sensitive to both the number of layers (L) and the number of diffusion steps (Kdiff ). Different combinations of these parameters significantly impact prediction results.

The experimental findings clearly indicate that the optimal configuration for balancing model complexity and prediction performance is L = 2 and Kdiff = 2.

### 6.6. RQ5: efficiency and complexity analysis

To address RQ5, we focus on analyzing the “efficiency innovation” of ST-GRAD. In financial forecasting, a model must not only be “accurate” but also “efficient” to meet the practical needs of real-time analytical pipelines. To comprehensively evaluate this, we compared the trade-off between predictive performance (F1) and multiple resource consumption metrics (Parameters, Training Time, Inference Latency, and Peak Memory) among ST-GRAD and three representative baselines: CNN- LSTM (an efficient non-graph model), COGRASP (a recent lightweight dynamic graph model), and AASTHGCN (a heavy state-of-the-art spatiotemporal model).

The quantitative experimental results, recorded under the same hardware environment and averaged per run, are summarized in Table 10. The results reveal that ST-GRAD effectively breaks the existing trade-off bottlenecks between computational overhead and predictive accuracy: (1) Overcoming Sequential Bottlenecks. While the sequence-based CNN-LSTM model is lightweight in terms of parameters and memory usage, its recurrent neural network architecture cannot be highly parallelized, leading to a severe sequential bottleneck (requiring ∼355s per training run).

ST-GRAD effectively eliminates this bottleneck, accelerating the training time by nearly 7x (reducing it to ∼51s) while achieving a superior F1 (0.8215 vs. 0.8105). (2) Avoiding Parameter Explosion. AASTHGCN achieves fast execution speeds due to its pure matrix-multiplication design.

However, this comes at the cost of massive parameter inflation (approximately 4.84 million parameters), which presents a significant scaling bottleneck. In contrast, ST-GRAD achieves superior predictive performance (F1: 0.8215 vs. 0.7964) with only 4.5% of AASTHGCN’s parameter capacity (216,429). This demonstrates that the MVCA mechanism provides a highly efficient structural alternative rather than relying on brute-force module stacking.

(3) Strategic Investment for Robustness. As a 2025 baseline, the purely data-driven COGRASP is ultra-lightweight and exhibits the fastest

### Table 10. Efficiency and performance trade-off (average per run)

| Model | F1 | Training time (s) | Inference latency (ms) | Peak memory (MB) | Session time (s) | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| CNN-LSTM (2024) | 0.8105 | 355.54 | 76.03 | 94.02 | 355.86 | 30,625 |
| COGRASP (2025) | 0.7626 | 23.55 | 5.95 | 1564.38 | 23.72 | 77,683 |
| AASTHGCN (2024) | 0.7964 | 32.60 | 10.63 | 718.03 | 33.03 | 4,837,441 |
| **ST-GRAD (ours)** | **0.8215** | 51.64 | 22.89 | 1916.71 | 51.84 | 216,429 |

### Table 11. Inter-view redundancy measurements (mean ± standard deviation over five runs)

| View pair | Cosine similarity (`S_cos`) | Mutual information (`I`) |
|---|---:|---:|
| ST vs. DS | 0.0985 ± 0.0567 | 0.0222 ± 0.0055 |
| ST vs. DT | 0.0703 ± 0.0826 | 0.0062 ± 0.0002 |
| DS vs. DT | 0.0278 ± 0.0290 | 0.0084 ± 0.0012 |

execution time. However, purely adaptive dynamic graphs are highly vulnerable to noise interference in volatile financial markets, resulting in a severe drop in accuracy (F1: 0.7626). To address this, ST-GRAD explicitly maintains three orthogonal views (MVCA) and computes bidirectional diffusion (BDGC) in parallel.

While this structural decoupling requires a moderately larger VRAM footprint (∼1.9GB, which is still highly accessible for modern GPUs) and an inference latency of 22.89 ms, this strategic investment in memory yields a crucial and substantial leap in prediction accuracy (F1: 0.8215). In quantitative forecasting, this tradeoff is highly favorable. Furthermore, as previously validated in Table 8, ST-GRAD exhibits exceptional stability, demonstrating the smallest variance in F1 across multiple runs.

In summary, our answer to RQ5 is affirmative. The design of ST-GRAD represents a profound “efficiency innovation” that successfully navigates the realistic trade-offs between parameter complexity, memory footprint, execution speed, and predictive accuracy.

### 6.7. Empirical validation of view orthogonality

This section evaluates the structural orthogonality described in Section 4.5 by analyzing the decorrelation of the feature representations learned by ΨS⇌T , ΨD⇌S , and ΨD⇌T . Specifically, we investigate how effectively the MVCA mechanism reduces inter-view information redundancy within the learned feature spaces. Table 11 reports the linear and non-linear dependencies between views, measured by Cosine Similarity (cos) and Estimated Mutual Information ().

To account for variance in random initializations, all metrics are averaged across five independent training runs. The quantitative results provide strong support for the orthogonality claim. The pair-wise cosine similarities are consistently near zero (ranging from 0.0278 to 0.0985), indicating that the feature spaces spanned by the three views are nearly mathematically orthogonal in the linear domain.

Furthermore, the estimated mutual information remains exceptionally low (ranging from 0.0062 to 0.0222), demonstrating that the views capture distinctly non-overlapping information components. To provide an intuitive qualitative perspective on this feature decorrelation, we employ the t-SNE algorithm to project the high-dimensional feature tensors into a 2D space. As shown in Fig. 8, the feature distributions of the ST, DS, and DT views occupy different topological regions with minimal spatial overlap.

This visual evidence perfectly corroborates the quantitative metrics, proving that the structural constraints of MVCA successfully prevent feature homogenization and enforce deep feature decorrelation. Furthermore, as depicted in the individual view visualizations (Fig. 8 a-c, colored by target labels), the samples within each decoupled topological space still exhibit distinct clustering patterns corresponding to their labels. This underscores that while the three views capture orthogonal information, they are simultaneously and actively extracting task-relevant, discriminative features without degrading into meaningless noise.

### 6.8. Out-of-sample robustness under market stress

To address the model’s robustness under extreme market conditions, we constructed a strictly isolated stress testing set directly from our original chronological validation set. To ensure absolute evaluation integrity and prevent validation leakage, we physically partitioned this original validation set (the latter 30% of the total dataset) into two strictly separated periods. The first 307 trading days were used as a clean validation set for model early stopping, which itself exhibited a challenging positive sample ratio of 43.65%.

Within the out-of-sample period, a specific 61-trading-day window, spanning November 2024 to January 2025, was exclusively reserved for blind stress testing. This specific window was chosen because it represents a profound macroeconomic stress event characterized by post-election uncertainties, extreme volatility, and aggressive sector rotations, driving the positive return ratio down to 46.52%. By optimizing and finalizing all models solely on the prior data and the clean validation set, we guarantee a completely blind out-ofsample evaluation on this stress window.

The performance comparison, averaged over 5 independent runs, is presented in Table 12. As shown in Table 12, the performance metrics during the No. 2024 to Jan 2025 stress window provide valuable insights into model robustness. This specific period was characterized by heightened volatility and structural sector rotations following macroeconomic policy shifts.

Consequently, the market exhibited strong directional momentum amidst the noise, which models with deep temporal modeling capabilities were able to partially capture. While baseline models such as COGRASP experienced slight performance variations (F1: 0.7586) as traditional spatial correlations shifted, ST-GRAD demonstrated notable stability. As a dynamic spatio-temporal graph model, ST-GRAD utilizes the MVCA mechanism to effectively decouple trend signals from volatile spatial noise.

Benefiting from this architecture and volatility-adaptive thresholds, ST-GRAD maintained a high F1 score of 0.8306 and a Recall of 0.8977, alongside a minimal standard deviation of 0.0005 across 5 runs. These results indicate that ST-GRAD does not overfit to static historical patterns, but rather robustly adapts to and extracts spatio-temporal features under shifting market conditions.

### 6.9. Interpretability and practical implications

Deep learning models in quantitative finance often face criticism for their “black box” nature. To address this, ST-GRAD offers an interpretable framework. By exposing its multi-view outputs, the model helps financial analysts bridge the gap between complex neural architectures and practical market analysis.

(1) Extracting Factor Insights via Multi-View Outputs. The orthogonal attention weights within the MVCA mechanism provide interpretable analytical signals. Visualizing these weights allows analysts to decode the structural drivers of market dynamics:

- Spatio-Temporal View (ΨS⇌T ): This view reveals the model’s focus on specific historical periods and peer assets. Analysts can use it to track volatility spillovers and sector lead-lag effects, aiding in risk attribution.

Fig. 8. t-SNE visualization of ST-GRAD view decoupling. In (a)-(c), colors correspond to the ground-truth labels, illustrating that each decoupled view retains distinct task-relevant discriminative patterns. In (d), colors distinguish the three spatio-temporal domains, visually confirming the minimal feature overlap and structural orthogonality.

### Table 12. Out-of-sample performance during the stated high-volatility stress period

The source caption reads “No. 2024 - Jan 2025”; the section text states November 2024 to January 2025. Results are mean ± standard deviation over five independent runs.

| Model | ACC | PRE | REC | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| COGRASP (2025) | 0.7732 ± 0.0046 | 0.7513 ± 0.0057 | 0.7665 ± 0.0220 | 0.7586 ± 0.0086 | 0.8539 ± 0.0035 |
| AASTHGCN (2024) | 0.8285 ± 0.0061 | **0.8193 ± 0.0131** | 0.8103 ± 0.0066 | 0.8147 ± 0.0050 | 0.8975 ± 0.0056 |
| CNN-LSTM (2024) | 0.8281 ± 0.0039 | 0.8138 ± 0.0043 | 0.8176 ± 0.0066 | 0.8157 ± 0.0045 | 0.9018 ± 0.0017 |
| **ST-GRAD (ours)** | **0.8297 ± 0.0019** | 0.7731 ± 0.0083 | **0.8977 ± 0.0125** | **0.8306 ± 0.0005** | **0.9044 ± 0.0014** |

- Feature-Spatio View (ΨD⇌S ): This view identifies which technical indicators drive predictions for specific assets. It enables analysts to monitor factor rotation, observing how market focus shifts between anomalies like momentum and volatility.

- Feature-Temporal View (ΨD⇌T ): This view captures market-wide consensus on specific features, such as a sudden sensitivity to volume spikes. It acts as a proxy for macro sentiment shifts, offering early warnings for systemic market shocks. (2) Uncovering Latent Structures for Risk Assessment. The adaptive adjacency matrix (Aadp) captures dynamic market topologies. Visualizing Aadp reveals data-driven asset correlations that often transcend traditional industry classifications. These latent structures assist analysts in identifying hidden concentration risks and optimizing asset diversification. While ST-GRAD demonstrates robust predictive capabilities, our current evaluation focuses strictly on signal generation rather than end-toend trading execution. Translating these predictive signals into actionable trading strategies would necessitate further cost-sensitive evaluations, including transaction fees, slippage, and strict trading protocols. Thus, we position ST-GRAD as a quantitative decision-support framework. It is designed to augment analysts in market trend analysis and risk assessment, maintaining a clear boundary between predictive modeling and application-level trading systems.

## 7. Conclusions

In this paper, we propose a novel, computationally efficient model named ST-GRAD to address the challenges of stock price trend prediction in financial markets. Firstly, we introduce the MVCA mechanism

to explicitly deconstruct and capture the entangled multi-dimensional dependencies in financial data across three orthogonal dimensions: ST (spatio-temporal), DS (feature-spatio), and DT (feature-temporal). Additionally, the BDGC module, which combines both a prior knowledge graph and a dynamically learned adaptive graph, allows the model to leverage stable industry associations while simultaneously capturing deeper hidden dependencies driven by market dynamics. Finally, by employing a hierarchical feature fusion strategy on the outputs of all ST- Blocks, ST-GRAD integrates multi-scale patterns, from local to global, enhancing its predictive power.

Extensive experiments on real stock datasets demonstrate that ST- GRAD significantly outperforms existing baseline models across multiple key evaluation metrics. The ablation study further confirms the effectiveness of each core component of the model. Notably, ST-GRAD achieves a state-of-the-art balance between model complexity and predictive performance, highlighting that the MVCA mechanism is both efficient and powerful in feature deconstruction.

When applied to real-world quantitative forecasting scenarios, ST- GRAD is able to capture complex factors driving stock price movements by integrating the asset’s own temporal evolution, cross-asset feature performance, and market-wide macro consensus. Additionally, by leveraging the model’s learned adaptive adjacency matrix and view fusion weights, ST-GRAD offers interpretable insights into dynamically changing sector rotations and shifting market focuses. This can serve as a robust decision-support foundation for market trend analysis and risk assessment.

However, while ST-GRAD excels in component efficiency, it faces certain limitations that outline clear directions for future research.First, regarding graph scalability in ultra-large markets, the O(N2) complexity associated with adaptive adjacency matrices poses a computational bottleneck when applied to thousands of stocks. Future research should explore techniques such as graph sampling and partitioning to address these hyper-graph scalability challenges while preserving the model’s architectural efficiency. Second, despite the partial transparency provided by the adaptive view fusion weights and dynamic adjacency matrices, the core neural architecture remains a “black-box.” Because high-risk financial decisions demand rigorous risk attribution, deeper interpretability is crucial.

Future work will focus on integrating posthoc interpretability methods (e.g., significance mapping or key subgraph extraction) and designing inherently interpretable components to more clearly reveal the economic logic driving predictions. Finally, beyond scalability and interpretability, the current study is constrained to signal generation. Although ST-GRAD serves as a robust quantitative decision-support framework, it lacks an end-to-end trading protocol.

To fully bridge the gap between predictive modeling and automated execution, subsequent research must evaluate ST-GRAD within a rigorous, cost-sensitive trading simulation that incorporates real-world friction, such as transaction fees, slippage, and portfolio optimization strategies. CRediT authorship contribution statement Yu Fu: Conceptualization, Methodology, Formal analysis, Investigation, Writing – original draft; Zhengming Huang: Methodology, Software, Validation, Data curation, Visualization, Writing – review & editing.; Bin Yu: Supervision, Project administration, Funding acquisition, Writing – review & editing. Data availability The data and source code that support the findings of this study are available in Anonymous GitHub at https://anonymous.4open.science/ r/ST-GRAD-Stock-Prediction-6E7D.

Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

Acknowledgement

Collate acknowledgements in a separate section at the end of the article before the references and do not, therefore, include them on the title page, as a footnote to the title or otherwise. List here those individuals who provided help during the research (e.g., providing language help, writing assistance or proof reading the article, etc.).

References

Ballings, M., Van den Poel, D., Hespeels, N., & Gryp, R. (2015). Evaluating multiple classifiers for stock price direction prediction.

Expert Systems with Applications, 42(20), 7046–7056. Bishop, C. M.

(1995). Neural networks for pattern recognition. Oxford University Press.

Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.

Journal of Econometrics, 31(3), 307–327. Box, G. E.

P., Jenkins, G. M., & Reinsel, G. C.

(1970). Time series analysis: Forecasting and control. Holden-Day.

Breiman, L. (2001). Random forests.

Machine Learning, 45(1), 5–32. Cheng, D., Yang, F., Xiang, S., & Liu, J. (2022).

Financial time series forecasting with multi-modality graph neural network. Pattern Recognition, 121, 108218. Cui, C., Wang, Z., Pu, Z., Liu, Y., Chen, K., Wang, Z., Wang, P., & Wu, C.

(2022). Temporal polynomial graph neural network for multivariate time series forecasting. In Advances in neural information processing systems (pp. 1331–1344).

(vol. 35). Engle, R. F.

(1982). Autoregressive conditional heteroscedasticity with estimates of the variance of united kingdom inflation. Econometrica: Journal of the Econometric Society, (pp. 987–1007).

(vol. 50). Fang, Z., Long, Q., Song, G., & Xie, K. (2021).

Spatial-temporal graph ODE networks for traffic flow forecasting. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 364–373). Fawcett, T.

(2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861–874.

Graves, A., Mohamed, A., & Hinton, G. (2013). Speech recognition with deep recurrent neural networks. 2013 IEEE International conference on acoustics, speech and signal processing, (pp. 6645–6649).

He, Z., Xu, L., Yu, J., & Wu, X. (2024). Dynamic multi-fusion spatio-temporal graph neural network for multivariate time series forecasting.

Expert Systems with Applications, 241, 122729. Hochreiter, S., & Schmidhuber, J. (1997).

Long short-term memory. Neural Computation, 9(8), 1735–1780. Ismail Fawaz, H., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D.

F., Weber, J., Webb, G. I., Idoumghar, L., Muller, P.-A., & Petitjean, F. (2020).

InceptionTime: Finding AlexNet for time series classification. Data Mining and Knowledge Discovery, 34(6), 1936–1962. Kim, Y.

(2014). Convolutional neural networks for sentence classification. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1746–1751).

Kipf, T. N. (2016).

Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907. LeCun, Y., Bengio, Y., & Hinton, G. (2015).

Deep learning. Nature, 521(7553), 436–444. Li, Z., Song, Z., Yuan, T., & Fu, X.

(2025). COGRASP: Co-occurrence graph based stock price forecasting. In Proceedings of the thirty-fourth international joint conference on artificial intelligence, IJCAI-25 (pp. 7527–7535).

International Joint Conferences on Artificial Intelligence Organization. https://doi.org/10.24963/ijcai.2025/837 Lu, X., Poon, J., & Khushi, M., et al. (2025). Leveraging biLSTM-GAT for enhanced stock market prediction: A dual-graph approach to portfolio optimization.

Applied Intelligence, 55(7), 601. Nelson, D. B.

(1991). Conditional heteroskedasticity in asset returns: A new approach. Econometrica: Journal of the Econometric Society, 55(7), 347–370.

Qian, H., Zhou, H., Zhao, Q., Chen, H., Yao, H., Wang, J., Liu, Z., Yu, F., Zhang, Z., & Zhou, J. (2024). MDGNN: Multi-relational dynamic graph neural network for comprehensive and dynamic stock investment prediction.

In Proceedings of the AAAI conference on artificial intelligence (pp. 14642–14650). (vol. 38). Ross, S.

A. (1976). The arbitrage pricing theory of capital asset pricing.

Journal of Economic Theory, 13(3), 341–360. Sanchez, E. H., Serrurier, M., & Ortner, M.

(2020). Learning disentangled representations via mutual information estimation. In European conference on computer vision (pp. 205–221).

Springer. Schuster, M., & Paliwal, K. K.

(1997). Bidirectional recurrent neural networks. IEEE Transactions on Signal Processing, 45(11), 2673–2681.

Sharpe, W. F. (1964).

Capital asset prices: A theory of market equilibrium under conditions of risk. Journal of Finance, 19(3), 425–442. Su, H., Wang, X., Qin, Y., & Chen, Q.

(2024). Attention based adaptive spatial–temporal hypergraph convolutional networks for stock price trend prediction. Expert Systems with Applications, 238, 121899.

Tay, F. E. H., & Cao, L.

(2001). Application of support vector machines in financial time series forecasting. Omega, 29(4), 309–317.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).

Attention is all you need. In Proceedings of the 31st international conference on neural information processing systems (p. 6000–6010). Red Hook, NY, USA: Curran Associates Inc.

Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2020a).

A comprehensive survey on graph neural networks. IEEE Transactions on Neural Networks and Learning Systems, 32(1), 4–24. Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C.

(2020b). Connecting the dots: Multivariate time series forecasting with graph neural networks. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 753–763).

Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for deep spatialtemporal graph modeling.

International Joint Conferences on Artificial Intelligence Organization. https://doi.org/10.24963/ijcai.2019/264 You, Z., Shi, Z., Bo, H., Cartlidge, J., Zhang, L., & Ge, Y. (2024). DGDNN: Decoupled graph diffusion neural network for stock movement prediction.

In Proceedings of the 16th international conference on agents and artificial intelligence (pp. 431–442). (vol. 2). Conference Proceedings. https://doi.org/10.5220/0012406400003636 Yu, B., Yin, H., & Zhu, Z.

(2017). Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. arXiv preprint arXiv:1709.04875. Zerveas, G., Jayaraman, S., Patel, D., Bhamidipaty, A., & Eickhoff, C.

(2021). A transformer-based framework for multivariate time series representation learning. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 2114–2124).

Zhao, Q., Hao, Y., & Li, X. (2024). Stock price prediction based on hybrid CNN-LSTM model.

Applied and Computational Engineering, 104, 110–116. Zheng, J., & Xie, L. (2025).

A dynamic stiefel graph neural network for efficient spatiotemporal time series forecasting. In J. Kwok (Ed.), Proceedings of the thirty-fourth international joint conference on artificial intelligence, IJCAI-25 (pp. 7155–7163).

International Joint Conferences on Artificial Intelligence Organization. https://doi.org/10. 24963/ijcai.2025/796
