# Explainable ST-GRAD: Paper-First Research and Implementation Plan

**Project objective:** Reproduce the ST-GRAD method described by Fu, Huang, and Yu (2026), then add and rigorously evaluate explainability methods that reveal how its multi-view attention and bidirectional graph diffusion contribute to individual stock-trend predictions.

**Primary source:** Fu et al., *ST-GRAD: Multi-view cross-attention and bidirectional diffusion for stock trend prediction*, *Expert Systems With Applications* 330 (2026), 133046. DOI: https://doi.org/10.1016/j.eswa.2026.133046.

**Implementation evidence:** the supplied anonymous archive `ST-GRAD-Stock-Prediction-6E7D.zip`, audited separately in `ST_GRAD_REPOSITORY_AUDIT.md`.

**Research stance:** paper-first. The paper defines the target method; the repository is evidence for resolving ambiguities and a compatibility baseline, not unquestionable ground truth. Where the paper is underspecified, record the decision, add tests, and report a sensitivity experiment if the decision could affect results. Never silently present an inferred detail as author-confirmed.

---

## 1. Research framing

This is not a project to maximize F1 by adding arbitrary modules. The central contribution is a trustworthy explanation layer for ST-GRAD that answers, for a particular stock and prediction date:

1. Which input indicators and historical observations most affected the output?
2. Which MVCA views (ST, DS, DT) mattered to this prediction, and how can that contribution be validated beyond reading their softmax fusion weights?
3. Which neighboring stocks and graph paths affected the prediction through the static prior and adaptive graph?
4. Are these explanations faithful to the model, stable under small input perturbations, and consistent with meaningful interventions?
5. When are the explanations unreliable, and how does explanation quality change across market regimes?

The paper claims that MVCA views capture complementary relationships and reports pairwise cosine similarity and estimated mutual information in Section 6.7 / Table 11. Those are representation redundancy measurements, **not sufficient explanations of individual predictions**. This project extends the paper by testing instance-level attribution and intervention-based explanations.

### Research questions

- **RQ1 — Reproduction:** Can the paper's data pipeline, graph, architecture, and evaluation protocol be implemented without the repository's identified deviations?
- **RQ2 — Input explanations:** Can feature-time-stock attributions identify influential inputs and pass perturbation-based faithfulness tests?
- **RQ3 — View explanations:** Do the three MVCA views make measurably different contributions to a prediction, beyond their learned fusion scalars?
- **RQ4 — Graph explanations:** Which stock nodes and graph edges materially affect predictions, and do explanations differ for static-prior versus adaptive propagation?
- **RQ5 — Reliability:** Are explanations stable, calibrated to model sensitivity, and robust across random seeds and the paper's clean/stress evaluation periods?

## 2. Scope and non-goals

### In scope

- A paper-aligned implementation of ST-GRAD and its original task: binary next-day stock-trend prediction.
- A separately preserved repository-compatibility mode for comparing against the released weights and code.
- Post-hoc explainability first, followed by optional explanation-aware training only if post-hoc methods establish a clear baseline.
- Reproducible data manifests, checkpoints, predictions, explanations, metrics, and configuration hashes.
- Evaluation on the paper's 50-stock universe and chronology; additional stocks or periods may be external-validity follow-ups, not replacements for the initial reproduction.

### Out of scope for the first milestone

- Claiming causal economic relationships from model attributions.
- Turning the system into a trading strategy or claiming profitability from classification metrics.
- Adding multiple architectural enhancements before the base model and explanations are tested.
- Using the stress window for checkpoint selection, threshold selection, hyperparameter tuning, or explanation-method selection.
- Treating attention maps or graph weights alone as proof of explanation faithfulness.

## 3. Evidence hierarchy and discrepancy policy

Use these labels in source comments, experiment logs, and documentation:

- **PAPER:** directly specified by the article's text, equation, table, or figure.
- **REPO:** directly observed in the supplied anonymous source archive.
- **INFERRED:** a necessary interpretation because the paper does not specify the detail.
- **DEVIATION:** intentionally differs from the paper for a controlled experiment.
- **UNRESOLVED:** insufficient evidence to choose confidently.

Keep `repo_compat` and `paper_aligned` configurations separate. Never load a repository checkpoint into a modified paper-aligned architecture and call the result a paper reproduction without verifying parameter names, shapes, and operator semantics.

The repository audit identified, at minimum, these issues to resolve in the paper-aligned path:

1. Target labels in `utils/data_processor.py` use the `Relative_Close` feature rather than an independently preserved raw closing-price series, and the volatility threshold includes an extra multiplier (`* 8`) not shown in the paper's formula.
2. Invalid/low-volume targets are initialized to zero rather than excluded using an explicit target mask.
3. MVCA signal branches use fixed all-ones `F.conv2d` filters rather than the learned transformations described by the equations.
4. A diffusion weight tensor exists but is not used in the forward pass and receives no gradient.
5. Hierarchical fusion/output behavior and normalization axes require equation/Algorithm 2 cross-checks.
6. The released training script evaluates on the same validation partition used for checkpoint selection; it does not physically isolate the paper's clean validation and stress windows.
7. The behavioral clustering encoder and K-Means process are absent; only precomputed cluster assignments are included.
8. The archive lacks the full baseline, ablation, and five-run experimental suite.

Each item must become a testable ticket. Do not change all items at once.

## 4. Paper-first baseline reproduction

### 4.1 Dataset

The paper reports Yahoo Finance OHLCV for 50 stocks, 10 per sector, across energy, healthcare, technology, basic materials, and industrials. Dates: 2020-08-19 to 2025-08-18, 1,255 observations per stock. Table 4 lists the ticker universe; use that exact order and save it in a manifest.

The repository contains the 50 per-ticker CSVs, `stock_order.npy`, `cluster_results.csv`, and a precomputed adjacency matrix. The provided cluster labels are useful for `repo_compat`, but they do not reproduce the stated CNN-LSTM + five statistics + Z-score + K-Means process from raw OHLCV.

Data requirements:

- Keep raw OHLCV immutable and versioned by hash.
- Validate dates, duplicate rows, missing bars, volume zeros, adjusted-vs-unadjusted price conventions, and cross-ticker calendar alignment.
- Record whether the Yahoo Finance files use adjusted prices; do not silently mix adjusted and unadjusted fields.
- Save the exact ordered ticker list and trading-date index with every processed dataset.
- Do not feed absolute raw prices into the model; retain close prices in a separate target-only sidecar for labels.
- All feature calculations at date `t` must use information available by `t` only.

### 4.2 Feature engineering

Implement the 24 features listed in paper Table 5, page 12, with individually unit-tested functions and explicit feature order. The paper's table includes relative close, open-close ratio, VWAP-like feature, close/MA, ROC, momentum, price acceleration, moving-average difference, RSI and relative RSI, MACD histogram, volume ratio/percentage change/momentum, price-volume correlation, volatility, Bollinger width/position, ATR, high-low ratio, close-in-range, and the remaining listed indicators. Verify the exact 24 names and equations against the rendered table and repository code before finalizing `feature_schema.json`.

Some Table 5 typography/equations are ambiguous, notably the VWAP expression. Preserve both the literal reading and conventional definition as named candidates if no stronger source resolves it. Do not silently replace the paper formula with a standard textbook formula. Log the choice and run a small sensitivity check if this feature materially changes values.

Every feature test should include hand-calculated small arrays, warm-up NaN behavior, division-by-zero behavior, and a causality test proving that modifying future rows cannot change past feature values.

### 4.3 Target labels and valid-target masks

Implement the paper's equations for the volatility-adaptive threshold and binary trend label using **raw close values in the target calculation only**. The article describes a one-day target relative to an average closing price over a historical window, plus a threshold adjusted by recent volume change and a minimum volume-change validity rule. Transcribe equations (28)–(30) from the paper into code comments and a unit test before training.

Do not copy the repository's extra `* 8` factor or its use of the first engineered feature unless a new primary-source check demonstrates that the printed paper is wrong. Preserve a Boolean `target_valid` mask for observations excluded by the paper's volume-change rule or unavailable values. Excluded targets must not silently become negative labels.

**Open interpretation to resolve:** the paper's notation for the historical average-price period is not fully explicit in the prose. Encode the chosen period in config, write a hand-computed fixture, and mark the period as INFERRED if it cannot be verified from the article or source.

### 4.4 Splits and leakage prevention

The paper reports 1,224 sequential samples after the 30-day warm-up, split into 856 training samples and 368 out-of-sample samples. It then describes the first 307 of the latter as clean validation and a 61-day stress window within the out-of-sample period. These partitions must be disjoint by target date.

- Training: first 856 chronological samples.
- Clean validation: 307 samples within the 368-sample out-of-sample region, per paper description.
- Stress test: 61 samples, held blind until all choices are frozen.
- Compute all normalization statistics on training samples only; freeze and apply them forward.
- Use the clean validation partition for checkpoint selection and validation-driven decisions only.
- Use the stress window once for final robustness evaluation. If its exact date boundaries cannot be reconciled with the trading calendar, record the date mapping and discrepancy; do not infer dates from month names alone.
- Sliding input windows may overlap in raw dates across adjacent splits, but each sample's target date must belong to exactly one split. Document this distinction explicitly.

### 4.5 Static behavioral prior graph

Paper method: build a CNN-LSTM representation per stock, concatenate five statistical trend indicators (mean, standard deviation/volatility, linear trend slope, period return, peak-to-peak amplitude), Z-score the combined vectors, run K-Means with `K=5`, connect stocks in the same cluster including self-loops, then symmetrically normalize the adjacency matrix.

Implementation requirements:

- Implement the encoder and five statistics as a standalone `build_prior_graph.py` pipeline.
- Record the date range used to build each stock profile. For a strict forecasting evaluation, graph features must not include future observations relative to the model's training/evaluation boundary. If the article's graph was built over the full period, reproduce that only in a clearly labeled paper-protocol compatibility experiment and also provide a leakage-safe training-period graph.
- Save encoder config/checkpoint, scaler, K-Means seed, labels, raw adjacency, normalized adjacency, ticker order, data hash, and graph-generation config.
- Keep the supplied cluster file as a separate `repo_compat` graph input.
- Test symmetry, self-loops, degree normalization, ticker-order alignment, and no cross-date leakage.

### 4.6 Model implementation

Canonical tensor convention for the paper-aligned code: input batches are `[B, F, T, N]`; latent state is `[B, D, T, N]`. Put all permutations in named helper functions and assert shapes at module boundaries.

Implement and test in this order:

1. Input normalization and 1x1 feature projection.
2. Per-stock temporal multi-head attention, four heads as reported, with residual and layer normalization.
3. MVCA ST view: pool max/mean/min over feature channels; map via learned up/down 1x1 convolutions; gate the projected input with sigmoid attention and tanh signal.
4. MVCA DS view: split time into non-overlapping blocks, pool max/mean/min within each block, produce feature-stock attention, and gate the correspondingly permuted tensor.
5. MVCA DT view: pool max/mean/min across stocks and gate the feature-time tensor.
6. Learnable three-scalar softmax fusion.
7. Adaptive adjacency from the two learnable node-embedding matrices, using the paper's specified nonlinearities/normalization.
8. Bidirectional graph diffusion using the normalized static prior and adaptive graph exactly as equations and Algorithm 1 specify, with every intended diffusion parameter participating in the forward pass.
9. ST-Block residual processing and LayerNorm with explicit normalization axes.
10. Hierarchical fusion across ST-Block outputs and output decoder, cross-checked against Algorithm 2 and Figure 1.

The paper's equation and pseudocode are authoritative targets, but if their tensor shapes or weight usage are inconsistent, do not invent a silent “correction.” Write a small tensor-shape derivation in `docs/ambiguities.md`, implement the minimal mathematically consistent interpretation, and include a unit test. If two interpretations are plausible and materially different, implement both as named variants and compare them under the same fixed protocol.

### 4.7 Training settings

Paper Table 7 reports PyTorch 2.4.1, CUDA 12.1, an NVIDIA GeForce RTX 4060 Laptop with 8 GB VRAM, hidden dimension `D=32`, node embedding dimension `d=10`, two ST-Blocks (`L=2`), two diffusion steps (`K_diff=2`), four temporal attention heads, AdamW, learning rate `1e-3`, weight decay `0.01`, batch size `32`, 50 epochs, and five runs. Treat these as reproduction targets, not proof that every other detail is specified.

Record seed, software versions, device, optimizer, learning-rate schedule, loss reduction, class weighting, checkpoint rule, and batch ordering. If a detail is absent from the paper, mark it INFERRED and do not present it as a published setting. Use five independent seeds and report each run as well as mean and standard deviation.

---

## 5. Explainability research design

### 5.1 Design principle

Start with **post-hoc explanations** on a frozen paper-aligned model. This isolates the question “can we explain this model?” from “does explanation-aware training change the model?” Only consider regularizing explanations after the post-hoc methods have a reproducible baseline.

For a prediction at target date `t` and stock `i`, define a case record containing:

- target ticker and target date;
- model logit, probability, predicted class, true label, and target-validity status;
- 30-day input date range;
- input feature schema and preprocessing version;
- per-feature/per-time attribution tensor;
- view-level intervention effects;
- graph node/edge intervention effects for the static and adaptive graph;
- model/checkpoint/config/data hashes;
- explanation method, baseline/reference, random seed, and runtime.

No explanation is allowed to use the target label, future OHLCV, stress outcomes, or post-target information as input to the explanation algorithm. The true label may be used after inference to group and evaluate explanations.

### 5.2 Explanation family A: feature and time attribution

**Primary method:** Integrated Gradients (IG) on the pre-sigmoid logit for each stock. Input shape `[F,T,N]`; explain stock `i`'s logit, attributing across the full multistock input unless a controlled target-stock-only variant is specified.

Implementation contract:

- Use a fixed, documented baseline. Start with zero in normalized feature space; compare against a training-set feature mean baseline. Do not choose baselines by looking at test outcomes.
- Calculate IG with a configurable number of integration steps and convergence delta. Test that attribution sums approximately to the logit difference between input and baseline.
- Aggregate signed attribution by feature, time step, and stock while preserving the full tensor for detailed inspection.
- Report positive and negative contributions separately; absolute magnitude alone hides direction.
- For target-stock-specific summaries, report both the target stock's input contributions and contributions from all other stocks.
- Provide feature-time heatmaps, top-feature rankings, and a temporal saliency curve. These are visualizations of the model's sensitivity, not causal claims.

**Validation tests:** deletion/insertion curves; compare confidence/logit change after masking top-attributed features/time points against random and bottom-attributed controls; IG completeness/convergence; model/label randomization sanity checks; explanation stability under small noise that does not flip the model output.

### 5.3 Explanation family B: MVCA view contribution

The learned fusion weights `softmax(alpha)` are global parameters and do not by themselves quantify per-example explanatory importance. Use them as descriptive metadata only.

Implement per-example **view interventions**:

1. Capture ST, DS, and DT view tensors before fusion.
2. For a selected sample/target stock, compute the original logit.
3. Replace one view at a time with a defined neutral reference (e.g., zeroed view tensor or batch/reference mean), keeping other views unchanged; measure the change in the target logit.
4. Also measure a normalized contribution by rescaling the selected view to a controlled level, with the intervention definition recorded.
5. Repeat with each view and combinations of views. Compare effects across stocks, dates, correctly/incorrectly classified cases, and clean/stress periods.
6. Evaluate whether intervention ranking predicts which view removal most degrades performance on a held-out evaluation subset.

Caveat: zeroing a latent view may create out-of-distribution representations. Therefore report at least one alternative intervention, such as replacing the view with its training-reference mean, and interpret results as model-dependence tests rather than causal explanations of markets.

**Hypothesis H2:** instance-level view intervention effects will vary across stocks and dates and will not be reliably represented by a single global softmax weight vector. Reject or qualify this claim if effects are near-constant or poorly reproducible.

### 5.4 Explanation family C: graph node and edge interventions

The model combines a static behavioral prior with an adaptive graph. Explanations should distinguish these channels rather than combine them into one opaque “neighbor importance” score.

Implement:

- **Node influence:** for target stock `i`, mask or replace the input/hidden representation of stock `j` and measure the change in `i`'s logit. Do not delete the target node.
- **Edge intervention:** set a selected adjacency entry to zero or to a documented reference weight, renormalize as appropriate, and measure the change in target logit. For symmetric static edges, intervene on `(i,j)` and `(j,i)` together unless a directional experiment is explicitly intended.
- **Graph-channel intervention:** remove the static-prior contribution, remove the adaptive-graph contribution, and remove each direction/propagation term separately where supported by the equations.
- **Neighborhood summaries:** rank influential neighboring stocks and visualize the top edges, distinguishing static-cluster membership from learned adaptive connectivity.
- **Controls:** random edges, low-weight edges, random nodes, degree-matched edges, and no-intervention baseline.

Report both the absolute logit change and the change in probability/class. Probability changes can saturate, so the pre-sigmoid logit is the primary quantitative outcome.

**Hypothesis H3:** targeted node/edge interventions will alter the target stock's logit more than degree-matched random interventions, on average, if the learned graph contains useful prediction-specific dependencies. This is an empirical hypothesis, not a guaranteed property.

### 5.5 Explanation family D: explanation reliability and uncertainty

Before adding a new uncertainty architecture, quantify existing explanation reliability:

- Seed stability: compare attribution ranks and top-k overlap across the five trained models.
- Perturbation stability: small feature perturbations with predictions unchanged; measure rank correlation and top-k overlap.
- Regime stability: compare clean-validation and stress-window explanation distributions without tuning on the stress window.
- Faithfulness: deletion/insertion curves and target-logit change relative to random controls.
- Sensitivity: compare explanation magnitude to finite-difference gradients for a small subset.
- Runtime and memory: report per-case explanation cost, including graph interventions.

Optional second-stage experiment: prediction uncertainty via deep ensembles using the five seed models. Compare predictive entropy/ensemble variance with error rate and explanation instability. Do not assert that uncertainty estimates are calibrated without evaluating calibration metrics.

### 5.6 Explanation interface / output format

Every explanation export should be machine-readable JSON/NPZ plus a human-readable report. Suggested schema:

```json
{
  "ticker": "EXAMPLE",
  "target_date": "YYYY-MM-DD",
  "prediction": {
    "logit": 0.0,
    "probability": 0.5,
    "class": 1
  },
  "input_window": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "model_hash": "...",
  "data_manifest_hash": "...",
  "explanations": {
    "feature_time_attribution": "path-or-array-key",
    "view_interventions": {},
    "graph_interventions": {},
    "stability": {}
  },
  "caveats": ["Model attribution is not a causal market explanation."]
}
```

The report should answer: what drove the model output, which intervention verified it, how stable was the explanation, and what is not known. Do not output unsupported natural-language narratives such as “stock A caused stock B to rise.” Use “the model relied on” or “the model output changed when…” instead.

---

## 6. Hypotheses and falsifiable predictions

| ID | Hypothesis | Primary measurement | Falsification / qualification condition |
|---|---|---|---|
| H1 | Paper-aligned corrections to labels and MVCA operators materially change at least some predictions relative to repository compatibility mode. | Per-date logits, prediction agreement, F1/ACC/AUC differences. | Differences are negligible across all metrics and predictions; document that outcome rather than claiming importance. |
| H2 | The three views have example-dependent contributions not captured by global fusion scalars alone. | Per-view intervention logit deltas; across-case variance; rank correlation with fusion weights. | Per-example effects are effectively constant or global weights already predict them well. |
| H3 | Learned graph neighborhoods contain prediction-relevant information beyond random/degree-matched graph edges. | Node/edge intervention deltas versus matched controls. | Targeted interventions do not exceed controls or results are unstable across seeds. |
| H4 | IG highlights inputs whose removal changes the target logit more than random or low-attribution removal. | Deletion/insertion AUC, logit change, completeness delta. | Attribution-guided removal is no better than controls, or results fail randomization sanity checks. |
| H5 | Explanations are more stable for high-confidence/correct predictions than for low-confidence/incorrect predictions. | Top-k overlap, Spearman rank correlation, attribution variance, ensemble variance. | No consistent association or effect reverses across seeds/regimes. |
| H6 | Graph/view explanations change across the paper's stress window in ways measurable without tuning on it. | Distribution shift in intervention effects and attribution profiles; compare to clean validation. | No detectable change; or effects are explained by target prevalence/input-scale differences alone. |

H1 is a reproduction diagnostic, not the principal explainability contribution. H2–H5 are the core research questions. H6 is a secondary robustness analysis and must remain blind until the methods are frozen.

## 7. Experimental matrix

### Experiment 0 — Source compatibility audit

- Run the released preprocessing and architecture with fixed seed/config.
- Regenerate both checkpoint metric reports, tensor shapes, parameter count, and hashes.
- Save raw predictions and labels.
- Purpose: prove that later changes are attributable to named differences.

### Experiment 1 — Paper-aligned reproduction

- Implement paper target formula, masks, train-only normalization, behavioral graph, learned MVCA signal transformations, active diffusion parameters, hierarchical fusion, and chronological protocol.
- Run five seeds using reported hyperparameters.
- Report per-seed Accuracy, Precision, Recall, F1, AUC; confusion matrices; class prevalence; valid-target counts; parameters; training/inference time; peak memory.
- Compare with paper Table 8 and other available results, clearly separating numeric closeness from protocol fidelity.

### Experiment 2 — Explanation method sanity checks

- Integrated Gradients completeness and convergence.
- Feature/time deletion and insertion against random and bottom-ranked controls.
- Model/label randomization checks.
- Perturbation stability on inputs where the prediction remains unchanged.
- Fail the release gate if explanations are attractive visually but fail basic faithfulness checks.

### Experiment 3 — MVCA view interventions

- Zero/reference-mean each of ST, DS, DT independently and in pairs.
- Compare per-example logit deltas, prediction flips, and performance changes on clean validation.
- Correlate effects with learned fusion weights, but do not equate correlation with explanation.
- Repeat across five checkpoints or a predeclared representative subset of seeds.

### Experiment 4 — Graph interventions

- Static-prior removal, adaptive-graph removal, and individual-channel removal.
- Top-k edge/node interventions versus random and degree-matched controls.
- Symmetric versus directed interventions only where mathematically valid.
- Report logit deltas and aggregate performance impact. Log intervention renormalization rules.

### Experiment 5 — Explanation stability

- Across five random seeds.
- Across small input perturbations.
- Across different IG baselines.
- Across correct vs incorrect and high- vs low-confidence cases.
- Report confidence intervals or bootstrap intervals for aggregate explanation metrics; avoid only qualitative examples.

### Experiment 6 — Blind stress evaluation

- Freeze model, baselines, thresholds, explanation methods, and analysis plan using train + clean validation only.
- Evaluate on the paper's isolated 61-day stress window once.
- Report predictive performance and explanation behavior, with explicit caveat that one historical stress period does not establish general stress robustness.

### Experiment 7 — Optional explanation-aware training (only after Experiments 2–6)

Candidate regularizers: attribution sparsity, perturbation consistency, or view-disentanglement regularization. Choose one at a time and specify the target property before running it. Compare against the identical paper-aligned baseline with equal seeds, compute budget, and evaluation protocol. Include the regularizer's cost and show that explanation metrics improve without merely suppressing useful signal or damaging predictive performance.

## 8. Metrics

### Predictive performance (paper-aligned)

- Accuracy, Precision, Recall, F1, ROC-AUC.
- Per-run values and mean ± standard deviation across five seeds.
- Positive-class prevalence and valid-label count per split.
- Stress-window metrics reported separately from clean validation.

### Input attribution

- IG completeness/convergence delta.
- Deletion/insertion curve AUC using consistent masking baselines.
- Top-k attribution overlap across seeds and perturbations.
- Spearman rank correlation between attribution maps.
- Randomization sanity-check outcomes.
- Runtime and memory per explained sample.

### View and graph explanation

- Absolute and signed target-logit change per intervention.
- Prediction flip rate and aggregate metric degradation after targeted intervention.
- Targeted-vs-random and targeted-vs-degree-matched effect difference.
- Across-seed rank stability and confidence intervals.

Do not combine these into one opaque “explainability score” unless a defensible aggregation is separately justified. Present predictive quality and explanation quality as distinct dimensions.

## 9. Implementation architecture

```text
st-grad-research/
├── configs/
│   ├── repo_compat.yaml
│   ├── paper_aligned.yaml
│   ├── explain_ig.yaml
│   ├── explain_views.yaml
│   └── explain_graph.yaml
├── data/
│   ├── raw/                 # immutable source data
│   ├── processed/           # generated, hash-indexed arrays
│   └── manifests/           # ticker order, date splits, feature schema
├── src/stgrad/
│   ├── data/
│   │   ├── download.py
│   │   ├── audit.py
│   │   ├── features.py
│   │   ├── labels.py
│   │   ├── windows.py
│   │   └── normalize.py
│   ├── graphs/
│   │   ├── behavioral_encoder.py
│   │   ├── clustering.py
│   │   ├── prior_graph.py
│   │   └── adaptive_graph.py
│   ├── models/
│   │   ├── temporal_attention.py
│   │   ├── mvca.py
│   │   ├── bdgc.py
│   │   ├── st_block.py
│   │   ├── st_grad.py
│   │   └── repo_compat.py
│   ├── explainability/
│   │   ├── integrated_gradients.py
│   │   ├── perturbation_metrics.py
│   │   ├── view_interventions.py
│   │   ├── graph_interventions.py
│   │   ├── stability.py
│   │   └── report.py
│   ├── training/
│   ├── evaluation/
│   └── visualization/
├── tests/
│   ├── test_features.py
│   ├── test_labels.py
│   ├── test_splits.py
│   ├── test_graphs.py
│   ├── test_shapes.py
│   ├── test_gradients.py
│   └── test_explanations.py
├── scripts/
│   ├── build_dataset.py
│   ├── build_graph.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── run_experiment_matrix.py
├── artifacts/{checkpoints,metrics,predictions,explanations,figures}/
└── docs/{ambiguities.md,deviations.md,experiment_log.md,model_card.md}
```

Use Python and PyTorch. Keep source compatibility code isolated. Avoid dependencies that are only used for plots in the training core. The explanation API must operate on a model in evaluation mode and accept a sample, target stock index, method config, and output directory.

## 10. Agent instructions: execution contract

The coding agent must follow the order below and must not jump to full training or architecture extensions before the corresponding gates pass.

### Phase A — Establish provenance and compatibility

1. Hash the uploaded ZIP and all raw data files; write `docs/source_manifest.md`.
2. Extract and inspect all source files; record runtime/dependency versions.
3. Reproduce the bundled preprocessing shapes, graph, model parameter count, checkpoint load, and checkpoint metrics.
4. Save repository predictions and outputs before any edits.
5. Create a discrepancy ledger mapping every paper equation/table requirement to source file/function, implementation status, and test.

**Gate A:** compatibility mode produces the recorded tensor shapes and checkpoint metrics within deterministic numerical tolerance. If it fails, fix the harness first; do not proceed by guessing.

### Phase B — Build the paper-aligned data pipeline

1. Create immutable raw-close sidecar aligned by date/ticker.
2. Implement the 24 features and `feature_schema.json` from Table 5; add hand-calculated tests.
3. Implement equations (28)–(30) and target validity mask. Check labels manually on synthetic examples.
4. Implement date windows and 856/307/61 split logic with exact target dates.
5. Fit normalization only on training samples; save means/scales and their axes.
6. Implement the behavioral graph encoder/statistics/clustering path; keep the supplied cluster file as a separate compatibility option.
7. Generate a data manifest with raw hashes, code version, date ranges, ticker order, feature order, split counts, label counts, graph provenance, and config hash.

**Gate B:** no feature or label depends on future data; invalid labels are masked from both loss and metrics; all split dates and shapes match the documented protocol; graph order matches tensor node order.

### Phase C — Implement paper-aligned operators

1. Implement each module independently with explicit shape annotations.
2. Add forward shape tests and backward gradient tests.
3. Compare every operator to the paper equation and Algorithm 1/2.
4. Ensure all intended trainable parameters participate in forward computation and receive gradients.
5. Resolve uncertain equations in `docs/ambiguities.md`; implement named alternatives when material.
6. Add deterministic small-tensor tests for view pooling, fusion, adjacency, diffusion, residuals, and output logits.

**Gate C:** shape, numerical, gradient, and parameter-usage tests pass. Do not use parameter-count equality as the only architecture correctness test.

### Phase D — Train and validate baseline

1. Train five seeds with paper-reported settings.
2. Select checkpoints on clean validation F1 only, per a documented rule.
3. Never use stress labels to tune any model, threshold, explanation baseline, or method.
4. Export raw logits/probabilities/predictions and metrics per sample and per seed.
5. Compare results to paper tables and explain differences using the discrepancy ledger.

**Gate D:** five run manifests and checkpoint hashes exist; aggregate metrics are reproducible; stress data remain unused.

### Phase E — Implement explanations

1. Implement IG on logits and completeness tests.
2. Implement deletion/insertion and randomization sanity checks.
3. Implement per-example ST/DS/DT interventions with zero and reference-mean replacements.
4. Implement static/adaptive graph channel, node, and edge interventions with matched random controls.
5. Add across-seed and perturbation stability metrics.
6. Export machine-readable arrays and JSON case reports.

**Gate E:** no method is accepted based only on heatmaps. At least one faithfulness test and one sanity check must pass; failure results must be recorded, not hidden.

### Phase F — Run analysis and write report

1. Run the predeclared experiment matrix.
2. Use paired comparisons across the same dates/seeds where possible; report effect sizes and intervals, not only p-values.
3. Open the stress set only after the analysis plan and settings are frozen.
4. Produce a report separating confirmed results, null results, failed hypotheses, and unresolved protocol deviations.
5. Only then propose explanation-aware training or additional mechanisms.

## 11. CLI contract

The agent should implement commands equivalent to:

```bash
python -m stgrad.cli audit-data --config configs/paper_aligned.yaml
python -m stgrad.cli build-data --config configs/paper_aligned.yaml
python -m stgrad.cli build-graph --config configs/paper_aligned.yaml
python -m stgrad.cli test-model --config configs/paper_aligned.yaml
python -m stgrad.cli train --config configs/paper_aligned.yaml --seed 43
python -m stgrad.cli train-multiseed --config configs/paper_aligned.yaml --seeds 43,101,202,303,404
python -m stgrad.cli evaluate --config configs/paper_aligned.yaml --split clean_val
python -m stgrad.cli explain --config configs/explain_ig.yaml --split clean_val --method integrated_gradients
python -m stgrad.cli explain --config configs/explain_views.yaml --split clean_val --method view_intervention
python -m stgrad.cli explain --config configs/explain_graph.yaml --split clean_val --method graph_intervention
python -m stgrad.cli evaluate --config configs/paper_aligned.yaml --split stress --frozen
```

Commands may be implemented differently, but must preserve separation of train/validation/stress, configuration provenance, and method outputs. The final stress command should require an explicit frozen-analysis flag and record the model/config hashes used.

## 12. Acceptance criteria

### Reproduction

- [ ] Repository compatibility metrics and artifacts are saved before changes.
- [ ] Paper-aligned labels use raw close series and equations (28)–(30), with synthetic unit tests.
- [ ] All 24 features match documented formulas and order; unresolved Table 5 ambiguity is logged.
- [ ] Training-only normalization and strict chronological target splits are verified by tests.
- [ ] Behavioral graph provenance is complete or explicitly marked as precomputed/unresolved.
- [ ] MVCA projections are learnable as specified; view shapes are explicit; no accidental broadcasting defines model semantics.
- [ ] All intended diffusion parameters are active and gradients are non-null.
- [ ] Five seeds, clean validation, and isolated stress evaluation are implemented.
- [ ] Main metrics, parameter count, runtime, memory, predictions, and run variance are exported.

### Explainability

- [ ] IG completeness/convergence is tested.
- [ ] Attribution deletion/insertion is compared to random and bottom-ranked controls.
- [ ] Model/label randomization sanity checks are run.
- [ ] View-level intervention effects are reported per example, not inferred solely from global fusion weights.
- [ ] Static and adaptive graph contributions are explained separately.
- [ ] Node/edge interventions are compared against random/degree-matched controls.
- [ ] Explanation stability across seeds and perturbations is quantified.
- [ ] Each case report contains target date, model/data hashes, method settings, results, and caveats.
- [ ] Stress-set explanations are not used to tune methods.
- [ ] Claims use model-dependence language, not causal market language.

## 13. Deliverables

1. `docs/source_manifest.md`: source archive and data hashes, provenance.
2. `docs/ambiguities.md`: paper equation/algorithm inconsistencies and chosen interpretations.
3. `docs/deviations.md`: every departure from the paper and why it was necessary.
4. `configs/paper_aligned.yaml` and `configs/repo_compat.yaml`.
5. Tested feature, label, split, graph, and model implementations.
6. Five paper-aligned checkpoints and per-seed result manifests.
7. Explanation library for feature/time, view, and graph interventions.
8. Faithfulness/stability evaluation tables and figures.
9. A final report with reproduction comparison, explanation hypotheses, null results, limitations, and next steps.

## 14. Risks and guardrails

- **Small dataset / narrow universe:** only 50 stocks and one historical period; explanation patterns may not generalize.
- **Label ambiguity:** label interpretation affects both model performance and attribution target; resolve before training.
- **Feature collinearity:** correlated indicators can split attribution across substitutes; explain groups and stability, not only single-feature ranks.
- **Latent interventions:** zeroing hidden tensors can create out-of-distribution states; compare multiple reference interventions and state the limitation.
- **Graph confounding:** edge importance is model-specific, not evidence of economic causality.
- **Attention-as-explanation fallacy:** attention scores are diagnostic signals, not standalone proof of why the prediction occurred.
- **Stress-set overuse:** repeated inspection can turn the stress set into a de facto validation set; freeze analysis in advance.
- **Paper/repository disagreement:** retain both paths and show exact differences; never quietly retrofit the released checkpoint to the paper-aligned model.

## 15. Primary source map

- **Paper pp. 3–4:** graph construction, problem formulation, input projection, temporal attention.
- **Paper pp. 5–8:** MVCA ST/DS/DT views, fusion, view orthogonality/complementarity.
- **Paper pp. 8–10:** BDGC, ST-Block and hierarchical integration.
- **Paper pp. 11–13:** data, Table 4 stock universe, Table 5 features, labels, splits, Table 7 hyperparameters.
- **Paper pp. 14–18:** main comparisons, ablations, sensitivity, resource use, view redundancy and stress-window results.
- **Repository audit:** `ST_GRAD_REPOSITORY_AUDIT.md` documents actual source files, supplied artifacts, checkpoint metrics, and deviations. Use it as implementation evidence, not a substitute for the paper.

---

**Final instruction to the coding agent:** prioritize correctness and auditability over speed. First reproduce the released code in compatibility mode, then implement the paper-aligned model, then build and validate explanations. Do not add new prediction mechanisms until the reproduction baseline and explanation faithfulness gates are complete. The intended research output is a defensible explainability study of ST-GRAD, including negative findings—not merely a dashboard of attention maps.
