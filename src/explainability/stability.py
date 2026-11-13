"""
Stability Evaluation for ST-GRAD Explanations (Phase 3 & Phase 6/7).
Measures:
1. Cross-Seed Stability across the 5 independently trained models (Seeds 42..46).
2. Input Perturbation Stability under small, label-preserving Gaussian noise.

Hardened against overstating stability:
- Enforces strict minimum valid trials (min_valid_trials=10). Zero or few valid trials fail the robustness check.
- Never defaults empty arrays to 1.0 (empty arrays yield 0.0 with status 'insufficient_valid_trials').
- Reports uncertainty: standard error and 95% confidence intervals.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from src.explainability.adapter import ModelTrackAdapter
from src.explainability.attribution import IntegratedGradients


def evaluate_seed_stability(
    models: List[Union[nn.Module, ModelTrackAdapter]],
    x: torch.Tensor,                                      # [1, F, T, N]
    A: torch.Tensor,                                      # [N, N]
    target_stock_idx: int,
    top_k: int = 5,
    ig_steps: int = 25
) -> Dict[str, Any]:
    """
    Measure explanation stability across independent random training seeds.
    Computes pairwise Spearman rank correlations, Top-k Jaccard overlaps,
    standard error, and 95% confidence intervals.
    """
    attributions = []
    top_k_sets = []

    for m in models:
        adapter = m if isinstance(m, ModelTrackAdapter) else ModelTrackAdapter(m)
        adapter.eval()
        ig = IntegratedGradients(adapter)
        res = ig.attribute(x, A, target_stock_idx=target_stock_idx, steps=ig_steps, method="gauss_legendre")
        attr = res['feature_importance_abs']
        attributions.append(attr)

        top_indices = set(np.argsort(attr)[::-1][:top_k])
        top_k_sets.append(top_indices)

    num_models = len(models)
    pairwise_corrs = []
    pairwise_jaccards = []

    for i in range(num_models):
        for j in range(i + 1, num_models):
            corr, _ = stats.spearmanr(attributions[i], attributions[j])
            pairwise_corrs.append(float(corr))

            set_i = top_k_sets[i]
            set_j = top_k_sets[j]
            jaccard = len(set_i.intersection(set_j)) / (len(set_i.union(set_j)) + 1e-8)
            pairwise_jaccards.append(float(jaccard))

    num_pairs = len(pairwise_corrs)
    if num_pairs > 0:
        mean_corr = float(np.mean(pairwise_corrs))
        std_corr = float(np.std(pairwise_corrs))
        se_corr = float(std_corr / np.sqrt(num_pairs))
        ci_95_corr = [float(mean_corr - 1.96 * se_corr), float(mean_corr + 1.96 * se_corr)]

        mean_jaccard = float(np.mean(pairwise_jaccards))
        std_jaccard = float(np.std(pairwise_jaccards))
    else:
        mean_corr, std_corr, se_corr = 0.0, 0.0, 0.0
        ci_95_corr = [0.0, 0.0]
        mean_jaccard, std_jaccard = 0.0, 0.0

    is_stable = bool(num_models >= 2 and mean_corr >= 0.60 and mean_jaccard >= 0.50)

    return {
        'num_seeds': num_models,
        'num_pairs_evaluated': num_pairs,
        'mean_pairwise_spearman_rank_correlation': mean_corr,
        'std_pairwise_spearman_rank_correlation': std_corr,
        'se_pairwise_spearman_rank_correlation': se_corr,
        'ci_95_spearman_rank_correlation': ci_95_corr,
        'mean_top_k_jaccard_overlap': mean_jaccard,
        'std_top_k_jaccard_overlap': std_jaccard,
        'top_k_features_per_seed': [list(s) for s in top_k_sets],
        'is_stable': is_stable
    }


def evaluate_perturbation_stability(
    model: Union[nn.Module, ModelTrackAdapter],
    x: torch.Tensor,
    A: torch.Tensor,
    target_stock_idx: int,
    noise_std: float = 0.02,
    num_trials: int = 25,
    min_valid_trials: int = 10,
    top_k: int = 5,
    ig_steps: int = 25,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Measure explanation robustness to small, label-preserving Gaussian noise.
    Strictly guards against overstatement: requires min_valid_trials prediction-preserving runs.
    """
    adapter = model if isinstance(model, ModelTrackAdapter) else ModelTrackAdapter(model)
    adapter.eval()
    ig = IntegratedGradients(adapter)

    # Clean attribution
    clean_res = ig.attribute(x, A, target_stock_idx=target_stock_idx, steps=ig_steps, method="gauss_legendre")
    clean_attr = clean_res['feature_importance_abs']
    clean_logit = clean_res['target_logit']
    clean_pred = (clean_logit >= 0.0)
    clean_top_k = set(np.argsort(clean_attr)[::-1][:top_k])
    clean_norm = clean_attr / (np.linalg.norm(clean_attr) + 1e-8)

    torch.manual_seed(seed)
    correlations = []
    cosine_sims = []
    jaccards = []
    prediction_flips = 0

    for _ in range(num_trials):
        noise = torch.randn_like(x) * noise_std
        x_noisy = x + noise

        # Check prediction preservation
        with torch.no_grad():
            noisy_logit = adapter(x_noisy, A)[0, target_stock_idx].item()
            noisy_pred = (noisy_logit >= 0.0)

        if noisy_pred == clean_pred:
            res_noisy = ig.attribute(x_noisy, A, target_stock_idx=target_stock_idx, steps=ig_steps, method="gauss_legendre")
            noisy_attr = res_noisy['feature_importance_abs']

            corr, _ = stats.spearmanr(clean_attr, noisy_attr)
            correlations.append(float(corr))

            noisy_norm = noisy_attr / (np.linalg.norm(noisy_attr) + 1e-8)
            cosine_sims.append(float(np.dot(clean_norm, noisy_norm)))

            noisy_top_k = set(np.argsort(noisy_attr)[::-1][:top_k])
            jaccards.append(len(clean_top_k.intersection(noisy_top_k)) / (len(clean_top_k.union(noisy_top_k)) + 1e-8))
        else:
            prediction_flips += 1

    valid_count = len(correlations)

    # Hard guard: do not claim robustness if insufficient valid trials
    if valid_count < min_valid_trials:
        return {
            'noise_std': noise_std,
            'total_trials': num_trials,
            'valid_trials': valid_count,
            'prediction_flips': prediction_flips,
            'status': 'insufficient_valid_trials',
            'mean_cosine_similarity': float(np.mean(cosine_sims)) if valid_count > 0 else 0.0,
            'mean_spearman_correlation': float(np.mean(correlations)) if valid_count > 0 else 0.0,
            'mean_top_k_jaccard_overlap': float(np.mean(jaccards)) if valid_count > 0 else 0.0,
            'is_robust': False,
            'diagnosis': (
                f"FAILED: Only {valid_count}/{num_trials} trials preserved predictions under noise_std={noise_std} "
                f"(minimum required: {min_valid_trials}). Prediction was too fragile near decision boundary."
            )
        }

    mean_corr = float(np.mean(correlations))
    std_corr = float(np.std(correlations))
    se_corr = float(std_corr / np.sqrt(valid_count))
    ci_95_corr = [float(mean_corr - 1.96 * se_corr), float(mean_corr + 1.96 * se_corr)]

    mean_cos = float(np.mean(cosine_sims))
    std_cos = float(np.std(cosine_sims))
    se_cos = float(std_cos / np.sqrt(valid_count))
    ci_95_cos = [float(mean_cos - 1.96 * se_cos), float(mean_cos + 1.96 * se_cos)]

    mean_jaccard = float(np.mean(jaccards))

    is_robust = bool(mean_cos >= 0.90 and mean_corr >= 0.70)

    return {
        'noise_std': noise_std,
        'total_trials': num_trials,
        'valid_trials': valid_count,
        'prediction_flips': prediction_flips,
        'status': 'verified',
        'mean_cosine_similarity': mean_cos,
        'std_cosine_similarity': std_cos,
        'se_cosine_similarity': se_cos,
        'ci_95_cosine_similarity': ci_95_cos,
        'mean_spearman_correlation': mean_corr,
        'std_spearman_correlation': std_corr,
        'se_spearman_correlation': se_corr,
        'ci_95_spearman_correlation': ci_95_corr,
        'mean_top_k_jaccard_overlap': mean_jaccard,
        'is_robust': is_robust,
        'diagnosis': (
            f"PASSED: Explanations are robust under Gaussian perturbation (cosine: {mean_cos:.4f}, rho: {mean_corr:.4f})."
            if is_robust else
            f"WARNING: Explanations exhibited perturbation sensitivity (rho: {mean_corr:.4f} < 0.70)."
        )
    }
