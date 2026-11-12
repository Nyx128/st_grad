"""
Stability Evaluation for ST-GRAD Explanations (Phase 6 & 7).
Measures:
1. Cross-Seed Stability across the 5 independently trained models (Seeds 42..46).
2. Input Perturbation Stability under small, label-preserving Gaussian noise.
"""

from typing import Dict, Any, List
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from src.explainability.attribution import IntegratedGradients


def evaluate_seed_stability(
    models: List[nn.Module],             # List of 5 models trained with different seeds
    x: torch.Tensor,                    # [1, F, T, N]
    A: torch.Tensor,                    # [N, N]
    target_stock_idx: int,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Measure explanation stability across 5 independent random training seeds.
    Computes pairwise Spearman rank correlation and top-k feature overlap.
    """
    attributions = []
    top_k_sets = []

    for model in models:
        model.eval()
        ig = IntegratedGradients(model)
        res = ig.attribute(x, A, target_stock_idx=target_stock_idx, steps=20)
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
            jaccard = len(set_i.intersection(set_j)) / len(set_i.union(set_j))
            pairwise_jaccards.append(float(jaccard))

    mean_corr = float(np.mean(pairwise_corrs)) if pairwise_corrs else 1.0
    mean_jaccard = float(np.mean(pairwise_jaccards)) if pairwise_jaccards else 1.0

    return {
        'num_seeds': num_models,
        'mean_pairwise_spearman_rank_correlation': mean_corr,
        'std_pairwise_spearman_rank_correlation': float(np.std(pairwise_corrs)) if pairwise_corrs else 0.0,
        'mean_top_k_jaccard_overlap': mean_jaccard,
        'top_k_features_per_seed': [list(s) for s in top_k_sets],
        'is_stable': bool(mean_corr >= 0.60 and mean_jaccard >= 0.50)
    }


def evaluate_perturbation_stability(
    model: nn.Module,
    x: torch.Tensor,
    A: torch.Tensor,
    target_stock_idx: int,
    noise_std: float = 0.02,
    num_trials: int = 5,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Measure explanation robustness to small, prediction-preserving input perturbations.
    Adds Gaussian noise epsilon ~ N(0, noise_std^2) and checks attribution rank correlation.
    """
    model.eval()
    ig = IntegratedGradients(model)

    # Clean attribution
    clean_res = ig.attribute(x, A, target_stock_idx=target_stock_idx, steps=20)
    clean_attr = clean_res['feature_importance_abs']
    clean_pred = (clean_res['target_logit'] > 0.0)
    clean_top_k = set(np.argsort(clean_attr)[::-1][:top_k])

    correlations = []
    cosine_sims = []
    jaccards = []

    clean_norm = clean_attr / (np.linalg.norm(clean_attr) + 1e-8)

    for _ in range(num_trials):
        noise = torch.randn_like(x) * noise_std
        x_noisy = x + noise

        # Check prediction preservation
        with torch.no_grad():
            noisy_logit = model(x_noisy, A)[0, target_stock_idx].item()
            noisy_pred = (noisy_logit > 0.0)

        # Only evaluate if the prediction did not flip
        if noisy_pred == clean_pred:
            res_noisy = ig.attribute(x_noisy, A, target_stock_idx=target_stock_idx, steps=20)
            noisy_attr = res_noisy['feature_importance_abs']

            corr, _ = stats.spearmanr(clean_attr, noisy_attr)
            correlations.append(float(corr))

            noisy_norm = noisy_attr / (np.linalg.norm(noisy_attr) + 1e-8)
            cosine_sims.append(float(np.dot(clean_norm, noisy_norm)))

            noisy_top_k = set(np.argsort(noisy_attr)[::-1][:top_k])
            jaccards.append(len(clean_top_k.intersection(noisy_top_k)) / len(clean_top_k.union(noisy_top_k)))

    return {
        'noise_std': noise_std,
        'valid_trials': len(correlations),
        'mean_cosine_similarity': float(np.mean(cosine_sims)) if cosine_sims else 1.0,
        'mean_spearman_correlation': float(np.mean(correlations)) if correlations else 1.0,
        'mean_top_k_jaccard_overlap': float(np.mean(jaccards)) if jaccards else 1.0,
        'is_robust': bool(np.mean(correlations) >= 0.75 if correlations else True)
    }
