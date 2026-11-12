"""
Faithfulness Verification Suite for ST-GRAD Explanations (Phase 6 & 7).
Implements:
1. Feature Deletion and Insertion Curves (AOPC - Area Over the Perturbation Curve).
2. Model Parameter Randomization Sanity Checks (Adebayo et al., NeurIPS 2018).
"""

import copy
from typing import Dict, Any, List
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from src.explainability.attribution import IntegratedGradients


def deletion_insertion_test(
    model: nn.Module,
    x: torch.Tensor,                    # [1, F, T, N]
    A: torch.Tensor,                    # [N, N]
    target_stock_idx: int,
    feature_attributions: np.ndarray,   # [F] array of feature attribution magnitudes
    steps: List[int] = [1, 2, 4, 8, 12, 16, 20],
    num_random_trials: int = 5
) -> Dict[str, Any]:
    """
    Evaluate explanation faithfulness via feature deletion and insertion curves.
    A faithful explanation will show that deleting top-attributed features causes
    significantly steeper logit degradation than random or bottom-attributed features.
    """
    model.eval()
    num_features = len(feature_attributions)

    with torch.no_grad():
        orig_logit = model(x, A)[0, target_stock_idx].item()
        base_x = torch.zeros_like(x)
        base_logit = model(base_x, A)[0, target_stock_idx].item()

    # Rank feature indices from highest to lowest attribution (copy to ensure positive contiguous stride)
    ranked_top = np.argsort(feature_attributions)[::-1].copy()
    ranked_bottom = np.argsort(feature_attributions).copy()

    deletion_curve_top = []
    deletion_curve_bottom = []
    deletion_curve_random = []

    insertion_curve_top = []

    for k in steps:
        if k > num_features:
            continue

        # 1. Top-k deletion
        top_k_indices = ranked_top[:k]
        x_del_top = x.clone()
        x_del_top[:, top_k_indices, :, :] = 0.0
        with torch.no_grad():
            logit_del_top = model(x_del_top, A)[0, target_stock_idx].item()
        deletion_curve_top.append(orig_logit - logit_del_top)

        # 2. Bottom-k deletion
        bottom_k_indices = ranked_bottom[:k]
        x_del_bottom = x.clone()
        x_del_bottom[:, bottom_k_indices, :, :] = 0.0
        with torch.no_grad():
            logit_del_bottom = model(x_del_bottom, A)[0, target_stock_idx].item()
        deletion_curve_bottom.append(orig_logit - logit_del_bottom)

        # 3. Random-k deletion control (averaged over trials)
        rand_deltas = []
        for _ in range(num_random_trials):
            rand_indices = np.random.choice(num_features, size=k, replace=False)
            x_del_rand = x.clone()
            x_del_rand[:, rand_indices, :, :] = 0.0
            with torch.no_grad():
                l_rand = model(x_del_rand, A)[0, target_stock_idx].item()
            rand_deltas.append(orig_logit - l_rand)
        deletion_curve_random.append(float(np.mean(rand_deltas)))

        # 4. Top-k insertion (from zero baseline)
        x_ins_top = base_x.clone()
        x_ins_top[:, top_k_indices, :, :] = x[:, top_k_indices, :, :]
        with torch.no_grad():
            logit_ins_top = model(x_ins_top, A)[0, target_stock_idx].item()
        insertion_curve_top.append(logit_ins_top - base_logit)

    # Compute Area Under Curves
    auc_del_top = float(np.mean(deletion_curve_top))
    auc_del_random = float(np.mean(deletion_curve_random))
    auc_del_bottom = float(np.mean(deletion_curve_bottom))
    auc_ins_top = float(np.mean(insertion_curve_top))

    # Faithfulness score: top deletion should cause greater drop than random
    faithfulness_ratio = (auc_del_top + 1e-6) / (auc_del_random + 1e-6)

    return {
        'steps': steps,
        'orig_logit': orig_logit,
        'deletion_top': deletion_curve_top,
        'deletion_random': deletion_curve_random,
        'deletion_bottom': deletion_curve_bottom,
        'insertion_top': insertion_curve_top,
        'auc_deletion_top': auc_del_top,
        'auc_deletion_random': auc_del_random,
        'auc_deletion_bottom': auc_del_bottom,
        'auc_insertion_top': auc_ins_top,
        'faithfulness_ratio': faithfulness_ratio,
        'is_faithful': bool(auc_del_top > auc_del_random)
    }


def model_randomization_test(
    model: nn.Module,
    x: torch.Tensor,
    A: torch.Tensor,
    target_stock_idx: int,
    original_attributions: np.ndarray,  # [F] original feature importance
    seed: int = 42
) -> Dict[str, Any]:
    """
    Cascading Model Parameter Randomization Sanity Check (Adebayo et al., 2018).
    Randomizes model weights and recomputes attribution to ensure the explanation
    strictly depends on learned parameters and is not insensitive to model state.
    """
    torch.manual_seed(seed)
    randomized_model = copy.deepcopy(model)
    randomized_model.eval()

    # Re-initialize output_conv and top st_block parameters
    with torch.no_grad():
        for p in randomized_model.output_conv.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
        for p in randomized_model.st_blocks[-1].parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    ig = IntegratedGradients(randomized_model)
    res_rand = ig.attribute(x, A, target_stock_idx=target_stock_idx, steps=20)
    rand_attributions = res_rand['feature_importance_abs']

    # Compute correlation between real and randomized attributions
    orig_norm = original_attributions / (np.linalg.norm(original_attributions) + 1e-8)
    rand_norm = rand_attributions / (np.linalg.norm(rand_attributions) + 1e-8)

    cosine_sim = float(np.dot(orig_norm, rand_norm))
    spearman_corr, p_value = stats.spearmanr(original_attributions, rand_attributions)

    # Sanity check passes if correlation drops significantly (typically < 0.5)
    passed_sanity_check = bool(abs(spearman_corr) < 0.50 and cosine_sim < 0.70)

    return {
        'cosine_similarity': cosine_sim,
        'spearman_correlation': float(spearman_corr),
        'p_value': float(p_value),
        'passed_sanity_check': passed_sanity_check,
        'diagnosis': (
            "PASSED: Attribution collapsed upon parameter randomization, proving dependence on learned representations."
            if passed_sanity_check else
            "WARNING: Attribution persisted after parameter randomization."
        )
    }
