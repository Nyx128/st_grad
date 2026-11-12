"""
Integrated Gradients Attribution for ST-GRAD (Phase 6 & 7).
Implements axiomatic path-integral attribution (Sundararajan et al., 2017)
across features (F=24), historical time steps (T=30), and stock nodes (N=50).
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class IntegratedGradients:
    """
    Computes Integrated Gradients for pre-sigmoid logits of ST-GRAD.
    Attributes predictions to input features, time steps, and graph nodes.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def attribute(
        self,
        x: torch.Tensor,                    # [1, F, T, N]
        A: torch.Tensor,                    # [N, N]
        target_stock_idx: int,             # Index of the stock being explained (0 to N-1)
        baseline: Optional[torch.Tensor] = None,  # [1, F, T, N] or None (defaults to zeros)
        steps: int = 30
    ) -> Dict[str, Any]:
        """
        Compute Integrated Gradients attribution for target_stock_idx.

        Returns:
            Dictionary containing:
                'attributions': [F, T, N] tensor of signed attributions
                'target_logit': float, original model logit for target stock
                'baseline_logit': float, model logit on baseline input
                'delta_logit': float, target_logit - baseline_logit
                'sum_attributions': float, sum of all attributions
                'convergence_delta': float, completeness check error
                'feature_importance': [F] signed attribution per indicator
                'feature_importance_abs': [F] magnitude attribution per indicator
                'temporal_importance': [T] attribution across 30 days
                'spatial_importance': [N] attribution across all 50 stocks
                'target_stock_attribution': [F, T] attribution for target stock alone
        """
        device = x.device
        if baseline is None:
            baseline = torch.zeros_like(x, device=device)

        # Baseline and target forward evaluations
        with torch.no_grad():
            orig_logit = self.model(x, A)[0, target_stock_idx].item()
            base_logit = self.model(baseline, A)[0, target_stock_idx].item()
            delta_logit = orig_logit - base_logit

        # Generate interpolated inputs along linear path
        # x_alpha = baseline + alpha * (x - baseline) for alpha in (1/steps .. 1.0)
        alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device)
        accumulated_grads = torch.zeros_like(x)

        for alpha in alphas:
            x_step = baseline + alpha * (x - baseline)
            x_step = x_step.clone().detach().requires_grad_(True)

            logits = self.model(x_step, A)
            target_out = logits[0, target_stock_idx]

            grad = torch.autograd.grad(target_out, x_step)[0]
            accumulated_grads += grad

        # Average gradients and multiply by input difference (x - baseline)
        avg_grads = accumulated_grads / steps
        attributions = (x - baseline) * avg_grads  # [1, F, T, N]
        attr_tensor = attributions[0].detach()     # [F, T, N]

        sum_attr = attr_tensor.sum().item()
        abs_conv_delta = abs(sum_attr - delta_logit)
        conv_delta = abs_conv_delta / (abs(delta_logit) + 1e-4)

        # 1. Feature importance [F]: sum across time and stocks
        feat_imp_signed = attr_tensor.sum(dim=(1, 2)).cpu().numpy()
        feat_imp_abs = attr_tensor.abs().sum(dim=(1, 2)).cpu().numpy()

        # 2. Temporal importance [T]: sum across features and stocks
        temp_imp = attr_tensor.sum(dim=(0, 2)).cpu().numpy()
        temp_imp_abs = attr_tensor.abs().sum(dim=(0, 2)).cpu().numpy()

        # 3. Spatial importance [N]: sum across features and time
        spatial_imp = attr_tensor.sum(dim=(0, 1)).cpu().numpy()
        spatial_imp_abs = attr_tensor.abs().sum(dim=(0, 1)).cpu().numpy()

        # 4. Target stock's own feature-temporal matrix [F, T]
        target_stock_attr = attr_tensor[:, :, target_stock_idx].cpu().numpy()

        return {
            'attributions': attr_tensor.cpu().numpy(),
            'target_logit': orig_logit,
            'baseline_logit': base_logit,
            'delta_logit': delta_logit,
            'sum_attributions': sum_attr,
            'convergence_delta': conv_delta,
            'absolute_convergence_delta': abs_conv_delta,
            'feature_importance_signed': feat_imp_signed,
            'feature_importance_abs': feat_imp_abs,
            'temporal_importance': temp_imp,
            'temporal_importance_abs': temp_imp_abs,
            'spatial_importance': spatial_imp,
            'spatial_importance_abs': spatial_imp_abs,
            'target_stock_attribution': target_stock_attr
        }
