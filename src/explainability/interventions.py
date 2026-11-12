"""
Intervention Modules for ST-GRAD (Phase 6 & 7).
Implements:
1. MVCA View-Level Interventions (ST, DS, DT counterfactual masking).
2. Graph-Level Interventions (Static prior vs Adaptive graph ablation and neighbor node interventions).
All interventions operate non-destructively via PyTorch forward hooks or input manipulations.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn


class ViewInterventions:
    """
    Instance-level and dataset-level interventions on the three MVCA views:
    - Spatio-Temporal view (ST)
    - Feature-Spatial view (DS)
    - Feature-Temporal view (DT)
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def evaluate_view_contributions(
        self,
        x: torch.Tensor,                    # [1, F, T, N]
        A: torch.Tensor,                    # [N, N]
        target_stock_idx: int,
        neutral_mode: str = "zero"          # "zero" or "mean"
    ) -> Dict[str, Any]:
        """
        Measure logit and probability changes when ablating ST, DS, DT views individually and jointly.
        """
        # Baseline prediction without intervention
        with torch.no_grad():
            orig_logit = self.model(x, A)[0, target_stock_idx].item()
            orig_prob = torch.sigmoid(torch.tensor(orig_logit)).item()

        view_names = ['ST', 'DS', 'DT']
        results = {
            'orig_logit': orig_logit,
            'orig_prob': orig_prob,
            'interventions': {}
        }

        # Test individual view ablations
        for view in view_names:
            hooks = []
            try:
                for block in self.model.st_blocks:
                    target_module = getattr(block, f"{view.lower()}_view")

                    def make_hook():
                        def hook(mod, input, output):
                            if neutral_mode == "zero":
                                return torch.zeros_like(output)
                            else:
                                return torch.mean(output, dim=1, keepdim=True).expand_as(output)
                        return hook

                    h = target_module.register_forward_hook(make_hook())
                    hooks.append(h)

                with torch.no_grad():
                    ablated_logit = self.model(x, A)[0, target_stock_idx].item()
                    ablated_prob = torch.sigmoid(torch.tensor(ablated_logit)).item()

                delta_logit = orig_logit - ablated_logit
                delta_prob = orig_prob - ablated_prob

                results['interventions'][view] = {
                    'ablated_logit': ablated_logit,
                    'ablated_prob': ablated_prob,
                    'delta_logit': delta_logit,
                    'delta_prob': delta_prob,
                    'importance_rank_score': abs(delta_logit)
                }
            finally:
                for h in hooks:
                    h.remove()

        # Rank views by impact on logit
        ranked = sorted(
            view_names,
            key=lambda v: results['interventions'][v]['importance_rank_score'],
            reverse=True
        )
        results['view_importance_ranking'] = ranked

        # Extract global softmax fusion weights for comparison
        fusion_weights = []
        for i, block in enumerate(self.model.st_blocks):
            alphas = torch.tensor([block.alpha_st.item(), block.alpha_ds.item(), block.alpha_dt.item()])
            weights = torch.softmax(alphas, dim=0).cpu().numpy().tolist()
            fusion_weights.append({
                'block_idx': i,
                'st_weight': weights[0],
                'ds_weight': weights[1],
                'dt_weight': weights[2]
            })
        results['global_fusion_weights'] = fusion_weights

        return results


class GraphInterventions:
    """
    Graph-level counterfactual interventions:
    - Static Prior vs Adaptive Graph ablation
    - Node-level peer stock masking
    - Edge rewiring and cluster cut interventions
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def evaluate_graph_channels(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        target_stock_idx: int
    ) -> Dict[str, Any]:
        """
        Disentangle contributions of Static Prior graph vs Adaptive dynamic graph.
        """
        with torch.no_grad():
            orig_logit = self.model(x, A)[0, target_stock_idx].item()
            orig_prob = torch.sigmoid(torch.tensor(orig_logit)).item()

        # 1. Ablate Static Prior (replace A with Identity - no cross-stock static diffusion)
        I_adj = torch.eye(A.shape[0], device=A.device)
        with torch.no_grad():
            no_static_logit = self.model(x, I_adj)[0, target_stock_idx].item()
            no_static_prob = torch.sigmoid(torch.tensor(no_static_logit)).item()

        # 2. Ablate Adaptive Graph (hook adaptive_adj to return zero)
        hooks = []
        try:
            for block in self.model.st_blocks:
                h = block.diffusion_conv.adaptive_adj.register_forward_hook(
                    lambda mod, inp, out: torch.zeros_like(out)
                )
                hooks.append(h)

            with torch.no_grad():
                no_adapt_logit = self.model(x, A)[0, target_stock_idx].item()
                no_adapt_prob = torch.sigmoid(torch.tensor(no_adapt_logit)).item()
        finally:
            for h in hooks:
                h.remove()

        # 3. Ablate BOTH (pure isolated time series without spatial diffusion)
        try:
            for block in self.model.st_blocks:
                h = block.diffusion_conv.adaptive_adj.register_forward_hook(
                    lambda mod, inp, out: torch.zeros_like(out)
                )
                hooks.append(h)

            with torch.no_grad():
                isolated_logit = self.model(x, I_adj)[0, target_stock_idx].item()
                isolated_prob = torch.sigmoid(torch.tensor(isolated_logit)).item()
        finally:
            for h in hooks:
                h.remove()

        return {
            'orig_logit': orig_logit,
            'orig_prob': orig_prob,
            'static_prior_ablation': {
                'logit': no_static_logit,
                'delta_logit': orig_logit - no_static_logit,
                'delta_prob': orig_prob - no_static_prob
            },
            'adaptive_graph_ablation': {
                'logit': no_adapt_logit,
                'delta_logit': orig_logit - no_adapt_logit,
                'delta_prob': orig_prob - no_adapt_prob
            },
            'total_spatial_ablation': {
                'logit': isolated_logit,
                'delta_logit': orig_logit - isolated_logit,
                'delta_prob': orig_prob - isolated_prob
            }
        }

    def evaluate_neighbor_node_influence(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        target_stock_idx: int,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Measure the impact of masking peer stocks j != target_stock_idx on target prediction.
        """
        with torch.no_grad():
            orig_logit = self.model(x, A)[0, target_stock_idx].item()

        num_stocks = x.shape[3]
        node_deltas = {}

        for j in range(num_stocks):
            if j == target_stock_idx:
                continue

            # Mask stock j's input features to 0
            x_masked = x.clone()
            x_masked[:, :, :, j] = 0.0

            # Mask stock j's graph edges to target
            A_masked = A.clone()
            A_masked[target_stock_idx, j] = 0.0
            A_masked[j, target_stock_idx] = 0.0

            with torch.no_grad():
                masked_logit = self.model(x_masked, A_masked)[0, target_stock_idx].item()
                node_deltas[j] = abs(orig_logit - masked_logit)

        # Sort peer nodes by influence
        ranked_peers = sorted(node_deltas.items(), key=lambda item: item[1], reverse=True)
        top_peers = ranked_peers[:top_k]

        return {
            'target_stock_idx': target_stock_idx,
            'orig_logit': orig_logit,
            'top_influential_peers': [
                {'stock_idx': p[0], 'delta_logit': float(p[1])} for p in top_peers
            ],
            'mean_peer_delta': float(np.mean(list(node_deltas.values()))),
            'max_peer_delta': float(np.max(list(node_deltas.values())))
        }
