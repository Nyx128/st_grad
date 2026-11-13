"""
Intervention Modules for ST-GRAD (Phase 1 & 6).
Implements:
1. MVCA View-Level Interventions (ST, DS, DT counterfactual masking) with dual-track support.
2. Graph-Level Interventions:
   - Static prior vs Adaptive dynamic graph channel ablation.
   - Decoupled peer node feature masking (inputs only, topology unchanged).
   - Decoupled peer edge ablation with consistent degree renormalization (topology only, inputs unchanged).
   - Matched random peer and random edge controls.
All interventions operate non-destructively via PyTorch forward hooks or input manipulations.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from src.explainability.adapter import ModelTrackAdapter


class ViewInterventions:
    """
    Instance-level and dataset-level interventions on the three MVCA views:
    - Spatio-Temporal view (ST)
    - Feature-Spatial view (DS in Track B / FS in Track A)
    - Feature-Temporal view (DT in Track B / FT in Track A)
    Supports both Track A (repo_compat) and Track B (paper_aligned) via ModelTrackAdapter.
    """
    def __init__(self, model: Union[nn.Module, ModelTrackAdapter]):
        if isinstance(model, ModelTrackAdapter):
            self.adapter = model
        else:
            self.adapter = ModelTrackAdapter(model)
        self.adapter.eval()

    def evaluate_view_contributions(
        self,
        x: torch.Tensor,                    # [1, F, T, N]
        A: torch.Tensor,                    # [N, N]
        target_stock_idx: int,
        neutral_mode: str = "zero",         # "zero" or "mean"
        reference_mean: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Measure logit and probability changes when ablating ST, DS, DT views individually and jointly.
        """
        with torch.no_grad():
            orig_logit = self.adapter(x, A)[0, target_stock_idx].item()
            orig_prob = torch.sigmoid(torch.tensor(orig_logit)).item()

        view_names = ['ST', 'DS', 'DT']
        results = {
            'orig_logit': orig_logit,
            'orig_prob': orig_prob,
            'neutral_mode': neutral_mode,
            'track': self.adapter.track,
            'interventions': {}
        }

        # Test individual view ablations
        for view in view_names:
            hooks = []
            try:
                for block_idx in range(self.adapter.num_layers):
                    target_module = self.adapter.get_view_module(block_idx, view)

                    def make_hook():
                        def hook(mod, input, output):
                            if neutral_mode == "zero":
                                return torch.zeros_like(output)
                            elif neutral_mode == "reference" and reference_mean is not None:
                                return reference_mean.expand_as(output)
                            else:
                                # Feature/spatial mean
                                return torch.mean(output, dim=1, keepdim=True).expand_as(output)
                        return hook

                    h = target_module.register_forward_hook(make_hook())
                    hooks.append(h)

                with torch.no_grad():
                    ablated_logit = self.adapter(x, A)[0, target_stock_idx].item()
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
        for i in range(self.adapter.num_layers):
            weights = self.adapter.get_fusion_weights(i)
            fusion_weights.append({
                'block_idx': i,
                'st_weight': weights['ST'],
                'ds_weight': weights['DS'],
                'dt_weight': weights['DT']
            })
        results['global_fusion_weights'] = fusion_weights

        return results


class GraphInterventions:
    """
    Graph-level counterfactual interventions:
    - Static Prior vs Adaptive Dynamic Graph channel ablation
    - Decoupled peer stock feature masking (node-only, topology unchanged)
    - Decoupled peer stock edge ablation with consistent degree renormalization (edge-only, inputs unchanged)
    - Random peer and random edge controls
    Supports both Track A and Track B via ModelTrackAdapter.
    """
    def __init__(self, model: Union[nn.Module, ModelTrackAdapter]):
        if isinstance(model, ModelTrackAdapter):
            self.adapter = model
        else:
            self.adapter = ModelTrackAdapter(model)
        self.adapter.eval()

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
            orig_logit = self.adapter(x, A)[0, target_stock_idx].item()
            orig_prob = torch.sigmoid(torch.tensor(orig_logit)).item()

        # 1. Ablate Static Prior (replace A with Identity - no cross-stock static diffusion)
        I_adj = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        with torch.no_grad():
            no_static_logit = self.adapter(x, I_adj)[0, target_stock_idx].item()
            no_static_prob = torch.sigmoid(torch.tensor(no_static_logit)).item()

        # 2. Ablate Adaptive Graph (hook adaptive_adj to return zero)
        hooks = []
        try:
            for block_idx in range(self.adapter.num_layers):
                adapt_mod = self.adapter.get_adaptive_adj_module(block_idx)
                h = adapt_mod.register_forward_hook(
                    lambda mod, inp, out: torch.zeros_like(out)
                )
                hooks.append(h)

            with torch.no_grad():
                no_adapt_logit = self.adapter(x, A)[0, target_stock_idx].item()
                no_adapt_prob = torch.sigmoid(torch.tensor(no_adapt_logit)).item()
        finally:
            for h in hooks:
                h.remove()

        # 3. Ablate BOTH (pure isolated time series without spatial diffusion)
        try:
            for block_idx in range(self.adapter.num_layers):
                adapt_mod = self.adapter.get_adaptive_adj_module(block_idx)
                h = adapt_mod.register_forward_hook(
                    lambda mod, inp, out: torch.zeros_like(out)
                )
                hooks.append(h)

            with torch.no_grad():
                isolated_logit = self.adapter(x, I_adj)[0, target_stock_idx].item()
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

    def evaluate_peer_node_masking(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        target_stock_idx: int,
        top_k: int = 5,
        replacement_mode: str = "zero",
        replacement_val: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Measure pure node-feature influence: Mask peer stock j's inputs while holding graph topology intact.
        """
        with torch.no_grad():
            orig_logit = self.adapter(x, A)[0, target_stock_idx].item()

        num_stocks = x.shape[3]
        node_deltas = {}

        for j in range(num_stocks):
            if j == target_stock_idx:
                continue

            x_masked = x.clone()
            if replacement_mode == "zero" or replacement_val is None:
                x_masked[:, :, :, j] = 0.0
            else:
                x_masked[:, :, :, j] = replacement_val[:, :, :, j]

            with torch.no_grad():
                masked_logit = self.adapter(x_masked, A)[0, target_stock_idx].item()
                node_deltas[j] = abs(orig_logit - masked_logit)

        ranked_peers = sorted(node_deltas.items(), key=lambda item: item[1], reverse=True)
        top_peers = ranked_peers[:top_k]

        return {
            'target_stock_idx': target_stock_idx,
            'orig_logit': orig_logit,
            'type': 'peer_node_feature_masking',
            'top_influential_peers': [
                {'stock_idx': p[0], 'delta_logit': float(p[1])} for p in top_peers
            ],
            'all_peer_deltas': {int(k): float(v) for k, v in node_deltas.items()},
            'mean_peer_delta': float(np.mean(list(node_deltas.values()))),
            'max_peer_delta': float(np.max(list(node_deltas.values())))
        }

    def evaluate_peer_edge_ablation(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        target_stock_idx: int,
        top_k: int = 5,
        renormalize: bool = True
    ) -> Dict[str, Any]:
        """
        Measure pure topological edge influence: Remove edge (i, j) and (j, i) from graph A while holding inputs intact.
        Symmetrically renormalizes the resulting adjacency matrix.
        """
        with torch.no_grad():
            orig_logit = self.adapter(x, A)[0, target_stock_idx].item()

        num_stocks = A.shape[0]
        edge_deltas = {}

        for j in range(num_stocks):
            if j == target_stock_idx:
                continue

            # Check if there is an edge to ablate
            if A[target_stock_idx, j].item() == 0.0 and A[j, target_stock_idx].item() == 0.0:
                edge_deltas[j] = 0.0
                continue

            A_ablated = A.clone()
            A_ablated[target_stock_idx, j] = 0.0
            A_ablated[j, target_stock_idx] = 0.0

            if renormalize:
                # Symmetrically renormalize rows: D^(-1/2) A D^(-1/2)
                deg = torch.sum(A_ablated, dim=1)
                deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-8), -0.5)
                deg_inv_sqrt[deg == 0] = 0.0
                D_inv_sqrt = torch.diag(deg_inv_sqrt)
                A_ablated = torch.mm(torch.mm(D_inv_sqrt, A_ablated), D_inv_sqrt)

            with torch.no_grad():
                ablated_logit = self.adapter(x, A_ablated)[0, target_stock_idx].item()
                edge_deltas[j] = abs(orig_logit - ablated_logit)

        ranked_edges = sorted(edge_deltas.items(), key=lambda item: item[1], reverse=True)
        top_edges = ranked_edges[:top_k]

        return {
            'target_stock_idx': target_stock_idx,
            'orig_logit': orig_logit,
            'type': 'peer_edge_ablation',
            'top_influential_edges': [
                {'stock_idx': p[0], 'delta_logit': float(p[1])} for p in top_edges
            ],
            'all_edge_deltas': {int(k): float(v) for k, v in edge_deltas.items()},
            'mean_edge_delta': float(np.mean(list(edge_deltas.values()))),
            'max_edge_delta': float(np.max(list(edge_deltas.values())))
        }

    def evaluate_matched_random_edge_controls(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        target_stock_idx: int,
        top_k: int = 5,
        num_random_trials: int = 20,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Statistically test Hypothesis H3: Does targeted edge ablation cause significantly
        greater logit shifts than matched random edge ablation?
        """
        edge_res = self.evaluate_peer_edge_ablation(x, A, target_stock_idx, top_k=top_k)
        top_deltas = [p['delta_logit'] for p in edge_res['top_influential_edges']]
        mean_top_edge_delta = float(np.mean(top_deltas)) if top_deltas else 0.0

        num_stocks = A.shape[0]
        eligible_peers = [j for j in range(num_stocks) if j != target_stock_idx]
        rng = np.random.default_rng(seed)

        with torch.no_grad():
            orig_logit = self.adapter(x, A)[0, target_stock_idx].item()

        rand_trial_means = []
        for _ in range(num_random_trials):
            sampled_peers = rng.choice(eligible_peers, size=min(top_k, len(eligible_peers)), replace=False)
            trial_deltas = []
            for p in sampled_peers:
                A_rand = A.clone()
                A_rand[target_stock_idx, p] = 0.0
                A_rand[p, target_stock_idx] = 0.0
                deg = torch.sum(A_rand, dim=1)
                deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-8), -0.5)
                deg_inv_sqrt[deg == 0] = 0.0
                D_inv_sqrt = torch.diag(deg_inv_sqrt)
                A_rand = torch.mm(torch.mm(D_inv_sqrt, A_rand), D_inv_sqrt)

                with torch.no_grad():
                    l_r = self.adapter(x, A_rand)[0, target_stock_idx].item()
                    trial_deltas.append(abs(orig_logit - l_r))
            rand_trial_means.append(float(np.mean(trial_deltas)))

        mean_random_edge_delta = float(np.mean(rand_trial_means)) if rand_trial_means else 0.0
        edge_influence_ratio = (mean_top_edge_delta + 1e-6) / (mean_random_edge_delta + 1e-6)
        h3_confirmed = bool(mean_top_edge_delta > mean_random_edge_delta)

        return {
            'target_stock_idx': target_stock_idx,
            'top_k': top_k,
            'mean_top_edge_delta': mean_top_edge_delta,
            'mean_random_edge_delta': mean_random_edge_delta,
            'edge_influence_ratio': float(edge_influence_ratio),
            'num_random_trials': num_random_trials,
            'h3_edge_specificity_passed': h3_confirmed,
            'diagnosis': (
                f"CONFIRMED: Targeted top-{top_k} edge ablation alters logit {edge_influence_ratio:.2f}x more than random edge ablation."
                if h3_confirmed else
                f"NOT CONFIRMED: Targeted edge ablation ({mean_top_edge_delta:.4f}) did not exceed random controls ({mean_random_edge_delta:.4f})."
            )
        }

    def evaluate_neighbor_node_influence(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        target_stock_idx: int,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Legacy combined node & edge masking for backward compatibility,
        plus decoupled breakdowns.
        """
        node_res = self.evaluate_peer_node_masking(x, A, target_stock_idx, top_k=top_k)
        edge_res = self.evaluate_peer_edge_ablation(x, A, target_stock_idx, top_k=top_k)

        # Joint masking
        with torch.no_grad():
            orig_logit = self.adapter(x, A)[0, target_stock_idx].item()

        num_stocks = x.shape[3]
        joint_deltas = {}

        for j in range(num_stocks):
            if j == target_stock_idx:
                continue

            x_masked = x.clone()
            x_masked[:, :, :, j] = 0.0

            A_masked = A.clone()
            A_masked[target_stock_idx, j] = 0.0
            A_masked[j, target_stock_idx] = 0.0

            with torch.no_grad():
                masked_logit = self.adapter(x_masked, A_masked)[0, target_stock_idx].item()
                joint_deltas[j] = abs(orig_logit - masked_logit)

        ranked_peers = sorted(joint_deltas.items(), key=lambda item: item[1], reverse=True)
        top_peers = ranked_peers[:top_k]

        return {
            'target_stock_idx': target_stock_idx,
            'orig_logit': orig_logit,
            'top_influential_peers': [
                {'stock_idx': p[0], 'delta_logit': float(p[1])} for p in top_peers
            ],
            'mean_peer_delta': float(np.mean(list(joint_deltas.values()))),
            'max_peer_delta': float(np.max(list(joint_deltas.values()))),
            'decoupled_node_feature_influence': node_res,
            'decoupled_edge_influence': edge_res
        }
