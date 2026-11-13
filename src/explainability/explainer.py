"""
ST-GRAD Explainer API Engine (Phase 1).
Provides the unified, high-level explainability interface:
    explainer = STGradExplainer(model="paper_aligned", baseline_strategy="train_mean")
    explanation = explainer.explain_instance(x, target_stock="NVDA", target_date="2024-11-05")

Aggregates:
- High-precision Gauss-Legendre Integrated Gradients attribution
- Directional decomposition (Bullish vs Bearish drivers)
- Intrinsic vs Spatial Peer attribution
- MVCA View counterfactual interventions (ST, DS, DT)
- Disentangled Graph interventions (Static Prior vs Adaptive Dynamic + Peer Nodes/Edges)
- Standardized Case Record export and human-readable Markdown Cards
- Strict model-agnosticism across Track A (repo_compat) and Track B (paper_aligned)
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.explainability.adapter import ModelTrackAdapter
from src.explainability.attribution import IntegratedGradients
from src.explainability.interventions import ViewInterventions, GraphInterventions
from src.paper_aligned.config import PaperAlignedConfig
from src.paper_aligned.features import FEATURE_NAMES
from src.paper_aligned.graph import get_prior_adjacency_matrix


class STGradExplainer:
    """
    Unified Explainability Engine for ST-GRAD Models.
    Works seamlessly with Track A (repo_compat) and Track B (paper_aligned).
    """
    def __init__(
        self,
        model: Optional[Union[nn.Module, ModelTrackAdapter, str]] = None,
        track: str = "paper_aligned",
        checkpoint_path: Optional[str] = None,
        adj_matrix: Optional[Union[torch.Tensor, np.ndarray, str]] = None,
        stock_order: Optional[Union[List[str], np.ndarray, str]] = None,
        feature_names: Optional[List[str]] = None,
        baseline_strategy: str = "zero",  # "zero" or "train_mean"
        train_mean: Optional[Union[torch.Tensor, np.ndarray]] = None,
        device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = PaperAlignedConfig()

        # 1. Model & Track Resolution
        if isinstance(model, ModelTrackAdapter):
            self.adapter = model
        elif isinstance(model, nn.Module):
            self.adapter = ModelTrackAdapter(model, track=track, checkpoint_path=checkpoint_path)
        elif isinstance(model, str):
            # Model string specifies track or checkpoint path
            if os.path.exists(model):
                self.adapter = ModelTrackAdapter.load(track=track, checkpoint_path=model, device=self.device)
            else:
                self.adapter = ModelTrackAdapter.load(track=model, checkpoint_path=checkpoint_path, device=self.device)
        else:
            # Default to loading track
            self.adapter = ModelTrackAdapter.load(track=track, checkpoint_path=checkpoint_path, device=self.device)

        self.adapter.to(self.device)
        self.adapter.eval()
        self.track = self.adapter.track

        # 2. Stock Ordering Resolution
        if stock_order is None:
            order_path = self.config.stock_order_path
            if os.path.exists(order_path):
                self.stock_order = list(np.load(order_path, allow_pickle=True))
            else:
                self.stock_order = [f"STOCK_{i:02d}" for i in range(self.adapter.num_nodes)]
        elif isinstance(stock_order, str):
            self.stock_order = list(np.load(stock_order, allow_pickle=True))
        else:
            self.stock_order = list(stock_order)

        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(self.stock_order)}
        self.idx_to_ticker = {i: ticker for i, ticker in enumerate(self.stock_order)}

        # 3. Adjacency Matrix Resolution
        if adj_matrix is None:
            if os.path.exists(self.config.cluster_file_path) and os.path.exists(self.config.stock_order_path):
                self.A = get_prior_adjacency_matrix(
                    self.config.cluster_file_path,
                    self.config.stock_order_path,
                    device=self.device
                )
            else:
                self.A = torch.eye(len(self.stock_order), device=self.device)
        elif isinstance(adj_matrix, str):
            self.A = torch.FloatTensor(np.load(adj_matrix)).to(self.device)
        elif isinstance(adj_matrix, np.ndarray):
            self.A = torch.FloatTensor(adj_matrix).to(self.device)
        else:
            self.A = adj_matrix.to(self.device)

        # 4. Feature Names
        self.feature_names = feature_names or FEATURE_NAMES

        # 5. Baseline Reference Setup
        self.baseline_strategy = baseline_strategy
        if train_mean is not None:
            if isinstance(train_mean, np.ndarray):
                self.train_mean = torch.FloatTensor(train_mean).to(self.device)
            else:
                self.train_mean = train_mean.to(self.device)
        else:
            # Attempt to load empirical train_mean from dataset artifact if available
            ds_path = os.path.join(self.config.artifacts_dir, "paper_aligned_dataset.npz")
            if os.path.exists(ds_path):
                try:
                    ds = np.load(ds_path, allow_pickle=True)
                    if 'train_mean' in ds:
                        # train_mean is [1, 1, 1, 24] -> permute to [1, 24, 1, 1]
                        tm = ds['train_mean']
                        if tm.ndim == 4 and tm.shape[-1] == len(self.feature_names):
                            tm = np.transpose(tm, (0, 3, 1, 2))
                        self.train_mean = torch.FloatTensor(tm).to(self.device)
                    else:
                        self.train_mean = None
                except Exception:
                    self.train_mean = None
            else:
                self.train_mean = None

        # 6. Initialize sub-explainer modules
        self.ig = IntegratedGradients(self.adapter)
        self.view_intervener = ViewInterventions(self.adapter)
        self.graph_intervener = GraphInterventions(self.adapter)

    def resolve_stock(self, stock: Union[int, str]) -> Tuple[int, str]:
        """Convert stock identifier (int or ticker string) to (index, ticker)."""
        if isinstance(stock, str):
            if stock not in self.ticker_to_idx:
                raise ValueError(f"Ticker '{stock}' not found in stock order universe.")
            idx = self.ticker_to_idx[stock]
            return idx, stock
        else:
            idx = int(stock)
            if not (0 <= idx < len(self.stock_order)):
                raise IndexError(f"Stock index {idx} out of range [0, {len(self.stock_order)}).")
            return idx, self.idx_to_ticker[idx]

    def _prepare_baseline(self, x: torch.Tensor, custom_baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Construct the reference baseline tensor [1, F, T, N]."""
        if custom_baseline is not None:
            return custom_baseline.to(self.device)

        if self.baseline_strategy == "train_mean" and self.train_mean is not None:
            # Broadcast train_mean [1, F, 1, 1] to match x [1, F, T, N]
            base = self.train_mean.expand_as(x).clone()
            return base
        else:
            return torch.zeros_like(x, device=self.device)

    def explain_instance(
        self,
        x: torch.Tensor,                                      # [1, F, T, N]
        target_stock: Union[int, str],
        A: Optional[torch.Tensor] = None,                    # Optional override
        target_date: Optional[str] = None,
        input_window_dates: Optional[Tuple[str, str]] = None,
        baseline: Optional[torch.Tensor] = None,
        ig_steps: int = 50,
        quadrature_method: str = "gauss_legendre",
        top_k_features: int = 5,
        top_k_peers: int = 5,
        include_views: bool = True,
        include_graph: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive, publication-grade explanation record for a single prediction.
        """
        stock_idx, ticker = self.resolve_stock(target_stock)
        adj = A.to(self.device) if A is not None else self.A
        x_in = x.to(self.device)
        if x_in.dim() == 3:
            x_in = x_in.unsqueeze(0)  # Add batch dim [1, F, T, N]

        # 1. Forward Model Prediction
        with torch.no_grad():
            logit = self.adapter(x_in, adj)[0, stock_idx].item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
            pred_class = 1 if prob >= 0.5 else 0
            direction = "UP" if pred_class == 1 else "DOWN"
            confidence = abs(prob - 0.5) * 2.0  # 0.0 to 1.0

        # 2. Baseline Preparation
        base_x = self._prepare_baseline(x_in, custom_baseline=baseline)

        # 3. Integrated Gradients Attribution
        attr_res = self.ig.attribute(
            x_in, adj,
            target_stock_idx=stock_idx,
            baseline=base_x,
            steps=ig_steps,
            method=quadrature_method,
            adaptive=True
        )

        feat_imp_signed = attr_res['feature_importance_signed']
        feat_imp_abs = attr_res['feature_importance_abs']

        # Rank features
        ranked_abs_indices = np.argsort(feat_imp_abs)[::-1]
        top_magnitude_features = [
            {
                'rank': r + 1,
                'feature': self.feature_names[i],
                'importance_score': float(feat_imp_abs[i]),
                'signed_impact': float(feat_imp_signed[i]),
                'direction': 'BULLISH' if feat_imp_signed[i] > 0 else 'BEARISH'
            }
            for r, i in enumerate(ranked_abs_indices[:top_k_features])
        ]

        # Top Bullish drivers (pushing prediction UP)
        bullish_indices = [i for i in np.argsort(feat_imp_signed)[::-1] if feat_imp_signed[i] > 0]
        top_bullish_features = [
            {'rank': r + 1, 'feature': self.feature_names[i], 'positive_contribution': float(feat_imp_signed[i])}
            for r, i in enumerate(bullish_indices[:top_k_features])
        ]

        # Top Bearish drivers (pushing prediction DOWN)
        bearish_indices = [i for i in np.argsort(feat_imp_signed) if feat_imp_signed[i] < 0]
        top_bearish_features = [
            {'rank': r + 1, 'feature': self.feature_names[i], 'negative_contribution': float(feat_imp_signed[i])}
            for r, i in enumerate(bearish_indices[:top_k_features])
        ]

        # Target Stock Intrinsic vs Peer Spatial Share
        target_stock_abs_sum = float(np.sum(np.abs(attr_res['target_stock_attribution'])))
        total_attr_abs_sum = float(np.sum(feat_imp_abs)) + 1e-8
        intrinsic_share = target_stock_abs_sum / total_attr_abs_sum

        # 4. MVCA View Interventions
        views_summary = None
        if include_views:
            views_res = self.view_intervener.evaluate_view_contributions(
                x_in, adj, target_stock_idx=stock_idx, neutral_mode="zero"
            )
            views_summary = {
                'importance_ranking': views_res['view_importance_ranking'],
                'interventions': {
                    v: {
                        'delta_logit': float(views_res['interventions'][v]['delta_logit']),
                        'delta_prob': float(views_res['interventions'][v]['delta_prob']),
                        'importance_score': float(views_res['interventions'][v]['importance_rank_score'])
                    }
                    for v in ['ST', 'DS', 'DT']
                },
                'global_fusion_weights': views_res['global_fusion_weights']
            }

        # 5. Graph Interventions
        graph_summary = None
        if include_graph:
            g_channels = self.graph_intervener.evaluate_graph_channels(x_in, adj, target_stock_idx=stock_idx)
            peer_nodes = self.graph_intervener.evaluate_peer_node_masking(x_in, adj, target_stock_idx=stock_idx, top_k=top_k_peers)
            peer_edges = self.graph_intervener.evaluate_peer_edge_ablation(x_in, adj, target_stock_idx=stock_idx, top_k=top_k_peers)

            graph_summary = {
                'static_prior_ablation_delta_logit': float(g_channels['static_prior_ablation']['delta_logit']),
                'adaptive_graph_ablation_delta_logit': float(g_channels['adaptive_graph_ablation']['delta_logit']),
                'total_spatial_ablation_delta_logit': float(g_channels['total_spatial_ablation']['delta_logit']),
                'top_influential_peer_tickers': [
                    {
                        'ticker': self.idx_to_ticker[p['stock_idx']],
                        'stock_idx': p['stock_idx'],
                        'delta_logit': float(p['delta_logit'])
                    }
                    for p in peer_nodes['top_influential_peers']
                ],
                'top_influential_edges': [
                    {
                        'peer_ticker': self.idx_to_ticker[p['stock_idx']],
                        'stock_idx': p['stock_idx'],
                        'delta_logit': float(p['delta_logit'])
                    }
                    for p in peer_edges['top_influential_edges']
                ]
            }

        # 6. Assemble Standardized Case Record
        record = {
            'ticker': ticker,
            'target_stock_idx': stock_idx,
            'target_date': target_date or "N/A",
            'input_window_dates': input_window_dates or ("N/A", "N/A"),
            'timestamp': datetime.now().isoformat(),
            'prediction': {
                'logit': logit,
                'probability': prob,
                'predicted_class': pred_class,
                'trend_direction': direction,
                'confidence': confidence
            },
            'attribution': {
                'top_magnitude_features': top_magnitude_features,
                'top_bullish_features': top_bullish_features,
                'top_bearish_features': top_bearish_features,
                'feature_importance_abs': [float(v) for v in feat_imp_abs],
                'feature_importance_signed': [float(v) for v in feat_imp_signed],
                'temporal_importance_abs': [float(v) for v in attr_res['temporal_importance_abs']],
                'temporal_recency_ratio': float(attr_res['temporal_recency_ratio']),
                'target_stock_intrinsic_share': float(intrinsic_share),
                'convergence': {
                    'is_converged': bool(attr_res['is_converged']),
                    'convergence_delta': float(attr_res['convergence_delta']),
                    'absolute_convergence_delta': float(attr_res['absolute_convergence_delta']),
                    'steps_used': int(attr_res['steps_used']),
                    'quadrature_method': quadrature_method,
                    'baseline_strategy': self.baseline_strategy
                }
            },
            'views': views_summary,
            'graph': graph_summary,
            'provenance': {
                'track': self.track,
                'checkpoint_hash': self.adapter.checkpoint_hash or "N/A",
                'device': str(self.device)
            },
            'caveats': [
                "Model attribution identifies features correlated with predictions, NOT real-world economic causality.",
                "Technical indicators exhibit multicollinearity; credit may be split among correlated proxies.",
                "Interventions measure model sensitivity under synthetic ablation and do not simulate market mechanics."
            ]
        }

        return record

    def format_markdown_card(self, record: Dict[str, Any]) -> str:
        """
        Format a case record into a clean, human-readable Markdown Report Card.
        """
        p = record['prediction']
        attr = record['attribution']
        views = record.get('views', {})
        graph = record.get('graph', {})

        md = []
        md.append(f"# ST-GRAD Prediction Explanation: {record['ticker']}")
        md.append(f"**Target Date:** {record['target_date']} | **Track:** `{record['provenance']['track']}` | **Model Hash:** `{record['provenance']['checkpoint_hash'][:10] if record['provenance']['checkpoint_hash'] != 'N/A' else 'N/A'}`\n")

        md.append("## 1. Directional Prediction Summary")
        md.append(f"- **Predicted Trend:** **`{p['trend_direction']}`** (Class `{p['predicted_class']}`)")
        md.append(f"- **Probability:** `{p['probability']:.4f}` (Confidence: `{p['confidence']*100:.1f}%`)")
        md.append(f"- **Pre-Sigmoid Logit:** `{p['logit']:.4f}`\n")

        md.append("## 2. Key Technical Indicator Drivers (Integrated Gradients)")
        md.append(f"*Convergence:* `{'PASSED' if attr['convergence']['is_converged'] else 'APPROXIMATE'}` (Delta: `{attr['convergence']['convergence_delta']*100:.2f}%` over `{attr['convergence']['steps_used']}` {attr['convergence']['quadrature_method']} steps)")
        md.append(f"*Intrinsic vs Peer Reliance:* `{attr['target_stock_intrinsic_share']*100:.1f}%` of attribution originates from {record['ticker']}'s own technicals.\n")

        md.append("| Rank | Technical Indicator | Absolute Score | Directional Impact |")
        md.append("|:---:|---|:---:|:---:|")
        for f in attr['top_magnitude_features']:
            badge = "🟢 Bullish (+)" if f['direction'] == 'BULLISH' else "🔴 Bearish (-)"
            md.append(f"| {f['rank']} | `{f['feature']}` | {f['importance_score']:.3f} | {badge} ({f['signed_impact']:+.3f}) |")
        md.append("")

        if views:
            md.append("## 3. Multi-View Cross-Attention (MVCA) Hierarchy")
            ranking_str = " > ".join([f"**{v}**" for v in views['importance_ranking']])
            md.append(f"- **View Impact Order:** {ranking_str}")
            for v, data in views['interventions'].items():
                md.append(f"  - `{v}` View Ablation Drop: `Δlogit = {data['delta_logit']:.4f}` (`Δprob = {data['delta_prob']:+.4f}`)")
            md.append("")

        if graph:
            md.append("## 4. Spatio-Temporal Graph & Peer Influence")
            md.append(f"- **Static Sector Prior Channel Drop:** `Δlogit = {graph['static_prior_ablation_delta_logit']:.4f}`")
            md.append(f"- **Adaptive Dynamic Graph Channel Drop:** `Δlogit = {graph['adaptive_graph_ablation_delta_logit']:.4f}`")
            if graph.get('top_influential_peer_tickers'):
                peer_list = ", ".join([f"`{p['ticker']}` (Δ{p['delta_logit']:.3f})" for p in graph['top_influential_peer_tickers'][:3]])
                md.append(f"- **Top Influential Peer Stocks:** {peer_list}")
            md.append("")

        md.append("## 5. Scientific Caveats & Disclaimers")
        for c in record['caveats']:
            md.append(f"> ⚠️ *{c}*")

        return "\n".join(md)

    def save_case_record(self, record: Dict[str, Any], output_path: str):
        """Serialize case record to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(output_path, "w") as f:
            json.dump(record, f, indent=2, default=convert_numpy)
        logging.info(f"Saved explanation case record to {output_path}")
