"""
Unit Tests for Explainability Modules (Phase 6).
Validates:
1. Integrated Gradients completeness axiom (convergence delta < 0.05).
2. Correct attribution tensor dimensions [F, T, N].
3. Non-destructive MVCA view interventions (clean hook removal).
4. Static prior vs Adaptive graph channel ablation.
5. Faithfulness deletion/insertion curves.
6. Model parameter randomization sanity check (Adebayo et al., 2018).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest
import numpy as np
import torch
import torch.nn as nn

from src.paper_aligned.config import PaperAlignedConfig
from src.paper_aligned.model import ST_GRAD_PaperAligned
from src.explainability.attribution import IntegratedGradients
from src.explainability.interventions import ViewInterventions, GraphInterventions
from src.explainability.faithfulness import deletion_insertion_test, model_randomization_test


class TestExplainability(unittest.TestCase):
    """Unit test suite for Phase 6 explainability methods."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.B, cls.F, cls.T, cls.N, cls.D = 1, 24, 30, 50, 32

        # Instantiate lightweight paper-aligned model
        cls.model = ST_GRAD_PaperAligned(
            in_channels=cls.F,
            num_nodes=cls.N,
            seq_len=cls.T,
            hidden_channels=cls.D,
            num_layers=1,
            diffusion_steps=2,
            embed_dim=10
        ).to(cls.device)
        cls.model.eval()

        # Synthetic inputs
        torch.manual_seed(42)
        cls.x = torch.randn(cls.B, cls.F, cls.T, cls.N, device=cls.device)
        cls.A = torch.eye(cls.N, device=cls.device)

    def test_01_integrated_gradients_completeness(self):
        """Verify that Integrated Gradients satisfies the completeness axiom within numerical tolerance."""
        ig = IntegratedGradients(self.model)
        target_stock = 0

        res = ig.attribute(self.x, self.A, target_stock_idx=target_stock, steps=50)

        # 1. Output shapes
        self.assertEqual(res['attributions'].shape, (self.F, self.T, self.N))
        self.assertEqual(len(res['feature_importance_abs']), self.F)
        self.assertEqual(len(res['temporal_importance_abs']), self.T)
        self.assertEqual(len(res['spatial_importance_abs']), self.N)

        # 2. Completeness: sum(IG) ≈ F(x) - F(x0)
        abs_delta = res['absolute_convergence_delta']
        self.assertLess(
            abs_delta, 0.05,
            f"Absolute completeness error {abs_delta:.4f} exceeds 0.05 logit tolerance"
        )

    def test_02_view_interventions_non_destructive(self):
        """Verify that MVCA view interventions correctly hook and cleanly restore model forward state."""
        intervener = ViewInterventions(self.model)
        target_stock = 5

        # Check clean output before intervention
        with torch.no_grad():
            clean_out_before = self.model(self.x, self.A)[0, target_stock].item()

        res = intervener.evaluate_view_contributions(self.x, self.A, target_stock_idx=target_stock)

        self.assertIn('ST', res['interventions'])
        self.assertIn('DS', res['interventions'])
        self.assertIn('DT', res['interventions'])
        self.assertEqual(len(res['view_importance_ranking']), 3)

        # Check clean output after intervention (must be identical, hooks removed)
        with torch.no_grad():
            clean_out_after = self.model(self.x, self.A)[0, target_stock].item()

        self.assertAlmostEqual(
            clean_out_before, clean_out_after, places=6,
            msg="Model forward pass was permanently corrupted by intervention hooks!"
        )

    def test_03_graph_interventions_channels(self):
        """Verify that static prior and adaptive graph channels can be ablated independently."""
        intervener = GraphInterventions(self.model)
        target_stock = 2

        res = intervener.evaluate_graph_channels(self.x, self.A, target_stock_idx=target_stock)

        self.assertIn('static_prior_ablation', res)
        self.assertIn('adaptive_graph_ablation', res)
        self.assertIn('total_spatial_ablation', res)

        self.assertTrue(np.isfinite(res['static_prior_ablation']['delta_logit']))
        self.assertTrue(np.isfinite(res['adaptive_graph_ablation']['delta_logit']))

    def test_04_faithfulness_deletion_insertion(self):
        """Verify that deletion and insertion curves execute and compute AUC ratios."""
        ig = IntegratedGradients(self.model)
        res = ig.attribute(self.x, self.A, target_stock_idx=0, steps=15)
        f_imp = res['feature_importance_abs']

        del_ins = deletion_insertion_test(
            self.model, self.x, self.A, target_stock_idx=0,
            feature_attributions=f_imp, steps=[1, 3, 6, 12]
        )

        self.assertIn('auc_deletion_top', del_ins)
        self.assertIn('auc_deletion_random', del_ins)
        self.assertIn('auc_insertion_top', del_ins)
        self.assertGreater(len(del_ins['deletion_top']), 0)

    def test_05_parameter_randomization_sanity_check(self):
        """Verify that cascading parameter randomization check executes and returns valid metrics."""
        ig = IntegratedGradients(self.model)
        res = ig.attribute(self.x, self.A, target_stock_idx=0, steps=15)
        f_imp = res['feature_importance_abs']

        rand_res = model_randomization_test(
            self.model, self.x, self.A, target_stock_idx=0,
            original_attributions=f_imp, seed=42
        )

        self.assertIn('spearman_correlation', rand_res)
        self.assertIn('cosine_similarity', rand_res)
        self.assertTrue(-1.0 <= rand_res['spearman_correlation'] <= 1.0)


if __name__ == '__main__':
    unittest.main()
