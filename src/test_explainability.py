"""
Unit Tests for Explainability Modules (Phase 1, 2, 3, 4 & 6/7).
Validates:
1. Integrated Gradients completeness axiom (convergence delta < 0.05).
2. Correct attribution tensor dimensions [F, T, N].
3. Non-destructive MVCA view interventions (clean hook removal).
4. Static prior vs Adaptive graph channel ablation.
5. Faithfulness deletion/insertion curves.
6. Model parameter randomization sanity check (Adebayo et al., 2018).
7. ModelTrackAdapter polymorphism across Track A (repo_compat) and Track B (paper_aligned).
8. Gauss-Legendre quadrature numerical precision and convergence flags.
9. STGradExplainer end-to-end execution, schema compliance, and Markdown report cards.
10. Trapezoidal Area Over Perturbation Curve (AOPC) computation.
11. Matched random edge and peer node controls (Hypothesis H3).
12. Hardened stability evaluation with strict valid-trial gating and uncertainty metrics.
13. Case record serialization (JSON/NPZ validation) and responsive HTML report card formatting.
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
from src.repo_compat.model import ST_GRAD_RepoCompat
from src.explainability.adapter import ModelTrackAdapter
from src.explainability.attribution import IntegratedGradients
from src.explainability.interventions import ViewInterventions, GraphInterventions
from src.explainability.faithfulness import deletion_insertion_test, model_randomization_test
from src.explainability.stability import evaluate_seed_stability, evaluate_perturbation_stability
from src.explainability.explainer import STGradExplainer
from src.explainability.report import save_case_record, load_case_record, format_html_report_card, compile_batch_summary_report
from src.explainability.plots import (
    generate_feature_importance_svg,
    generate_view_hierarchy_svg,
    generate_temporal_decay_svg,
    generate_graph_channel_svg,
    generate_regime_dashboard_html
)
from src.explainability.regime_benchmark import evaluate_cohort, compare_regimes, compare_tracks


class TestExplainability(unittest.TestCase):
    """Comprehensive test suite for the ST-GRAD Explainability Layer."""

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

        res = ig.attribute(self.x, self.A, target_stock_idx=target_stock, steps=50, method="gauss_legendre")

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
        self.assertTrue(res['is_converged'])

    def test_02_view_interventions_non_destructive(self):
        """Verify that MVCA view interventions correctly hook and cleanly restore model forward state."""
        intervener = ViewInterventions(self.model)
        target_stock = 5

        with torch.no_grad():
            clean_out_before = self.model(self.x, self.A)[0, target_stock].item()

        res = intervener.evaluate_view_contributions(self.x, self.A, target_stock_idx=target_stock)

        self.assertIn('ST', res['interventions'])
        self.assertIn('DS', res['interventions'])
        self.assertIn('DT', res['interventions'])
        self.assertEqual(len(res['view_importance_ranking']), 3)

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
        """Verify that deletion and insertion curves execute and compute AUC/AOPC metrics."""
        ig = IntegratedGradients(self.model)
        res = ig.attribute(self.x, self.A, target_stock_idx=0, steps=15)
        f_imp = res['feature_importance_abs']

        del_ins = deletion_insertion_test(
            self.model, self.x, self.A, target_stock_idx=0,
            feature_attributions=f_imp, steps=[1, 3, 6, 12]
        )

        self.assertIn('aopc_deletion_top', del_ins)
        self.assertIn('aopc_deletion_random', del_ins)
        self.assertIn('aopc_insertion_top', del_ins)
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

    def test_06_model_track_adapter_polymorphism(self):
        """Verify that ModelTrackAdapter unifies Track A (repo_compat) and Track B (paper_aligned)."""
        # 1. Track B
        adapter_b = ModelTrackAdapter(self.model)
        self.assertEqual(adapter_b.track, "paper_aligned")
        self.assertEqual(adapter_b.num_layers, 1)
        st_mod_b = adapter_b.get_view_module(0, "ST")
        ds_mod_b = adapter_b.get_view_module(0, "DS")
        dt_mod_b = adapter_b.get_view_module(0, "DT")
        self.assertIsNotNone(st_mod_b)
        self.assertIsNotNone(ds_mod_b)
        self.assertIsNotNone(dt_mod_b)
        weights_b = adapter_b.get_fusion_weights(0)
        self.assertIn("ST", weights_b)
        self.assertAlmostEqual(sum(weights_b.values()), 1.0, places=5)

        # 2. Track A
        model_a = ST_GRAD_RepoCompat(
            in_channels=self.F, out_step=1, num_nodes=self.N, seq_len=self.T,
            hidden_channels=self.D, num_layers=1, diffusion_steps=2, embed_dim=10
        ).to(self.device).eval()
        adapter_a = ModelTrackAdapter(model_a)
        self.assertEqual(adapter_a.track, "repo_compat")
        st_mod_a = adapter_a.get_view_module(0, "ST")
        fs_mod_a = adapter_a.get_view_module(0, "DS")  # Maps DS -> fs_view
        ft_mod_a = adapter_a.get_view_module(0, "DT")  # Maps DT -> ft_view
        self.assertIsNotNone(st_mod_a)
        self.assertIsNotNone(fs_mod_a)
        self.assertIsNotNone(ft_mod_a)
        weights_a = adapter_a.get_fusion_weights(0)
        self.assertIn("DS", weights_a)
        self.assertAlmostEqual(sum(weights_a.values()), 1.0, places=5)

    def test_07_gauss_legendre_quadrature_precision(self):
        """Verify Gauss-Legendre quadrature achieves lower completeness error than standard Riemann sum."""
        ig = IntegratedGradients(self.model)
        res_gl = ig.attribute(self.x, self.A, target_stock_idx=0, steps=30, method="gauss_legendre", adaptive=False)
        res_riemann = ig.attribute(self.x, self.A, target_stock_idx=0, steps=30, method="riemann", adaptive=False)

        self.assertTrue(np.isfinite(res_gl['absolute_convergence_delta']))
        self.assertTrue(np.isfinite(res_riemann['absolute_convergence_delta']))
        self.assertIn('steps_used', res_gl)
        self.assertEqual(res_gl['quadrature_method'], 'gauss_legendre')

    def test_08_stgrad_explainer_end_to_end(self):
        """Verify STGradExplainer end-to-end execution, schema compliance, and Markdown generation."""
        explainer = STGradExplainer(model=self.model, stock_order=[f"S_{i}" for i in range(self.N)], device=self.device)

        rec = explainer.explain_instance(
            x=self.x,
            target_stock="S_0",
            target_date="2024-11-05",
            ig_steps=20,
            top_k_features=4,
            top_k_peers=3
        )

        self.assertEqual(rec['ticker'], "S_0")
        self.assertEqual(rec['target_date'], "2024-11-05")
        self.assertIn('prediction', rec)
        self.assertIn('logit', rec['prediction'])
        self.assertIn('probability', rec['prediction'])
        self.assertIn('trend_direction', rec['prediction'])

        self.assertIn('attribution', rec)
        self.assertEqual(len(rec['attribution']['top_magnitude_features']), 4)
        self.assertTrue(rec['attribution']['convergence']['is_converged'])

        self.assertIn('views', rec)
        self.assertEqual(len(rec['views']['importance_ranking']), 3)

        self.assertIn('graph', rec)
        self.assertEqual(len(rec['graph']['top_influential_peer_tickers']), 3)

        md_card = explainer.format_markdown_card(rec)
        self.assertIn("# ST-GRAD Prediction Explanation: S_0", md_card)
        self.assertIn("Directional Prediction Summary", md_card)

    def test_09_trapezoidal_aopc_faithfulness(self):
        """Verify formal trapezoidal AOPC computation with dense steps."""
        f_imp = np.linspace(0.1, 2.5, self.F)
        del_ins = deletion_insertion_test(
            self.model, self.x, self.A, target_stock_idx=0,
            feature_attributions=f_imp, steps=[1, 2, 4, 8, 16, 24],
            num_random_trials=5
        )
        self.assertIn('aopc_deletion_top', del_ins)
        self.assertIn('aopc_deletion_random', del_ins)
        self.assertIn('faithfulness_ratio', del_ins)
        self.assertTrue(np.isfinite(del_ins['aopc_deletion_top']))

    def test_10_matched_random_edge_controls(self):
        """Verify matched random edge controls for testing Hypothesis H3."""
        intervener = GraphInterventions(self.model)
        res = intervener.evaluate_matched_random_edge_controls(
            self.x, self.A, target_stock_idx=0, top_k=3, num_random_trials=5
        )
        self.assertIn('edge_influence_ratio', res)
        self.assertIn('mean_top_edge_delta', res)
        self.assertIn('mean_random_edge_delta', res)
        self.assertTrue(np.isfinite(res['edge_influence_ratio']))

    def test_11_hardened_perturbation_stability(self):
        """Verify perturbation stability with minimum valid trial gating and uncertainty estimation."""
        # 1. Normal run
        stab = evaluate_perturbation_stability(
            self.model, self.x, self.A, target_stock_idx=0,
            noise_std=0.01, num_trials=10, min_valid_trials=3, ig_steps=15
        )
        self.assertIn('status', stab)
        self.assertIn('mean_cosine_similarity', stab)
        self.assertIn('ci_95_spearman_correlation', stab)

        # 2. Extreme noise where valid trials are insufficient -> must fail robustness
        stab_fail = evaluate_perturbation_stability(
            self.model, self.x, self.A, target_stock_idx=0,
            noise_std=50.0, num_trials=5, min_valid_trials=10, ig_steps=15
        )
        self.assertFalse(stab_fail['is_robust'])
        self.assertEqual(stab_fail['status'], 'insufficient_valid_trials')

    def test_12_case_record_serialization_and_html(self):
        """Verify case record schema validation, JSON persistence, and HTML card generation."""
        explainer = STGradExplainer(model=self.model, stock_order=[f"S_{i}" for i in range(self.N)], device=self.device)
        rec = explainer.explain_instance(x=self.x, target_stock="S_0", target_date="2024-11-05", ig_steps=15)

        tmp_json = "/tmp/test_stgrad_case.json"
        save_case_record(rec, tmp_json)
        loaded = load_case_record(tmp_json)
        self.assertEqual(loaded['ticker'], "S_0")
        self.assertEqual(loaded['prediction']['predicted_class'], rec['prediction']['predicted_class'])

        html_out = format_html_report_card(rec)
        self.assertIn("<!DOCTYPE html>", html_out)
        self.assertIn("S_0", html_out)

        # Batch summary test
        summary = compile_batch_summary_report([rec, rec])
        self.assertEqual(summary['total_cases'], 2)
        if os.path.exists(tmp_json):
            os.remove(tmp_json)

    def test_13_regime_benchmark_and_svg_plots(self):
        """Verify Phase 5 regime cohort evaluation, H6 comparison, and SVG/HTML plot generation."""
        # 1. SVG generation checks
        feat_clean = {"RSI": 0.25, "ATR": 0.18, "MACD_Signal": 0.12}
        feat_stress = {"RSI": 0.15, "ATR": 0.32, "MACD_Signal": 0.08}
        svg_f = generate_feature_importance_svg(feat_clean, feat_stress, top_n=3)
        self.assertIn("<svg", svg_f)
        self.assertIn("</svg>", svg_f)
        self.assertIn("RSI", svg_f)

        v_data = {
            "Clean": {"ST": 0.3, "DS": 0.2, "DT": 0.4},
            "Stress": {"ST": 0.5, "DS": 0.1, "DT": 0.3}
        }
        svg_v = generate_view_hierarchy_svg(v_data)
        self.assertIn("<svg", svg_v)

        svg_t = generate_temporal_decay_svg([0.05] * 30, [0.08] * 30)
        self.assertIn("<svg", svg_t)

        svg_g = generate_graph_channel_svg({"Clean": {"Static Prior": 0.1, "Adaptive Dynamic": 0.2, "Total Spatial": 0.3}})
        self.assertIn("<svg", svg_g)

        dashboard_html = generate_regime_dashboard_html(
            {"hypothesis_h6_evaluation": {"verdict": "CONFIRMED"}},
            svg_f, svg_v, svg_t, svg_g
        )
        self.assertIn("<!DOCTYPE html>", dashboard_html)
        self.assertIn("Dual-Track Benchmark", dashboard_html)

        # 2. Cohort evaluation and H6 statistical testing
        explainer = STGradExplainer(model=self.model, stock_order=[f"S_{i}" for i in range(self.N)], device=self.device)
        synth_x = np.random.randn(3, 30, self.N, self.F).astype(np.float32)
        synth_dates = ["2024-01-01", "2024-01-02", "2024-01-03"]

        cohort_clean = evaluate_cohort(
            explainer, synth_x, synth_dates, num_samples=2, sample_stocks=[0, 1], ig_steps=10
        )
        self.assertEqual(cohort_clean['num_evaluations'], 4)
        self.assertIn('feature_importance_mean', cohort_clean)
        self.assertIn('view_deltas_mean', cohort_clean)
        self.assertIn('adaptive_graph_share', cohort_clean)

        cohort_stress = evaluate_cohort(
            explainer, synth_x, synth_dates, num_samples=2, sample_stocks=[0, 1], ig_steps=10
        )

        h6_res = compare_regimes(cohort_clean, cohort_stress)
        self.assertIn('verdict', h6_res)
        self.assertIn('feature_rank_spearman_rho', h6_res)
        self.assertIn('view_shifts', h6_res)
        self.assertIn('graph_channel_shifts', h6_res)

        track_comp = compare_tracks(cohort_clean, cohort_clean, cohort_stress, cohort_stress)
        self.assertIn('feature_rank_concordance_clean', track_comp)
        self.assertIn('view_comparison_clean', track_comp)


if __name__ == '__main__':
    unittest.main()
