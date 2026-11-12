"""
Unit tests and mathematical verification for Phase 1 Paper Formalization.
Tests Table 5 technical indicator formulas, Eq. (28)-(30) label generation,
validity masking, and tensor dimension operations.
"""

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TestFeatureFormulas(unittest.TestCase):
    """Verifies Table 5 technical indicator formulas against hand-calculated values."""

    def test_relative_close(self):
        # f_rel_c = Close / MA30(Close)
        close = np.ones(30) * 10.0
        close[-1] = 12.0  # last value 12, mean is (29*10 + 12)/30 = 302/30 = 10.0667
        ma30 = np.mean(close)
        rel_c = close[-1] / ma30
        expected = 12.0 / (302.0 / 30.0)
        self.assertAlmostEqual(rel_c, expected, places=5)

    def test_open_close_ratio(self):
        # f_ocr = (Close - Open) / Open
        open_val, close_val = 100.0, 105.0
        ocr = (close_val - open_val) / open_val
        self.assertAlmostEqual(ocr, 0.05, places=5)

    def test_bollinger_position(self):
        # f_bb_p = (Close - (MA20 - 2*sigma20)) / (4*sigma20)
        # When Close == MA20, position should be exactly 0.5
        ma = 50.0
        std = 2.5
        bb_low = ma - 2 * std  # 45.0
        bb_width = 4 * std     # 10.0
        close = 50.0
        pos = (close - bb_low) / bb_width
        self.assertAlmostEqual(pos, 0.5, places=5)

    def test_roc(self):
        # f_roc = Close_t / Close_{t-12} - 1
        c_t = 120.0
        c_t_12 = 100.0
        roc = (c_t / c_t_12) - 1.0
        self.assertAlmostEqual(roc, 0.20, places=5)


class TestTargetGeneration(unittest.TestCase):
    """Verifies Eq. (28)-(30) target return, threshold, and validity masking."""

    def test_volatility_adaptive_threshold_literal(self):
        # Eq. (28): T = T_base * (1 + k * tanh(|vol|)), T_base=0.0003, k=3
        t_base = 0.0003
        k = 3.0
        vol_pct = 0.05
        # Literal paper
        threshold_paper = t_base * (1.0 + k * np.tanh(np.abs(vol_pct)))
        # Repo implementation (with * 8 factor)
        threshold_repo = t_base * (1.0 + k * np.tanh(np.abs(vol_pct) * 8.0))
        self.assertGreater(threshold_repo, threshold_paper)
        expected_paper = 0.0003 * (1.0 + 3.0 * np.tanh(0.05))
        self.assertAlmostEqual(threshold_paper, expected_paper, places=7)

    def test_benchmark_relative_return(self):
        # Eq. (29): R_{t+1} = (P_{t+1} - P_bar_t) / P_bar_t
        p_next = 105.0
        p_bar = 100.0
        ret = (p_next - p_bar) / p_bar
        self.assertAlmostEqual(ret, 0.05, places=5)

    def test_inactive_sample_filtering(self):
        # Filter samples with |vol| < 0.003 (0.3%)
        vol_pct_active = 0.005
        vol_pct_inactive = 0.002
        self.assertTrue(abs(vol_pct_active) >= 0.003)
        self.assertFalse(abs(vol_pct_inactive) >= 0.003)


class TestModelTensorShapes(unittest.TestCase):
    """Verifies module shapes matching the formal specification."""

    def test_temporal_attention_shape(self):
        B, D, T, N = 2, 32, 30, 50
        x = torch.randn(B, D, T, N)
        # Reshape to (B*N, T, D)
        x_reshaped = x.permute(0, 3, 2, 1).reshape(B * N, T, D)
        self.assertEqual(x_reshaped.shape, (100, 30, 32))

        mha = nn.MultiheadAttention(embed_dim=D, num_heads=4, batch_first=True)
        attn_out, _ = mha(x_reshaped, x_reshaped, x_reshaped)
        self.assertEqual(attn_out.shape, (100, 30, 32))

        # Restore to (B, D, T, N)
        x_restored = attn_out.reshape(B, N, T, D).permute(0, 3, 2, 1)
        self.assertEqual(x_restored.shape, (B, D, T, N))

    def test_adaptive_adjacency_shape_and_stochasticity(self):
        N, d = 50, 10
        E1 = nn.Parameter(torch.randn(N, d))
        E2 = nn.Parameter(torch.randn(N, d))
        adj_logits = F.relu(torch.mm(E1, E2.t())) + 1e-6
        A_adp = F.softmax(adj_logits, dim=1)
        self.assertEqual(A_adp.shape, (50, 50))
        # Check row sums are 1.0 (row-stochastic)
        row_sums = A_adp.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(50), rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
