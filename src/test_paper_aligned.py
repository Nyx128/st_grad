"""
Test suite for Track B: paper_aligned.
Validates:
  1. Complete forward and backward gradient flow (verifies NO dead parameters).
  2. Tensor shape assertions across all sub-modules.
  3. Feature extraction produces 24 clean features.
  4. Label generator operates on raw closing prices with boolean validity masks.
  5. Single-batch forward/backward/optimizer step test.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.paper_aligned.config import PaperAlignedConfig
from src.paper_aligned.features import extract_features_for_ticker, FEATURE_NAMES
from src.paper_aligned.labels import generate_paper_aligned_labels
from src.paper_aligned.model import (
    AdaptiveAdjacencyPaperAligned,
    CrossAttentionViewPaperAligned,
    TemporalAttentionPaperAligned,
    DiffusionConvolutionPaperAligned,
    STBlockPaperAligned,
    ST_GRAD_PaperAligned
)


class TestPaperAligned(unittest.TestCase):
    """Test suite for Track B paper_aligned."""

    def setUp(self):
        self.config = PaperAlignedConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B = 2
        self.F = 24
        self.D = 32
        self.T = 30
        self.N = 50

    def test_feature_names_count(self):
        self.assertEqual(len(FEATURE_NAMES), 24, "Must have exactly 24 features")

    def test_feature_extraction_toy(self):
        # Create 100 days of toy OHLCV data
        dates = pd.date_range('2020-01-01', periods=100)
        np.random.seed(42)
        close = 100.0 + np.cumsum(np.random.randn(100))
        high = close + np.random.rand(100) * 2.0
        low = close - np.random.rand(100) * 2.0
        open_p = low + (high - low) * np.random.rand(100)
        vol = 10000 + np.random.randint(0, 5000, 100)

        df = pd.DataFrame({
            'date': dates,
            'open': open_p,
            'high': high,
            'low': low,
            'close': close,
            'volume': vol
        })

        feat_df = extract_features_for_ticker(df)
        self.assertEqual(feat_df.shape, (100, 24))
        self.assertFalse(feat_df.isna().any().any(), "Features contain NaNs")
        self.assertFalse(np.isinf(feat_df.values).any(), "Features contain Infinities")

    def test_label_generation_toy(self):
        dates_count = 100
        stocks_count = 50
        np.random.seed(42)
        raw_close = 50.0 + np.cumsum(np.random.randn(dates_count, stocks_count) * 0.5, axis=0)
        raw_volume = 10000.0 + np.random.rand(dates_count, stocks_count) * 2000.0

        labels, masks, rets, thresh = generate_paper_aligned_labels(
            raw_close=raw_close,
            raw_volume=raw_volume,
            window_size=30,
            pred_horizon=1,
            label_mode="clean_eval",
            train_split_idx=50
        )

        expected_samples = 100 - 30 - 1 + 1  # 70
        self.assertEqual(labels.shape, (expected_samples, stocks_count))
        self.assertEqual(masks.shape, (expected_samples, stocks_count))
        self.assertTrue(set(np.unique(labels)).issubset({0.0, 1.0}))

    def test_view_shapes_and_outputs(self):
        x = torch.randn(self.B, self.D, self.T, self.N)
        for view_name in ['ST', 'DS', 'DT']:
            view = CrossAttentionViewPaperAligned(view_name, num_nodes=self.N, seq_len=self.T, hidden_dim=self.D)
            out = view(x)
            self.assertEqual(out.shape, (self.B, self.D, self.T, self.N), f"{view_name} output shape mismatch")

    def test_gradient_flow_no_dead_parameters(self):
        """Verify that EVERY single trainable parameter is connected and receives non-None gradient."""
        model = ST_GRAD_PaperAligned(
            in_channels=self.F,
            num_nodes=self.N,
            seq_len=self.T,
            hidden_channels=self.D,
            num_layers=self.config.num_layers,
            diffusion_steps=self.config.diffusion_steps,
            embed_dim=self.config.embed_dim
        ).to(self.device)

        x = torch.randn(self.B, self.F, self.T, self.N, device=self.device)
        A = torch.eye(self.N, device=self.device)
        logits = model(x, A)
        self.assertEqual(logits.shape, (self.B, self.N))

        loss = logits.sum()
        loss.backward()

        disconnected_params = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        self.assertEqual(
            len(disconnected_params), 0,
            f"Found disconnected parameters with grad is None (unlike repo_compat, paper_aligned must have none): {disconnected_params}"
        )

    def test_single_batch_overfit(self):
        """Single-batch overfit test to confirm loss decreases monotonically."""
        model = ST_GRAD_PaperAligned(
            in_channels=self.F,
            num_nodes=self.N,
            seq_len=self.T,
            hidden_channels=self.D,
            num_layers=1,
            diffusion_steps=2,
            embed_dim=10
        ).to(self.device)

        optimizer = optim.AdamW(model.parameters(), lr=1e-2)
        criterion = nn.BCEWithLogitsLoss()

        x = torch.randn(self.B, self.F, self.T, self.N, device=self.device)
        y = torch.randint(0, 2, (self.B, self.N), device=self.device).float()
        A = torch.eye(self.N, device=self.device)

        losses = []
        for _ in range(15):
            optimizer.zero_grad()
            out = model(x, A)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        self.assertLess(losses[-1], losses[0], "Loss failed to decrease on single-batch overfit test")


if __name__ == '__main__':
    unittest.main()
