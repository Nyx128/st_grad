"""
Test suite for Track A: repo_compat.
Validates exact parameter count (216,429), state_dict compatibility with author weights,
and reproduction of Table 8 validation metrics.
"""

import os
import sys
import unittest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.repo_compat.config import RepoCompatConfig
from src.repo_compat.model import ST_GRAD_RepoCompat
from src.repo_compat.evaluate import evaluate_repo_compat_checkpoint


class TestRepoCompat(unittest.TestCase):
    """Test suite for Track A repo_compat."""

    def setUp(self):
        self.config = RepoCompatConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_parameter_count(self):
        model = ST_GRAD_RepoCompat(
            in_channels=self.config.num_features,
            out_step=1,
            num_nodes=self.config.num_stocks,
            seq_len=self.config.window_size,
            hidden_channels=self.config.hidden_channels,
            num_layers=self.config.num_layers,
            diffusion_steps=self.config.diffusion_steps,
            embed_dim=self.config.embed_dim
        )
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(total_params, 216429, f"Expected 216,429 params, got {total_params}")

    def test_checkpoint_loading_best_model(self):
        model = ST_GRAD_RepoCompat(
            in_channels=self.config.num_features,
            out_step=1,
            num_nodes=self.config.num_stocks,
            seq_len=self.config.window_size,
            hidden_channels=self.config.hidden_channels,
            num_layers=self.config.num_layers,
            diffusion_steps=self.config.diffusion_steps,
            embed_dim=self.config.embed_dim
        )
        state_dict = torch.load(self.config.best_model_path, map_location="cpu")
        load_res = model.load_state_dict(state_dict)
        self.assertEqual(len(load_res.missing_keys), 0, f"Missing keys: {load_res.missing_keys}")
        self.assertEqual(len(load_res.unexpected_keys), 0, f"Unexpected keys: {load_res.unexpected_keys}")

    def test_reproduction_metrics_best_model(self):
        metrics = evaluate_repo_compat_checkpoint(
            checkpoint_path=self.config.best_model_path,
            config=self.config,
            device=self.device
        )
        # Expected from audit: Acc ~ 0.8317, Pre ~ 0.7651, Rec ~ 0.8911, F1 ~ 0.8233, AUC ~ 0.9041
        self.assertAlmostEqual(metrics['accuracy'], 0.8317, delta=0.005)
        self.assertAlmostEqual(metrics['f1'], 0.8233, delta=0.005)
        self.assertAlmostEqual(metrics['auc'], 0.9041, delta=0.005)


if __name__ == '__main__':
    unittest.main()
