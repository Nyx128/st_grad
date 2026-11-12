"""
Evaluation routine for Track A: repo_compat.
Loads weights into ST_GRAD_RepoCompat and evaluates on the 368-sample validation set.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.repo_compat.config import RepoCompatConfig
from src.repo_compat.model import ST_GRAD_RepoCompat
from src.repo_compat.data import get_repo_compat_dataloaders
from src.common.metrics import compute_metrics


def evaluate_repo_compat_checkpoint(
    checkpoint_path: str,
    config: Optional[RepoCompatConfig] = None,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluate a checkpoint on the repository validation partition.
    """
    if config is None:
        config = RepoCompatConfig()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    _, val_loader, A = get_repo_compat_dataloaders(config, device=device)

    model = ST_GRAD_RepoCompat(
        in_channels=config.num_features,
        out_step=1,
        num_nodes=config.num_stocks,
        seq_len=config.window_size,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        diffusion_steps=config.diffusion_steps,
        embed_dim=config.embed_dim
    ).to(device)

    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch, A)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_probs = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    metrics = compute_metrics(y_true=all_labels, y_pred=all_preds, y_prob=all_probs)
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Track A Checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = RepoCompatConfig()
    ckpt = args.checkpoint or cfg.best_model_path
    print(f"Evaluating {ckpt}...")
    results = evaluate_repo_compat_checkpoint(ckpt, cfg)
    print("\n--- Track A Evaluation Results ---")
    for k, v in results.items():
        print(f"  {k:10}: {v:.4f}")
