"""
Configuration for Track A: Repository Compatibility (repo_compat)
Captures the exact hyperparameters, paths, and settings of the bundled repository.
"""

import os
from dataclasses import dataclass

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ST-GRAD-Stock-Prediction-6E7D"))

@dataclass
class RepoCompatConfig:
    # Environment & Reproducibility
    seed: int = 43

    # Paths
    repo_root: str = REPO_ROOT
    data_dir: str = os.path.join(REPO_ROOT, "data", "raw")
    cluster_file: str = os.path.join(REPO_ROOT, "data", "raw", "cluster_results.csv")
    order_path: str = os.path.join(REPO_ROOT, "data", "raw", "stock_order.npy")
    adj_matrix_path: str = os.path.join(REPO_ROOT, "data", "processed", "normalized_adj_matrix.npy")
    dataset_path: str = os.path.join(REPO_ROOT, "data", "processed", "stock_dataset.npz")
    save_dir: str = os.path.join(REPO_ROOT, "models", "saved")
    best_model_path: str = os.path.join(REPO_ROOT, "models", "saved", "best_model_ours.pth")
    default_model_path: str = os.path.join(REPO_ROOT, "models", "saved", "st_grad_best_model.pth")

    # Data & Window Hyperparameters
    window_size: int = 30
    pred_horizon: int = 1
    num_stocks: int = 50
    num_features: int = 24
    base_threshold: float = 0.0003
    min_volume_change: float = 0.003
    k_factor: float = 3.0
    volatility_mult: float = 8.0   # Discrepancy: extra * 8 factor in tanh
    stochastic_dropout: float = 0.10

    # Model Hyperparameters
    hidden_channels: int = 32
    num_layers: int = 2
    diffusion_steps: int = 2
    embed_dim: int = 10
    temporal_heads: int = 4
    dropout_input: float = 0.3
    dropout_head: float = 0.5

    # Training Hyperparameters
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 50
    shuffle: bool = False          # Discrepancy: DataLoader shuffle=False
    use_pos_weight: bool = True    # Discrepancy: BCEWithLogitsLoss with pos_weight
