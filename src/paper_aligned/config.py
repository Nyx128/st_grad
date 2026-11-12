"""
Configuration for Track B: Paper-Aligned (paper_aligned)
Implements published equations, rigorous leakage prevention, and clean stress testing.
"""

import os
from dataclasses import dataclass, field
from typing import List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

@dataclass
class PaperAlignedConfig:
    # Reproducibility & Seeds
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    primary_seed: int = 43

    # Paths
    raw_data_dir: str = os.path.join(PROJECT_ROOT, "ST-GRAD-Stock-Prediction-6E7D", "data", "raw")
    stock_order_path: str = os.path.join(PROJECT_ROOT, "ST-GRAD-Stock-Prediction-6E7D", "data", "raw", "stock_order.npy")
    cluster_file_path: str = os.path.join(PROJECT_ROOT, "ST-GRAD-Stock-Prediction-6E7D", "data", "raw", "cluster_results.csv")
    artifacts_dir: str = os.path.join(PROJECT_ROOT, "artifacts", "paper_aligned")

    # Data Universe & Dimensions
    num_stocks: int = 50
    num_features: int = 24
    window_size: int = 30
    pred_horizon: int = 1

    # Feature Calculation Options
    vwap_mode: str = "table_literal"  # "table_literal" (Table 5) or "conventional"

    # Target & Volatility Threshold (Eqs. 28-30)
    base_threshold: float = 0.0003
    k_factor: float = 3.0
    volatility_mult: float = 1.0       # Strict Eq. (28): no * 8 factor
    min_volume_change: float = 0.003   # 0.3% minimum volume change threshold
    stochastic_dropout: float = 0.10   # 10% label smoothing dropout
    label_mode: str = "clean_eval"     # "clean_eval" (dropout on train only) or "paper_literal"

    # Dataset Partitioning (Chronological)
    train_samples: int = 856           # First 70%
    clean_val_samples: int = 307       # 307 non-stress out-of-sample days
    stress_samples: int = 61           # 61 stress days (Nov 2024 - Jan 2025)
    total_samples: int = 1224

    # Normalization Axis
    normalization_mode: str = "per_feature"  # [1, 24, 1, 1] fit strictly on train

    # Architecture Hyperparameters
    hidden_channels: int = 32
    num_layers: int = 2
    diffusion_steps: int = 2
    embed_dim: int = 10
    temporal_heads: int = 4
    dropout_input: float = 0.3
    dropout_head: float = 0.5

    # Training Protocol
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 50
    shuffle_train: bool = True
    loss_weighting: str = "unweighted"  # "unweighted" or "pos_weight"
