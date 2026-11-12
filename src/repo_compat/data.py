"""
Data loading and preprocessing for Track A: repo_compat.
Includes non-destructive compatibility shim for Pandas 3.0+.
"""

import os
import warnings
from typing import Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Apply non-destructive Pandas 3.0 compatibility shim
import pandas as pd
_orig_df_fillna = pd.DataFrame.fillna
_orig_s_fillna = pd.Series.fillna

def _compat_df_fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
    if method == 'ffill':
        return self.ffill(axis=axis, inplace=inplace, limit=limit)
    elif method == 'bfill':
        return self.bfill(axis=axis, inplace=inplace, limit=limit)
    return _orig_df_fillna(self, value=value, axis=axis, inplace=inplace, limit=limit)

def _compat_s_fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None):
    if method == 'ffill':
        return self.ffill(axis=axis, inplace=inplace, limit=limit)
    elif method == 'bfill':
        return self.bfill(axis=axis, inplace=inplace, limit=limit)
    return _orig_s_fillna(self, value=value, axis=axis, inplace=inplace, limit=limit)

pd.DataFrame.fillna = _compat_df_fillna
pd.Series.fillna = _compat_s_fillna

from src.repo_compat.config import RepoCompatConfig


def load_repo_compat_data(
    config: RepoCompatConfig,
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load preprocessed dataset and normalized adjacency matrix for Track A.
    Returns:
        X_train: [856, 24, 30, 50]
        X_val:   [368, 24, 30, 50]
        y_train: [856, 50]
        y_val:   [368, 50]
        A:       [50, 50]
    """
    if not os.path.exists(config.dataset_path) or not os.path.exists(config.adj_matrix_path):
        raise FileNotFoundError(
            f"Processed data not found at {config.dataset_path} or {config.adj_matrix_path}. "
            "Please run preprocessing first."
        )

    data = np.load(config.dataset_path, allow_pickle=True)
    # Original data is [samples, T, N, F] -> permuted to [samples, F, T, N]
    X_train = torch.FloatTensor(data['X_train']).permute(0, 3, 1, 2)
    y_train = torch.FloatTensor(data['y_train']).squeeze(-1)
    X_val = torch.FloatTensor(data['X_val']).permute(0, 3, 1, 2)
    y_val = torch.FloatTensor(data['y_val']).squeeze(-1)

    A = torch.FloatTensor(np.load(config.adj_matrix_path)).to(device)
    return X_train, X_val, y_train, y_val, A


def get_repo_compat_dataloaders(
    config: RepoCompatConfig,
    device: torch.device = torch.device("cpu")
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Build DataLoaders for Track A.
    Note: Preserves shuffle=False from the original repository.
    """
    X_train, X_val, y_train, y_val, A = load_repo_compat_data(config, device=device)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=config.batch_size,
        shuffle=config.shuffle
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=config.batch_size,
        shuffle=False
    )
    return train_loader, val_loader, A
