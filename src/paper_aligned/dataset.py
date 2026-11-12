"""
Dataset Pipeline for Track B: paper_aligned.
Handles calendar alignment, feature extraction, label generation,
chronological partitioning (856 train / 307 clean val / 61 stress),
and leakage-safe per-feature normalization.
"""

import os
import json
import logging
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.paper_aligned.config import PaperAlignedConfig
from src.paper_aligned.features import extract_features_for_ticker, FEATURE_NAMES
from src.paper_aligned.labels import generate_paper_aligned_labels

SECTOR_FOLDERS = {
    'energy': 'Energy',
    'healthcare': 'Healthcare',
    'technology': 'Technology',
    'materials': 'BasicMaterials',
    'industrial': 'Industrials'
}


class STGRADDataset(Dataset):
    """PyTorch Dataset returning features, labels, and validity masks."""
    def __init__(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray):
        # Permute from [samples, T, N, F] to [samples, F, T, N]
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2)
        self.y = torch.FloatTensor(y)
        self.mask = torch.BoolTensor(mask)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx], self.mask[idx]


def build_paper_aligned_dataset(config: PaperAlignedConfig) -> Dict[str, Any]:
    """
    Build the complete paper-aligned dataset pipeline end-to-end.
    """
    os.makedirs(config.artifacts_dir, exist_ok=True)
    stock_order = np.load(config.stock_order_path, allow_pickle=True)

    # 1. Map tickers to file paths
    file_dict = {}
    for folder in os.listdir(config.raw_data_dir):
        folder_path = os.path.join(config.raw_data_dir, folder)
        if os.path.isdir(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith('.csv'):
                    ticker = f.replace('.csv', '')
                    file_dict[ticker] = os.path.join(folder_path, f)

    # 2. Load raw ticker dataframes
    dfs = {code: pd.read_csv(file_dict[code]) for code in stock_order}

    # 3. Find common calendar dates across all stocks
    common_dates = dfs[stock_order[0]]['date']
    for code in stock_order[1:]:
        common_dates = pd.merge(
            pd.DataFrame({'date': common_dates}),
            pd.DataFrame({'date': dfs[code]['date']}),
            on='date'
        )['date']
    common_dates = common_dates.sort_values().values
    num_dates = len(common_dates)
    logging.info(f"Aligned {num_dates} common trading dates across {len(stock_order)} tickers.")

    # 4. Extract raw prices & features for all tickers aligned to dates
    raw_close = np.zeros((num_dates, len(stock_order)), dtype=np.float32)
    raw_volume = np.zeros((num_dates, len(stock_order)), dtype=np.float32)
    feature_panel = np.zeros((num_dates, len(stock_order), config.num_features), dtype=np.float32)

    for idx, code in enumerate(stock_order):
        df_code = dfs[code].set_index('date').reindex(common_dates).ffill().bfill().reset_index()
        raw_close[:, idx] = df_code['close'].values
        raw_volume[:, idx] = df_code['volume'].values

        feat_df = extract_features_for_ticker(df_code, vwap_mode=config.vwap_mode)
        feature_panel[:, idx, :] = feat_df.values

    # 5. Build sliding input windows: X [samples, T=30, N=50, F=24]
    num_samples = num_dates - config.window_size - config.pred_horizon + 1
    X = np.stack([feature_panel[i:i + config.window_size] for i in range(num_samples)], axis=0)

    # Target date corresponding to each sample
    sample_target_dates = common_dates[config.window_size: config.window_size + num_samples]

    # 6. Generate labels, masks, returns, and thresholds strictly from raw prices
    labels, masks, returns, thresholds = generate_paper_aligned_labels(
        raw_close=raw_close,
        raw_volume=raw_volume,
        window_size=config.window_size,
        pred_horizon=config.pred_horizon,
        base_threshold=config.base_threshold,
        k_factor=config.k_factor,
        min_volume_change=config.min_volume_change,
        stochastic_dropout=config.stochastic_dropout,
        label_mode=config.label_mode,
        train_split_idx=config.train_samples,
        seed=config.primary_seed
    )

    # 7. Identify the 61-day stress testing window (November 2024 to January 2025)
    # Target dates in sample_target_dates
    is_stress_date = np.array([
        ('2024-11-01' <= str(d) <= '2025-01-31') for d in sample_target_dates
    ])
    # The out-of-sample period starts at index 856
    oos_indices = np.arange(config.train_samples, num_samples)
    stress_indices = np.array([idx for idx in oos_indices if is_stress_date[idx]])
    clean_val_indices = np.array([idx for idx in oos_indices if not is_stress_date[idx]])

    logging.info(f"Total OOS samples: {len(oos_indices)}")
    logging.info(f"Clean validation samples: {len(clean_val_indices)} (expected ~307)")
    logging.info(f"Stress test samples: {len(stress_indices)} (expected ~61)")

    # 8. Leakage-safe per-feature normalization fitted exclusively on training set (first 856 samples)
    X_train_raw = X[:config.train_samples]
    # Compute mean and std per feature in float64 to avoid float32 accumulation truncation over 1.28M elements
    train_mean = np.mean(X_train_raw, axis=(0, 1, 2), dtype=np.float64, keepdims=True).astype(np.float32)  # [1, 1, 1, 24]
    train_std = np.std(X_train_raw, axis=(0, 1, 2), dtype=np.float64, keepdims=True).astype(np.float32)   # [1, 1, 1, 24]
    train_std = np.where(train_std < 1e-6, 1.0, train_std)

    # Apply frozen statistics forward
    X_norm = (X - train_mean) / train_std
    X_norm = np.nan_to_num(X_norm, nan=0.0)

    # 9. Slice partitions
    train_slice = slice(0, config.train_samples)
    X_train = X_norm[train_slice]
    y_train = labels[train_slice]
    m_train = masks[train_slice]

    X_val_clean = X_norm[clean_val_indices]
    y_val_clean = labels[clean_val_indices]
    m_val_clean = masks[clean_val_indices]

    X_stress = X_norm[stress_indices]
    y_stress = labels[stress_indices]
    m_stress = masks[stress_indices]

    X_oos_all = X_norm[oos_indices]
    y_oos_all = labels[oos_indices]
    m_oos_all = masks[oos_indices]

    # Save processed artifacts
    dataset_out = os.path.join(config.artifacts_dir, "paper_aligned_dataset.npz")
    np.savez_compressed(
        dataset_out,
        X_train=X_train.astype(np.float32),
        y_train=y_train.astype(np.float32),
        m_train=m_train,
        X_val_clean=X_val_clean.astype(np.float32),
        y_val_clean=y_val_clean.astype(np.float32),
        m_val_clean=m_val_clean,
        X_stress=X_stress.astype(np.float32),
        y_stress=y_stress.astype(np.float32),
        m_stress=m_stress,
        X_oos_all=X_oos_all.astype(np.float32),
        y_oos_all=y_oos_all.astype(np.float32),
        m_oos_all=m_oos_all,
        train_mean=train_mean.astype(np.float32),
        train_std=train_std.astype(np.float32),
        stock_order=stock_order,
        features=FEATURE_NAMES,
        sample_target_dates=sample_target_dates
    )
    logging.info(f"Saved paper-aligned dataset to {dataset_out}")

    return {
        'dataset_path': dataset_out,
        'train_samples': len(X_train),
        'clean_val_samples': len(X_val_clean),
        'stress_samples': len(X_stress),
        'oos_all_samples': len(X_oos_all),
        'stock_order': stock_order,
        'features': FEATURE_NAMES
    }


def get_paper_aligned_dataloaders(
    config: PaperAlignedConfig,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Load processed dataset from disk and return PyTorch DataLoaders.
    """
    dataset_file = os.path.join(config.artifacts_dir, "paper_aligned_dataset.npz")
    if not os.path.exists(dataset_file):
        build_paper_aligned_dataset(config)

    data = np.load(dataset_file, allow_pickle=True)

    train_ds = STGRADDataset(data['X_train'], data['y_train'], data['m_train'])
    val_clean_ds = STGRADDataset(data['X_val_clean'], data['y_val_clean'], data['m_val_clean'])
    stress_ds = STGRADDataset(data['X_stress'], data['y_stress'], data['m_stress'])
    oos_all_ds = STGRADDataset(data['X_oos_all'], data['y_oos_all'], data['m_oos_all'])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=config.shuffle_train)
    val_clean_loader = DataLoader(val_clean_ds, batch_size=config.batch_size, shuffle=False)
    stress_loader = DataLoader(stress_ds, batch_size=config.batch_size, shuffle=False)
    oos_all_loader = DataLoader(oos_all_ds, batch_size=config.batch_size, shuffle=False)

    return {
        'train': train_loader,
        'val_clean': val_clean_loader,
        'stress': stress_loader,
        'oos_all': oos_all_loader,
        'stock_order': data['stock_order'],
        'sample_target_dates': data['sample_target_dates']
    }
