"""
Graph Construction & Normalization for Track B: paper_aligned.
Builds and symmetrically normalizes the static behavioral prior graph (Eq. 1).
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
import torch


def symmetric_degree_normalize(adj: np.ndarray) -> np.ndarray:
    """
    Symmetrically normalize adjacency matrix according to Eq. (1):
      A_norm = D^(-1/2) A D^(-1/2)
    where D_ii = sum_j A_ij.
    """
    adj = np.asarray(adj, dtype=np.float32)
    row_sum = adj.sum(axis=1)
    d_inv_sqrt = np.zeros_like(row_sum)
    mask = row_sum > 0
    d_inv_sqrt[mask] = np.power(row_sum[mask], -0.5)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    return normalized.astype(np.float32)


def get_prior_adjacency_matrix(
    cluster_file_path: str,
    stock_order_path: str,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Construct normalized prior adjacency matrix A_norm from trend clusters.
    """
    df = pd.read_csv(cluster_file_path)
    stocks = df['Stock'].values
    trend_clusters = df['Trend'].values
    stock_order = np.load(stock_order_path, allow_pickle=True)

    # Map clusters in stock_order sequence
    cluster_map = {s: c for s, c in zip(stocks, trend_clusters)}
    ordered_clusters = [cluster_map[s] for s in stock_order]
    num_stocks = len(stock_order)

    adj = np.zeros((num_stocks, num_stocks), dtype=np.float32)
    for i in range(num_stocks):
        for j in range(num_stocks):
            if i == j or ordered_clusters[i] == ordered_clusters[j]:
                adj[i, j] = 1.0

    a_norm = symmetric_degree_normalize(adj)
    return torch.FloatTensor(a_norm).to(device)


def build_train_only_prior_adjacency(
    raw_close_train: np.ndarray,
    stock_order: np.ndarray,
    n_clusters: int = 5,
    seed: int = 42,
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Construct prior adjacency matrix strictly using training period observations.
    Implements Section 2.3 Macro-Trend Profiling on training window:
      1. Sequence mean
      2. Volatility (standard deviation)
      3. Linear trend slope
      4. Overall period return
      5. Peak-to-peak amplitude
    Followed by Z-score standardization and K-Means clustering (K=5).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    num_dates, num_stocks = raw_close_train.shape
    stats = np.zeros((num_stocks, 5), dtype=np.float32)

    time_idx = np.arange(num_dates)
    for i in range(num_stocks):
        p = raw_close_train[:, i]
        p_mean = float(np.mean(p))
        p_std = float(np.std(p))
        slope = float(np.polyfit(time_idx, p, 1)[0])
        ret = float((p[-1] - p[0]) / (p[0] + 1e-8))
        p2p = float((np.max(p) - np.min(p)) / (np.min(p) + 1e-8))
        stats[i] = [p_mean, p_std, slope, ret, p2p]

    scaled_stats = StandardScaler().fit_transform(stats)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_stats)

    adj = np.zeros((num_stocks, num_stocks), dtype=np.float32)
    for i in range(num_stocks):
        for j in range(num_stocks):
            if i == j or cluster_labels[i] == cluster_labels[j]:
                adj[i, j] = 1.0

    a_norm = symmetric_degree_normalize(adj)
    return torch.FloatTensor(a_norm).to(device), cluster_labels

