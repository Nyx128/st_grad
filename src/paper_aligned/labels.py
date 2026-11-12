"""
Target Label Generation for Track B: paper_aligned.
Strictly implements Equations (28)-(30) using raw closing prices:
  - Return: R_{i,t+1} = (P_{i,t+1} - P_bar_{i,t}) / P_bar_{i,t}
  - Threshold: T_{i,t} = T_base * (1 + k * tanh(|vol_{i,t}|))
  - Validity mask: |vol_{i,t}| >= 0.003 and valid prices
  - Stochastic dropout modes: 'clean_eval' and 'paper_literal'
"""

from typing import Tuple, Optional
import numpy as np


def generate_paper_aligned_labels(
    raw_close: np.ndarray,          # [dates, stocks] raw closing prices
    raw_volume: np.ndarray,         # [dates, stocks] raw trading volumes
    window_size: int = 30,
    pred_horizon: int = 1,
    base_threshold: float = 0.0003,
    k_factor: float = 3.0,
    min_volume_change: float = 0.003,
    stochastic_dropout: float = 0.10,
    label_mode: str = "clean_eval",
    train_split_idx: int = 856,
    seed: int = 43,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate target labels, validity masks, returns, and dynamic thresholds.

    Args:
        raw_close: Array of raw closing prices, shape [num_dates, num_stocks].
        raw_volume: Array of raw volumes, shape [num_dates, num_stocks].
        window_size: Length of feature history window (30).
        pred_horizon: Prediction horizon (1 day).
        base_threshold: T_base in Eq. (28) (0.0003).
        k_factor: k in Eq. (28) (3.0).
        min_volume_change: Minimum |vol| to be considered active (0.003).
        stochastic_dropout: Rate of positive dropout (0.10).
        label_mode: 'clean_eval' (dropout on train only) or 'paper_literal' (dropout on all).
        train_split_idx: Index demarcating training set (856).
        seed: Random seed for stochastic label smoothing.

    Returns:
        labels: Binary labels array, shape [num_samples, num_stocks] (float32).
        valid_mask: Boolean mask array, shape [num_samples, num_stocks].
        returns: Benchmark-relative returns R_{t+1}, shape [num_samples, num_stocks].
        thresholds: Dynamic thresholds T_{i,t}, shape [num_samples, num_stocks].
    """
    rng = np.random.RandomState(seed)
    num_dates, num_stocks = raw_close.shape
    num_samples = num_dates - window_size - pred_horizon + 1

    labels = np.zeros((num_samples, num_stocks), dtype=np.float32)
    valid_mask = np.zeros((num_samples, num_stocks), dtype=bool)
    returns = np.zeros((num_samples, num_stocks), dtype=np.float32)
    thresholds = np.zeros((num_samples, num_stocks), dtype=np.float32)

    # Compute volume percentage change: V_t / V_{t-1} - 1
    vol_pct_full = np.zeros_like(raw_volume)
    vol_pct_full[1:] = (raw_volume[1:] - raw_volume[:-1]) / (raw_volume[:-1] + eps)

    for i in range(num_samples):
        # Window dates: i to i + window_size - 1
        # Target day: i + window_size (next trading day)
        current_day_idx = i + window_size - 1
        target_day_idx = i + window_size

        # P_bar is the average raw close over the 30-day window
        p_bar = np.mean(raw_close[i:i + window_size], axis=0)  # [num_stocks]
        p_next = raw_close[target_day_idx]                      # [num_stocks]
        p_curr = raw_close[current_day_idx]                     # [num_stocks]

        # Volume change at day t
        vol_pct = vol_pct_full[current_day_idx]                 # [num_stocks]

        # Validity mask check
        v_mask = (
            (~np.isnan(p_curr)) & (~np.isnan(p_next)) & (~np.isnan(vol_pct)) &
            (p_curr > eps) & (p_bar > eps) & (np.abs(vol_pct) >= min_volume_change)
        )
        valid_mask[i] = v_mask

        # Dynamic threshold: Eq. (28) (strict paper formula without factor 8)
        volatility = np.tanh(np.abs(vol_pct))
        t_thresh = base_threshold * (1.0 + k_factor * volatility)
        thresholds[i] = t_thresh

        # Benchmark-relative return: Eq. (29)
        ret = (p_next - p_bar) / (p_bar + eps)
        returns[i] = ret

        # True directional upward signal
        pos_signal = (ret > t_thresh)

        # Stochastic smoothing (Eq. 30):
        # In clean_eval mode, dropout applies ONLY to training partition (i < train_split_idx).
        # In paper_literal mode, dropout applies across the entire dataset.
        if label_mode == "clean_eval" and i >= train_split_idx:
            # Deterministic evaluation targets for validation and stress testing
            labels[i, v_mask] = pos_signal[v_mask].astype(np.float32)
        else:
            survived = rng.uniform(0.0, 1.0, size=num_stocks) > stochastic_dropout
            labels[i, v_mask] = (pos_signal[v_mask] & survived[v_mask]).astype(np.float32)

    return labels, valid_mask, returns, thresholds
