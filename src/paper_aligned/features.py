"""
Causal Technical Indicator Extraction for Track B: paper_aligned.
Computes the 24 technical indicators specified in Table 5 of Fu et al. (2026).
Ensures zero lookahead leakage by using purely trailing rolling windows.
"""

from typing import List, Dict
import numpy as np
import pandas as pd

FEATURE_NAMES: List[str] = [
    'Relative_Close',           # 0
    'OC_Ratio',                 # 1
    'VWAP',                     # 2
    'RSI',                      # 3
    'RSI_Relative',             # 4
    'MACD_Signal',              # 5
    'Volume_Ratio',             # 6
    'ROC',                      # 7
    'Momentum_5',               # 8
    'Momentum_10',              # 9
    'MA5_vs_MA20',              # 10
    'BB_Width',                 # 11
    'BB_Position',              # 12
    'ATR',                      # 13
    'Volume_Change_Pct',        # 14
    'Close_vs_High',            # 15
    'Volatility',               # 16
    'Price_Volume_Corr',        # 17
    'Volume_Momentum',          # 18
    'Price_Acceleration',       # 19
    'High_Low_Ratio',           # 20
    'Close_MA_Ratio',           # 21
    'Sector_Relative_Strength', # 22
    'Volatility_Ratio'          # 23
]


def extract_features_for_ticker(
    df: pd.DataFrame,
    vwap_mode: str = "table_literal",
    eps: float = 1e-8
) -> pd.DataFrame:
    """
    Extract the 24 Table 5 technical features for a single stock dataframe.
    Input df must contain: ['open', 'high', 'low', 'close', 'volume'].
    """
    df = df.copy()

    # Ensure price columns are clean and forward-filled
    price_cols = ['open', 'high', 'low', 'close']
    df[price_cols] = df[price_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    df['volume'] = df['volume'].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 0. Relative Close: Close / MA30(Close)
    ma30_close = df['close'].rolling(30, min_periods=1).mean()
    df['Relative_Close'] = df['close'] / (ma30_close + eps)

    # 1. OC Ratio: (Close - Open) / Open
    df['OC_Ratio'] = (df['close'] - df['open']) / (df['open'] + eps)

    # 2. VWAP
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    if vwap_mode == "conventional":
        cum_tp_vol = (typical_price * df['volume']).rolling(30, min_periods=1).sum()
        cum_vol = df['volume'].rolling(30, min_periods=1).sum()
        df['VWAP'] = cum_tp_vol / (cum_vol + eps)
    else:
        # Table 5 literal expression
        tp_sum = typical_price.rolling(30, min_periods=1).sum()
        vol_sum = df['volume'].rolling(30, min_periods=1).sum()
        df['VWAP'] = (tp_sum + df['volume'] * df['close']) / (vol_sum + df['volume'] + eps)

    # 3 & 4. RSI (k=21) & Relative RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(21, min_periods=1).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(21, min_periods=1).mean()
    rs = gain / (loss + eps)
    df['RSI'] = (100.0 - (100.0 / (1.0 + rs))).clip(0.0, 100.0)
    df['RSI_Relative'] = df['RSI'] / 50.0 - 1.0

    # 5. MACD Signal: Normalized MACD histogram
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    df['MACD_Signal'] = macd_hist / (df['close'] + eps)

    # 6. Volume Ratio: Volume / MA30(Volume)
    ma30_vol = df['volume'].rolling(30, min_periods=1).mean()
    df['Volume_Ratio'] = df['volume'] / (ma30_vol + eps)

    # 7. ROC: 12-day price rate of change
    df['ROC'] = df['close'].pct_change(periods=12).fillna(0.0)

    # 8 & 9. Momentum 5 and 10
    df['Momentum_5'] = df['close'] / (df['close'].shift(5) + eps) - 1.0
    df['Momentum_10'] = df['close'] / (df['close'].shift(10) + eps) - 1.0

    # 10. MA5 vs MA20
    ma5_close = df['close'].rolling(5, min_periods=1).mean()
    ma20_close = df['close'].rolling(20, min_periods=1).mean()
    df['MA5_vs_MA20'] = (ma5_close - ma20_close) / (ma20_close + eps)

    # 11 & 12. Bollinger Band Width & Position
    std20_close = df['close'].rolling(20, min_periods=1).std().fillna(0.0)
    df['BB_Width'] = (4.0 * std20_close) / (ma20_close + eps)
    bb_low = ma20_close - 2.0 * std20_close
    df['BB_Position'] = (df['close'] - bb_low) / (4.0 * std20_close + eps)

    # 13. ATR (k=14)
    prev_close = df['close'].shift(1).bfill()
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs()
        )
    )
    df['ATR'] = tr.rolling(14, min_periods=1).mean()

    # 14. Volume Change Percentage (vol_{i,t})
    df['Volume_Change_Pct'] = df['volume'].pct_change().fillna(0.0)

    # 15. Close vs High (position in daily range)
    df['Close_vs_High'] = (df['close'] - df['low']) / (df['high'] - df['low'] + eps)

    # 16. Volatility: 5-day std / 5-day mean
    std5_close = df['close'].rolling(5, min_periods=1).std().fillna(0.0)
    df['Volatility'] = std5_close / (ma5_close + eps)

    # 17. Price Volume Correlation (10-day rolling)
    ret1 = df['close'].pct_change().fillna(0.0)
    vol_pct = df['Volume_Change_Pct']
    df['Price_Volume_Corr'] = ret1.rolling(10, min_periods=1).corr(vol_pct).fillna(0.0)

    # 18. Volume Momentum (5-day)
    df['Volume_Momentum'] = df['volume'].pct_change(periods=5).fillna(0.0)

    # 19. Price Acceleration (diff of 1-day return)
    df['Price_Acceleration'] = ret1.diff().fillna(0.0)

    # 20. High Low Ratio
    df['High_Low_Ratio'] = df['high'] / (df['low'] + eps)

    # 21. Close MA Ratio
    df['Close_MA_Ratio'] = df['close'] / (ma20_close + eps)

    # 22. Sector Relative Strength (proxy as printed)
    mom5 = ret1.rolling(5, min_periods=1).mean()
    df['Sector_Relative_Strength'] = mom5 - mom5.rolling(10, min_periods=1).mean()

    # 23. Volatility Ratio: std5 / std20
    df['Volatility_Ratio'] = std5_close / (std20_close + eps)

    # Sanitize and select the 24 columns in exact order
    result = df[FEATURE_NAMES].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return result
