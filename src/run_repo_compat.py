"""
Repository Compatibility Runner for ST-GRAD (Phase 0)
Ensures clean execution of the bundled repository code in modern Python/Pandas environments.
Provides backwards-compatibility for Pandas 3.0+ while leaving raw repo files untouched.
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch

# Ensure repository is on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ST-GRAD-Stock-Prediction-6E7D"))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Apply Pandas 3.0 compatibility shim for Series.fillna and DataFrame.fillna
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

# Set seed
def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(43)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure working directory is REPO_ROOT for relative data paths in train.py
os.chdir(REPO_ROOT)

from train import train_model, test_model
from utils.data_processor import process_and_save_data
from utils.graph_builder import build_adjacency_matrix

def parse_args():
    parser = argparse.ArgumentParser(description="ST-GRAD Repo Compat Runner")
    parser.add_argument('--mode', type=str, default='predict', choices=['graph', 'data', 'train', 'predict', 'all'])
    parser.add_argument('--data_dir', type=str, default=os.path.join(REPO_ROOT, 'data/raw'))
    parser.add_argument('--cluster_file', type=str, default=os.path.join(REPO_ROOT, 'data/raw/cluster_results.csv'))
    parser.add_argument('--order_path', type=str, default=os.path.join(REPO_ROOT, 'data/raw/stock_order.npy'))
    parser.add_argument('--adj_matrix_path', type=str, default=os.path.join(REPO_ROOT, 'data/processed/normalized_adj_matrix.npy'))
    parser.add_argument('--dataset_path', type=str, default=os.path.join(REPO_ROOT, 'data/processed/stock_dataset.npz'))
    parser.add_argument('--save_dir', type=str, default=os.path.join(REPO_ROOT, 'models/saved'))
    parser.add_argument('--load_path', type=str, default=os.path.join(REPO_ROOT, 'models/saved/best_model_ours.pth'))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--diffusion_steps', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.adj_matrix_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.dataset_path), exist_ok=True)

    if args.mode in ['graph', 'all']:
        logging.info("Building adjacency matrix...")
        build_adjacency_matrix(args.cluster_file, args.adj_matrix_path, args.order_path)
        logging.info("Graph construction finished.")

    if args.mode in ['data', 'all']:
        logging.info("Processing raw stock data...")
        process_and_save_data(args.data_dir, args.order_path, args.dataset_path)
        logging.info("Data processing finished.")

    if args.mode in ['train', 'all']:
        logging.info("Training ST-GRAD model (repo_compat configuration)...")
        train_model(args)
        logging.info("Training completed.")

    if args.mode == 'predict':
        logging.info(f"Evaluating checkpoint: {args.load_path}...")
        test_model(args)

if __name__ == "__main__":
    main()
