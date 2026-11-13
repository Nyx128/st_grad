"""
CLI Entry Point: Explain a Stock Prediction with ST-GRAD.
Usage examples:
    python src/explain.py --ticker HPE --split val_clean --index 0
    python src/explain.py --ticker CTRA --date 2024-02-28 --track paper_aligned
    python src/explain.py --ticker AMKR --track repo_compat --index 5 --save_json artifacts/explainability/hpe_case.json
"""

import os
import sys
import argparse
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explainability.explainer import STGradExplainer
from src.paper_aligned.config import PaperAlignedConfig
from src.paper_aligned.dataset import get_paper_aligned_dataloaders


def main():
    parser = argparse.ArgumentParser(description="ST-GRAD Prediction Explanation CLI")
    parser.add_argument("--ticker", type=str, default="HPE", help="Stock ticker symbol (e.g. HPE, CTRA, AMKR)")
    parser.add_argument("--date", type=str, default=None, help="Target prediction date (YYYY-MM-DD)")
    parser.add_argument("--split", type=str, default="val_clean", choices=["train", "val_clean", "stress", "oos_all"],
                        help="Data split to sample from")
    parser.add_argument("--index", type=int, default=0, help="Sample index within the split")
    parser.add_argument("--track", type=str, default="paper_aligned", choices=["paper_aligned", "repo_compat"],
                        help="Model track (paper_aligned or repo_compat)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path override")
    parser.add_argument("--ig_steps", type=int, default=50, help="Gauss-Legendre integration steps")
    parser.add_argument("--save_json", type=str, default=None, help="Output JSON path to save the case record")
    parser.add_argument("--save_card", type=str, default=None, help="Output Markdown path to save the card")

    args = parser.parse_args()
    config = PaperAlignedConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load dataset artifact
    dataset_file = os.path.join(config.artifacts_dir, "paper_aligned_dataset.npz")
    if not os.path.exists(dataset_file):
        print(f"Dataset artifact not found at {dataset_file}. Building dataset...")
        from src.paper_aligned.dataset import build_paper_aligned_dataset
        build_paper_aligned_dataset(config)

    data = np.load(dataset_file, allow_pickle=True)
    stock_order = list(data['stock_order'])
    all_dates = list(data['sample_target_dates'])

    # Determine split data
    if args.split == "val_clean":
        x_split = data['X_val_clean']
        y_split = data['y_val_clean']
        m_split = data['m_val_clean']
        # Map date
        offset = config.train_samples
    elif args.split == "stress":
        x_split = data['X_stress']
        y_split = data['y_stress']
        m_split = data['m_stress']
        offset = config.train_samples + len(data['X_val_clean'])
    elif args.split == "oos_all":
        x_split = data['X_oos_all']
        y_split = data['y_oos_all']
        m_split = data['m_oos_all']
        offset = config.train_samples
    else:  # train
        x_split = data['X_train']
        y_split = data['y_train']
        m_split = data['m_train']
        offset = 0

    if args.index >= len(x_split):
        raise IndexError(f"Index {args.index} exceeds split size ({len(x_split)} samples).")

    # Sample input tensor: [T, N, F] -> [1, F, T, N]
    sample_raw = x_split[args.index:args.index + 1]
    x_sample = torch.FloatTensor(sample_raw).permute(0, 3, 1, 2).to(device)

    target_date = args.date or str(all_dates[offset + args.index])

    # 2. Instantiate Unified Explainer
    explainer = STGradExplainer(
        track=args.track,
        checkpoint_path=args.checkpoint,
        stock_order=stock_order,
        device=device
    )

    # 3. Generate Explanation
    record = explainer.explain_instance(
        x=x_sample,
        target_stock=args.ticker,
        target_date=target_date,
        ig_steps=args.ig_steps,
        top_k_features=5,
        top_k_peers=5
    )

    # Add ground-truth label to record
    stock_idx, _ = explainer.resolve_stock(args.ticker)
    true_lbl = int(y_split[args.index, stock_idx])
    is_valid = bool(m_split[args.index, stock_idx])
    record['ground_truth'] = {
        'label': true_lbl,
        'direction': 'UP' if true_lbl == 1 else 'DOWN',
        'is_valid_sample': is_valid
    }

    # 4. Generate & Print Markdown Card
    card = explainer.format_markdown_card(record)

    # Add ground truth comparison to card output
    match_str = "CORRECT" if record['prediction']['predicted_class'] == true_lbl else "INCORRECT"
    gt_section = f"\n**Ground Truth Label:** `{record['ground_truth']['direction']}` (Class `{true_lbl}`) | **Outcome:** **`{match_str}`**"
    card_lines = card.split("\n")
    # Insert after prediction summary header
    card_lines.insert(7, gt_section)
    final_card = "\n".join(card_lines)

    print("\n" + final_card + "\n")

    # 5. Optional saves
    if args.save_json:
        explainer.save_case_record(record, args.save_json)
        print(f"Case record JSON saved to: {args.save_json}")

    if args.save_card:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_card)), exist_ok=True)
        with open(args.save_card, "w") as f:
            f.write(final_card)
        print(f"Markdown card saved to: {args.save_card}")


if __name__ == "__main__":
    main()
