"""
Unified Command-Line Interface for ST-GRAD (Phase 4).
Provides end-to-end operational commands matching Section 11 of the Research Plan:
- python src/cli.py explain --ticker HPE --split val_clean --index 0
- python src/cli.py evaluate --track paper_aligned --split clean_val
- python src/cli.py evaluate-explanations --track paper_aligned --samples 15
- python src/cli.py audit-data
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.explainability.explainer import STGradExplainer
from src.explainability.experiments import run_explainability_experiments
from src.explainability.regime_benchmark import run_full_phase5_benchmark
from src.paper_aligned.config import PaperAlignedConfig
from src.paper_aligned.dataset import get_paper_aligned_dataloaders
from src.paper_aligned.graph import get_prior_adjacency_matrix
from src.paper_aligned.evaluate import evaluate_paper_aligned_model
from src.repo_compat.evaluate import evaluate_repo_compat_checkpoint
from src.explainability.adapter import ModelTrackAdapter
from src.explainability.report import format_markdown_card, format_html_report_card


def cmd_explain(args):
    """Execute instance-level explanation for a specified stock and date."""
    config = PaperAlignedConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_file = os.path.join(config.artifacts_dir, "paper_aligned_dataset.npz")
    if not os.path.exists(dataset_file):
        from src.paper_aligned.dataset import build_paper_aligned_dataset
        build_paper_aligned_dataset(config)

    data = np.load(dataset_file, allow_pickle=True)
    stock_order = list(data['stock_order'])
    all_dates = list(data['sample_target_dates'])

    # Resolve partition
    if args.split == "val_clean":
        x_split, y_split, m_split = data['X_val_clean'], data['y_val_clean'], data['m_val_clean']
        offset = config.train_samples
    elif args.split == "stress":
        x_split, y_split, m_split = data['X_stress'], data['y_stress'], data['m_stress']
        offset = config.train_samples + len(data['X_val_clean'])
    elif args.split == "oos_all":
        x_split, y_split, m_split = data['X_oos_all'], data['y_oos_all'], data['m_oos_all']
        offset = config.train_samples
    else:
        x_split, y_split, m_split = data['X_train'], data['y_train'], data['m_train']
        offset = 0

    if args.index >= len(x_split):
        raise IndexError(f"Sample index {args.index} exceeds split size ({len(x_split)}).")

    sample_raw = x_split[args.index:args.index + 1]
    x_sample = torch.FloatTensor(sample_raw).permute(0, 3, 1, 2).to(device)
    target_date = args.date or str(all_dates[offset + args.index])

    explainer = STGradExplainer(
        track=args.track,
        checkpoint_path=args.checkpoint,
        stock_order=stock_order,
        baseline_strategy=args.baseline_strategy,
        device=device
    )

    record = explainer.explain_instance(
        x=x_sample,
        target_stock=args.ticker,
        target_date=target_date,
        ig_steps=args.ig_steps,
        quadrature_method=args.quadrature,
        top_k_features=5,
        top_k_peers=5
    )

    stock_idx, _ = explainer.resolve_stock(args.ticker)
    true_lbl = int(y_split[args.index, stock_idx])
    record['ground_truth'] = {
        'label': true_lbl,
        'direction': 'UP' if true_lbl == 1 else 'DOWN',
        'is_valid_sample': bool(m_split[args.index, stock_idx])
    }

    card = explainer.format_markdown_card(record)
    match_str = "CORRECT" if record['prediction']['predicted_class'] == true_lbl else "INCORRECT"
    gt_section = f"\n**Ground Truth Label:** `{record['ground_truth']['direction']}` (Class `{true_lbl}`) | **Outcome:** **`{match_str}`**"
    card_lines = card.split("\n")
    card_lines.insert(7, gt_section)
    final_card = "\n".join(card_lines)

    print("\n" + final_card + "\n")

    if args.save_json:
        explainer.save_case_record(record, args.save_json)
        print(f"Case record JSON saved to: {args.save_json}")

    if args.save_card:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_card)), exist_ok=True)
        with open(args.save_card, "w") as f:
            f.write(final_card)
        print(f"Markdown card saved to: {args.save_card}")

    if args.save_html:
        html = format_html_report_card(record)
        os.makedirs(os.path.dirname(os.path.abspath(args.save_html)), exist_ok=True)
        with open(args.save_html, "w") as f:
            f.write(html)
        print(f"HTML card saved to: {args.save_html}")


def cmd_evaluate(args):
    """Evaluate predictive performance metrics for either track."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.track == "repo_compat":
        from src.repo_compat.config import RepoCompatConfig
        cfg = RepoCompatConfig()
        ckpt = args.checkpoint or cfg.best_model_path
        metrics = evaluate_repo_compat_checkpoint(checkpoint_path=ckpt, config=cfg, device=device)
    else:
        from src.paper_aligned.config import PaperAlignedConfig
        cfg = PaperAlignedConfig()
        ckpt = args.checkpoint or os.path.join(cfg.artifacts_dir, "experiments", "best_model_seed_42.pth")
        adapter = ModelTrackAdapter.load(track="paper_aligned", checkpoint_path=ckpt, device=device)
        dataloaders = get_paper_aligned_dataloaders(cfg, device=device)
        loader_key = "val_clean" if args.split in ("clean_val", "val_clean") else args.split
        loader = dataloaders[loader_key]
        A_norm = get_prior_adjacency_matrix(cfg.cluster_file_path, cfg.stock_order_path, device=device)
        metrics = evaluate_paper_aligned_model(adapter.raw_model, loader, A_norm, device=device)

    print(f"\nEvaluation Results ({args.track.upper()} on {args.split}):")
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"  {k:<20}: {v:.4f}")
        else:
            print(f"  {k:<20}: {v}")
    print()


def cmd_evaluate_explanations(args):
    """Run controlled hypothesis testing suite (H1-H4)."""
    run_explainability_experiments(
        track=args.track,
        num_samples_to_explain=args.samples,
        output_dir=args.output_dir
    )


def cmd_audit_data(args):
    """Run data integrity and validity mask audit."""
    from src.paper_aligned.dataset import build_paper_aligned_dataset
    config = PaperAlignedConfig()
    summary = build_paper_aligned_dataset(config)
    print("\nData Integrity Audit Summary:")
    for k, v in summary.items():
        if isinstance(v, list):
            print(f"  {k:<25}: count={len(v)}")
        else:
            print(f"  {k:<25}: {v}")
    print()


def cmd_benchmark_regimes(args):
    """Run Phase 5 Dual-Track & Regime Shift Benchmark (Hypothesis H6)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_full_phase5_benchmark(
        num_clean=args.samples_clean,
        num_stress=args.samples_stress,
        output_dir=args.output_dir,
        device=device
    )


def main():
    parser = argparse.ArgumentParser(description="ST-GRAD Unified CLI Orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: explain
    p_exp = subparsers.add_parser("explain", help="Generate explanation for an instance prediction")
    p_exp.add_argument("--ticker", type=str, default="HPE", help="Stock ticker symbol")
    p_exp.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD")
    p_exp.add_argument("--split", type=str, default="val_clean", choices=["train", "val_clean", "stress", "oos_all"])
    p_exp.add_argument("--index", type=int, default=0, help="Sample index within split")
    p_exp.add_argument("--track", type=str, default="paper_aligned", choices=["paper_aligned", "repo_compat"])
    p_exp.add_argument("--checkpoint", type=str, default=None)
    p_exp.add_argument("--baseline_strategy", type=str, default="zero", choices=["zero", "train_mean"])
    p_exp.add_argument("--ig_steps", type=int, default=50)
    p_exp.add_argument("--quadrature", type=str, default="gauss_legendre", choices=["gauss_legendre", "riemann"])
    p_exp.add_argument("--save_json", type=str, default=None)
    p_exp.add_argument("--save_card", type=str, default=None)
    p_exp.add_argument("--save_html", type=str, default=None)
    p_exp.set_defaults(func=cmd_explain)

    # Subcommand: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate predictive metrics on a split")
    p_eval.add_argument("--track", type=str, default="paper_aligned", choices=["paper_aligned", "repo_compat"])
    p_eval.add_argument("--split", type=str, default="clean_val", choices=["clean_val", "stress", "oos_all"])
    p_eval.add_argument("--checkpoint", type=str, default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    # Subcommand: evaluate-explanations
    p_eexp = subparsers.add_parser("evaluate-explanations", help="Run controlled hypothesis tests (H1-H4)")
    p_eexp.add_argument("--track", type=str, default="paper_aligned", choices=["paper_aligned", "repo_compat"])
    p_eexp.add_argument("--samples", type=int, default=15)
    p_eexp.add_argument("--output_dir", type=str, default="artifacts/explainability")
    p_eexp.set_defaults(func=cmd_evaluate_explanations)

    # Subcommand: audit-data
    p_audit = subparsers.add_parser("audit-data", help="Verify dataset integrity and sample masks")
    p_audit.set_defaults(func=cmd_audit_data)

    # Subcommand: benchmark-regimes
    p_bench = subparsers.add_parser("benchmark-regimes", help="Run Phase 5 dual-track and regime benchmark (H6)")
    p_bench.add_argument("--samples-clean", type=int, default=20, help="Number of clean validation dates to sample")
    p_bench.add_argument("--samples-stress", type=int, default=20, help="Number of stress window dates to sample")
    p_bench.add_argument("--output_dir", type=str, default="artifacts/explainability")
    p_bench.set_defaults(func=cmd_benchmark_regimes)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
