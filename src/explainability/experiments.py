"""
Phase 7: Controlled Explainability Experiment Runner.
Executes rigorous hypothesis tests for H1, H2, H3, and H4:
- H1: Feature and temporal attribution faithfulness (deletion/insertion + Adebayo sanity check).
- H2: Distinguishability of MVCA views via counterfactual interventions.
- H3: Disentanglement of Static Prior vs Adaptive Graph propagation and neighbor influence.
- H4: Stability across 5 training seeds and input perturbations.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from src.paper_aligned.config import PaperAlignedConfig
from src.paper_aligned.dataset import get_paper_aligned_dataloaders
from src.paper_aligned.graph import get_prior_adjacency_matrix
from src.paper_aligned.model import ST_GRAD_PaperAligned
from src.paper_aligned.features import FEATURE_NAMES

from src.explainability.attribution import IntegratedGradients
from src.explainability.interventions import ViewInterventions, GraphInterventions
from src.explainability.faithfulness import deletion_insertion_test, model_randomization_test
from src.explainability.stability import evaluate_seed_stability, evaluate_perturbation_stability


def load_model(checkpoint_path: str, config: PaperAlignedConfig, device: torch.device) -> nn.Module:
    model = ST_GRAD_PaperAligned(
        in_channels=config.num_features,
        num_nodes=config.num_stocks,
        seq_len=config.window_size,
        hidden_channels=config.hidden_channels,
        num_layers=config.num_layers,
        diffusion_steps=config.diffusion_steps,
        embed_dim=config.embed_dim,
        dropout_input=config.dropout_input,
        dropout_head=config.dropout_head
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def run_explainability_experiments(
    num_samples_to_explain: int = 15,
    output_dir: str = "artifacts/explainability"
) -> Dict[str, Any]:
    """
    Run complete Phase 7 experimental evaluation across H1, H2, H3, and H4.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    os.makedirs(output_dir, exist_ok=True)
    config = PaperAlignedConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Starting Phase 7 Explainability Experiments on {device}...")

    # Load 5 seed models
    exp_dir = os.path.join(config.artifacts_dir, "experiments")
    seeds = [42, 43, 44, 45, 46]
    models = []
    for s in seeds:
        ckpt = os.path.join(exp_dir, f"best_model_seed_{s}.pth")
        if not os.path.exists(ckpt):
            # Fallback to artifacts root
            ckpt = os.path.join(config.artifacts_dir, f"best_model_seed_{s}.pth")
        models.append(load_model(ckpt, config, device))
    primary_model = models[0]  # Seed 42 as primary model

    dataloaders = get_paper_aligned_dataloaders(config, device=device)
    val_loader = dataloaders['val_clean']
    stress_loader = dataloaders['stress']

    A_norm = get_prior_adjacency_matrix(
        config.cluster_file_path,
        config.stock_order_path,
        device=device
    )

    stock_order = np.load(config.stock_order_path, allow_pickle=True)

    # Collect sample batches for testing
    val_batches = []
    for x_b, y_b, m_b in val_loader:
        val_batches.append((x_b.to(device), y_b.to(device), m_b.to(device)))
        if len(val_batches) * config.batch_size >= num_samples_to_explain:
            break

    # =========================================================================
    # HYPOTHESIS 1: Input Attribution Faithfulness & Period Sensitivity
    # =========================================================================
    logging.info("--- Testing Hypothesis H1: Input Attribution Faithfulness ---")
    ig = IntegratedGradients(primary_model)

    faithfulness_results = []
    randomization_results = []
    top_features_overall = np.zeros(config.num_features)
    temporal_profile_overall = np.zeros(config.window_size)

    sample_count = 0
    for x_b, y_b, m_b in val_batches:
        for idx_in_batch in range(len(x_b)):
            if sample_count >= num_samples_to_explain:
                break
            x_sample = x_b[idx_in_batch:idx_in_batch + 1]

            # Choose 2 target stocks per sample (e.g., stock 0 and stock 25)
            for target_stock in [0, 25]:
                if not m_b[idx_in_batch, target_stock]:
                    continue

                attr_res = ig.attribute(x_sample, A_norm, target_stock_idx=target_stock, steps=25)
                f_imp = attr_res['feature_importance_abs']
                top_features_overall += f_imp
                temporal_profile_overall += attr_res['temporal_importance_abs']

                # Deletion/Insertion test
                del_ins = deletion_insertion_test(
                    primary_model, x_sample, A_norm, target_stock_idx=target_stock,
                    feature_attributions=f_imp, steps=[1, 2, 4, 8, 12, 16]
                )
                faithfulness_results.append(del_ins)

                # Parameter randomization check on first 4 cases
                if len(randomization_results) < 4:
                    rand_check = model_randomization_test(
                        primary_model, x_sample, A_norm, target_stock_idx=target_stock,
                        original_attributions=f_imp
                    )
                    randomization_results.append(rand_check)

            sample_count += 1

    mean_auc_top_del = float(np.mean([r['auc_deletion_top'] for r in faithfulness_results]))
    mean_auc_rand_del = float(np.mean([r['auc_deletion_random'] for r in faithfulness_results]))
    mean_auc_ins_top = float(np.mean([r['auc_insertion_top'] for r in faithfulness_results]))
    faithfulness_ratio = mean_auc_top_del / (mean_auc_rand_del + 1e-8)

    # Statistical test: top deletion causes greater drop than random
    del_diffs = [r['auc_deletion_top'] - r['auc_deletion_random'] for r in faithfulness_results]
    t_stat, p_val = stats.ttest_1samp(del_diffs, 0.0)
    h1_passed = bool(t_stat > 0 and p_val < 0.01)

    mean_rand_spearman = float(np.mean([r['spearman_correlation'] for r in randomization_results]))
    rand_sanity_passed = bool(all(r['passed_sanity_check'] for r in randomization_results))

    # Feature ranking
    feature_ranking_indices = np.argsort(top_features_overall)[::-1]
    feature_ranking = [
        {'rank': r + 1, 'feature': FEATURE_NAMES[i], 'importance_score': float(top_features_overall[i])}
        for r, i in enumerate(feature_ranking_indices)
    ]

    h1_summary = {
        'hypothesis': 'H1: Input Attribution Faithfulness',
        'passed': bool(h1_passed and rand_sanity_passed),
        'metrics': {
            'mean_auc_top_deletion': mean_auc_top_del,
            'mean_auc_random_deletion': mean_auc_rand_del,
            'mean_auc_top_insertion': mean_auc_ins_top,
            'faithfulness_ratio': float(faithfulness_ratio),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'model_randomization_mean_spearman': mean_rand_spearman,
            'passed_parameter_randomization_sanity_check': rand_sanity_passed
        },
        'top_5_features': feature_ranking[:5],
        'bottom_5_features': feature_ranking[-5:],
        'temporal_recency_bias_ratio': float(
            np.sum(temporal_profile_overall[-5:]) / (np.sum(temporal_profile_overall[:5]) + 1e-8)
        )
    }

    # =========================================================================
    # HYPOTHESIS 2: MVCA View Distinguishability via Interventions
    # =========================================================================
    logging.info("--- Testing Hypothesis H2: MVCA View Interventions ---")
    view_intervener = ViewInterventions(primary_model)
    view_intervention_cases = []

    for x_b, y_b, m_b in val_batches[:2]:
        for idx in range(min(len(x_b), 5)):
            x_s = x_b[idx:idx + 1]
            for target_stock in [0, 10, 20]:
                res = view_intervener.evaluate_view_contributions(x_s, A_norm, target_stock_idx=target_stock)
                view_intervention_cases.append(res)

    # Analyze variance of view rankings
    st_impacts = [c['interventions']['ST']['importance_rank_score'] for c in view_intervention_cases]
    ds_impacts = [c['interventions']['DS']['importance_rank_score'] for c in view_intervention_cases]
    dt_impacts = [c['interventions']['DT']['importance_rank_score'] for c in view_intervention_cases]

    # Check that impacts are distinguishable (non-identical)
    f_stat, p_view = stats.f_oneway(st_impacts, ds_impacts, dt_impacts)
    top_ranks = [c['view_importance_ranking'][0] for c in view_intervention_cases]
    unique_top_views = set(top_ranks)

    h2_passed = bool(len(unique_top_views) > 1 or f_stat > 3.0)

    h2_summary = {
        'hypothesis': 'H2: MVCA View Distinguishability via Interventions',
        'passed': h2_passed,
        'metrics': {
            'mean_st_logit_impact': float(np.mean(st_impacts)),
            'mean_ds_logit_impact': float(np.mean(ds_impacts)),
            'mean_dt_logit_impact': float(np.mean(dt_impacts)),
            'f_statistic': float(f_stat),
            'p_value': float(p_view),
            'unique_top_ranked_views_across_instances': list(unique_top_views),
            'global_softmax_weights_block_0': view_intervention_cases[0]['global_fusion_weights'][0]
        }
    }

    # =========================================================================
    # HYPOTHESIS 3: Graph Channels (Static vs Adaptive) & Neighbor Influence
    # =========================================================================
    logging.info("--- Testing Hypothesis H3: Graph Interventions ---")
    graph_intervener = GraphInterventions(primary_model)

    static_deltas = []
    adapt_deltas = []
    neighbor_deltas = []

    for x_b, y_b, m_b in val_batches[:2]:
        for idx in range(min(len(x_b), 5)):
            x_s = x_b[idx:idx + 1]
            target_stock = 0
            g_res = graph_intervener.evaluate_graph_channels(x_s, A_norm, target_stock_idx=target_stock)
            static_deltas.append(abs(g_res['static_prior_ablation']['delta_logit']))
            adapt_deltas.append(abs(g_res['adaptive_graph_ablation']['delta_logit']))

            n_res = graph_intervener.evaluate_neighbor_node_influence(x_s, A_norm, target_stock_idx=target_stock)
            neighbor_deltas.append(n_res['top_influential_peers'][0]['delta_logit'])

    mean_static = float(np.mean(static_deltas))
    mean_adapt = float(np.mean(adapt_deltas))
    mean_neighbor = float(np.mean(neighbor_deltas))

    # Graph channels are distinct if their impacts differ significantly
    t_stat_g, p_val_g = stats.ttest_rel(static_deltas, adapt_deltas)
    h3_passed = bool(p_val_g < 0.05 or abs(mean_static - mean_adapt) > 0.05)

    h3_summary = {
        'hypothesis': 'H3: Graph Interventions & Channel Disentanglement',
        'passed': h3_passed,
        'metrics': {
            'mean_static_prior_ablation_delta_logit': mean_static,
            'mean_adaptive_graph_ablation_delta_logit': mean_adapt,
            'mean_top_peer_node_delta_logit': mean_neighbor,
            'static_vs_adaptive_t_stat': float(t_stat_g),
            'static_vs_adaptive_p_value': float(p_val_g)
        }
    }

    # =========================================================================
    # HYPOTHESIS 4: Cross-Seed and Input Perturbation Stability
    # =========================================================================
    logging.info("--- Testing Hypothesis H4: Stability across Seeds and Perturbations ---")
    x_test_sample = val_batches[0][0][0:1]

    # Seed stability across all 5 models
    seed_stability = evaluate_seed_stability(models, x_test_sample, A_norm, target_stock_idx=0, top_k=5)

    # Input perturbation stability
    perturb_stability = evaluate_perturbation_stability(primary_model, x_test_sample, A_norm, target_stock_idx=0, noise_std=0.02)

    h4_passed = bool(seed_stability['is_stable'] and perturb_stability['is_robust'])

    h4_summary = {
        'hypothesis': 'H4: Cross-Seed & Input Perturbation Stability',
        'passed': h4_passed,
        'cross_seed_stability': seed_stability,
        'input_perturbation_stability': perturb_stability
    }

    # =========================================================================
    # Master Report Compilation
    # =========================================================================
    master_report = {
        'experiment_name': 'phase_7_controlled_explainability_experiments',
        'num_samples_evaluated': num_samples_to_explain,
        'summary': {
            'H1_faithfulness': h1_summary['passed'],
            'H2_view_distinguishability': h2_summary['passed'],
            'H3_graph_channels': h3_summary['passed'],
            'H4_stability': h4_summary['passed']
        },
        'H1_input_attribution': h1_summary,
        'H2_mvca_interventions': h2_summary,
        'H3_graph_interventions': h3_summary,
        'H4_stability_results': h4_summary
    }

    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    report_path = os.path.join(output_dir, "explainability_experiments_report.json")
    with open(report_path, "w") as f:
        json.dump(master_report, f, indent=2, default=convert_numpy)

    logging.info(f"Phase 7 Explainability Report successfully saved to {report_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("      PHASE 7 CONTROLLED EXPLAINABILITY HYPOTHESIS EVALUATION")
    print("=" * 80)
    print(f"{'Hypothesis':<40} | {'Test Metric':<20} | {'Result':<15}")
    print("-" * 80)
    h1_faith_str = f"Faith Ratio: {faithfulness_ratio:.2f}x"
    h1_rand_str = f"Corr: {mean_rand_spearman:.3f}"
    h2_fstat_str = f"F-Stat: {f_stat:.2f}"
    h3_pval_str = f"P-Val: {p_val_g:.4f}"
    h4_seed_str = f"Rank Corr: {seed_stability['mean_pairwise_spearman_rank_correlation']:.3f}"
    h4_pert_str = f"Cosine Sim: {perturb_stability['mean_cosine_similarity']:.3f}"

    print(f"{'H1: Attribution Faithfulness':<40} | {h1_faith_str:<20} | {'CONFIRMED' if h1_passed else 'REJECTED'}")
    print(f"{'H1: Parameter Randomization Sanity':<40} | {h1_rand_str:<20} | {'CONFIRMED' if rand_sanity_passed else 'REJECTED'}")
    print(f"{'H2: MVCA View Distinguishability':<40} | {h2_fstat_str:<20} | {'CONFIRMED' if h2_passed else 'REJECTED'}")
    print(f"{'H3: Static vs Adaptive Channels':<40} | {h3_pval_str:<20} | {'CONFIRMED' if h3_passed else 'REJECTED'}")
    print(f"{'H4: Cross-Seed Stability':<40} | {h4_seed_str:<20} | {'CONFIRMED' if seed_stability['is_stable'] else 'REJECTED'}")
    print(f"{'H4: Perturbation Robustness':<40} | {h4_pert_str:<20} | {'CONFIRMED' if perturb_stability['is_robust'] else 'REJECTED'}")
    print("=" * 80)

    return master_report


if __name__ == "__main__":
    run_explainability_experiments()
