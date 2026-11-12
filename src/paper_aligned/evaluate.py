"""
Evaluation routines for Track B: paper_aligned.
Evaluates model on masked batches and computes Accuracy, Precision, Recall, F1, and ROC-AUC.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common.metrics import compute_metrics


def evaluate_paper_aligned_model(
    model: nn.Module,
    data_loader: DataLoader,
    A: torch.Tensor,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    return_arrays: bool = False
) -> Any:
    """
    Evaluate ST_GRAD_PaperAligned on a DataLoader.
    Filters predictions and targets through the sample validity mask.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for X_batch, y_batch, m_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            m_batch = m_batch.to(device)

            logits = model(X_batch, A)
            if criterion is not None:
                # Masked loss reduction
                raw_loss = F_binary_cross_entropy_with_logits(logits, y_batch, reduction='none')
                masked_loss = (raw_loss * m_batch.float()).sum() / (m_batch.float().sum() + 1e-8)
                total_loss += masked_loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
            all_masks.append(m_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds).flatten()
    all_probs = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    all_masks = np.concatenate(all_masks).flatten()

    metrics = compute_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_prob=all_probs,
        mask=all_masks
    )

    if return_arrays:
        return metrics, all_probs, all_preds, all_labels, all_masks
    return metrics


def F_binary_cross_entropy_with_logits(logits, targets, reduction='none'):
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction=reduction)
