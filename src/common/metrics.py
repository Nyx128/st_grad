"""
Common evaluation metrics for ST-GRAD reproduction.
Computes Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
Supports optional boolean masking to exclude invalid/filtered samples.
"""

from typing import Dict, Optional, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probabilities for the positive class (optional).
        mask: Optional boolean mask where True means the sample is valid for evaluation.

    Returns:
        Dictionary with accuracy, precision, recall, f1, and optionally auc.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if mask is not None:
        mask = np.asarray(mask).flatten().astype(bool)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if y_prob is not None:
            y_prob = np.asarray(y_prob).flatten()[mask]
    elif y_prob is not None:
        y_prob = np.asarray(y_prob).flatten()

    if len(y_true) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.5
        }

    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }

    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics['confusion_matrix'] = [[int(tn), int(fp)], [int(fn), int(tp)]]
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tp'] = int(tp)

    if y_prob is not None:
        if len(np.unique(y_true)) > 1:
            metrics['auc'] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics['auc'] = 0.5
    else:
        metrics['auc'] = 0.5

    return metrics
