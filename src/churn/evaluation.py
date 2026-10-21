"""Evaluate churn prediction models using classification metrics."""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_metrics(y_true, y_pred, probs):
    """Return accuracy, precision, recall, and ROC-AUC."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, probs)),
    }
