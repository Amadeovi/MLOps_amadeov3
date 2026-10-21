"""Tests for classification metrics."""

import pytest

from churn.evaluation import get_metrics


@pytest.mark.parametrize(
    ("y_true", "y_pred", "probs", "expected"),
    [
        (
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0.1, 0.2, 0.8, 0.9],
            {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "auc": 1.0},
        ),
        (
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [0.1, 0.8, 0.4, 0.9],
            {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "auc": 0.75},
        ),
        (
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0.1, 0.2, 0.3, 0.4],
            {"accuracy": 0.5, "precision": 0.0, "recall": 0.0, "auc": 1.0},
        ),
    ],
)
def test_get_metrics(y_true, y_pred, probs, expected):
    """Calculate the expected metrics for known predictions."""
    result = get_metrics(y_true, y_pred, probs)

    assert result == pytest.approx(expected)
    assert all(isinstance(value, float) for value in result.values())
