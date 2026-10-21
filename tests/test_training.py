"""Tests for model training."""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from churn.training import train_model


def test_train_model_fits_estimator_and_returns_metrics():
    """Fit the supplied estimator and evaluate both data splits."""
    X_train = np.array([[0], [1], [2], [3]])
    y_train = np.array([0, 0, 1, 1])

    X_valid = np.array([[0.5], [2.5]])
    y_valid = np.array([0, 1])

    model = DecisionTreeClassifier(max_depth=1, random_state=42)

    fitted_model, metrics_train, metrics_valid = train_model(model, X_train, y_train, X_valid, y_valid)

    assert fitted_model is model
    np.testing.assert_array_equal(fitted_model.predict(X_valid), y_valid)

    expected = {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "auc": 1.0,
    }
    assert metrics_train == pytest.approx(expected)
    assert metrics_valid == pytest.approx(expected)
