"""Tests for customer feature validation."""

import numpy as np
import pandas as pd
import pytest

from churn.validation import validate_features


def test_validate_features_orders_columns_and_preserves_input():
    """Return features in training order without modifying the input."""
    features = pd.DataFrame(
        {
            "seconds_of_use": [120.0, 300.0],
            "call_failure": [1, 2],
        },
        index=[10, 20],
    )
    original = features.copy(deep=True)

    result = validate_features(
        features,
        expected_columns=["call_failure", "seconds_of_use"],
    )

    expected = original.loc[:, ["call_failure", "seconds_of_use"]]
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(features, original)
    assert result is not features


@pytest.mark.parametrize(
    ("features", "expected_columns", "message"),
    [
        (
            pd.DataFrame(columns=["call_failure"]),
            ["call_failure"],
            "Provide at least one customer",
        ),
        (
            pd.DataFrame([[1, 2]], columns=["call_failure", "call_failure"]),
            ["call_failure"],
            "Duplicate feature names",
        ),
        (
            pd.DataFrame({"call_failure": [1]}),
            ["call_failure", "seconds_of_use"],
            "Missing: .*seconds_of_use",
        ),
        (
            pd.DataFrame({"call_failure": [1], "churn": [0]}),
            ["call_failure"],
            "unexpected: .*churn",
        ),
        (
            pd.DataFrame({"call_failure": [None]}),
            ["call_failure"],
            "Missing values",
        ),
        (
            pd.DataFrame({"call_failure": ["1"]}),
            ["call_failure"],
            "Non-numeric values",
        ),
        (
            pd.DataFrame({"call_failure": [np.inf]}),
            ["call_failure"],
            "must be finite",
        ),
        (
            pd.DataFrame({"call_failure": [-np.inf]}),
            ["call_failure"],
            "must be finite",
        ),
        (
            pd.DataFrame({"call_failure": [np.nan]}),
            ["call_failure"],
            "Missing values",
        ),
    ],
)
def test_validate_features_rejects_invalid_input(features, expected_columns, message):
    """Reject customer data that violates the feature requirements."""
    with pytest.raises(ValueError, match=message):
        validate_features(features, expected_columns)
