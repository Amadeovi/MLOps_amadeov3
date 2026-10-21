"""Tests for customer data preprocessing."""

import pandas as pd

from churn.preprocess import preprocess_data


def test_preprocess_data_normalizes_columns_and_preserves_data():
    """Normalize column names without changing customer values."""
    df = pd.DataFrame(
        {
            "Customer ID": [1, 2],
            "Seconds  of Use": [120, 300],
            "Churn": [0, 1],
        }
    )

    expected = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "seconds_of_use": [120, 300],
            "churn": [0, 1],
        }
    )

    result = preprocess_data(df)

    pd.testing.assert_frame_equal(result, expected)
