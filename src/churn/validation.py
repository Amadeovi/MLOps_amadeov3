"""Validate customer features before inference."""

import numpy as np
from pandas.api.types import is_numeric_dtype


def validate_features(features, expected_columns):
    """Validate numeric features and arrange them in training order.

    Args:
        features: DataFrame containing customer features.
        expected_columns: Feature names in the order used during training.

    Returns:
        A copy of the validated DataFrame in training feature order.

    Raises:
        ValueError: If features are empty or contain invalid columns or values.
    """
    if features.empty:
        raise ValueError("Provide at least one customer.")

    if not features.columns.is_unique:
        raise ValueError("Duplicate feature names are not allowed.")

    missing = sorted(set(expected_columns) - set(features.columns))
    unexpected = sorted(set(features.columns) - set(expected_columns))

    if missing or unexpected:
        raise ValueError(f"Invalid features. Missing: {missing}; unexpected: {unexpected}.")

    ordered = features.loc[:, expected_columns].copy()

    null_columns = ordered.columns[ordered.isna().any()].tolist()
    if null_columns:
        raise ValueError(f"Missing values in columns: {null_columns}.")

    non_numeric = [column for column in ordered.columns if not is_numeric_dtype(ordered[column])]
    if non_numeric:
        raise ValueError(f"Non-numeric values in columns: {non_numeric}.")

    if not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise ValueError("Feature values must be finite.")

    return ordered
