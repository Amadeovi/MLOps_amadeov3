"""Load saved models and generate customer churn predictions."""

import mlflow.sklearn
import pandas as pd
from loguru import logger

from churn.validation import validate_features


def load_model(model_uri):
    """Load a model saved with MLflow's sklearn flavor.

    Args:
        model_uri: URI identifying the saved model.

    Returns:
        The loaded estimator or pipeline.
    """
    return mlflow.sklearn.load_model(model_uri)


def predict_customers(model, features):
    """Predict churn labels and probabilities.

    Args:
        model: Fitted binary classifier with classes ordered as [0, 1].
        features: DataFrame matching the training feature schema and order.

    Returns:
        DataFrame containing churn predictions and probabilities.
    """
    validated_features = validate_features(
        features,
        expected_columns=list(model.feature_name_),
    )
    logger.info(f"Validated {len(validated_features)} customer records")
    logger.info(f"Number of customer features {len(validated_features.columns)}")

    return pd.DataFrame(
        {
            "churn_prediction": model.predict(validated_features),
            "churn_probability": model.predict_proba(validated_features)[:, 1],
        },
        index=validated_features.index,
    )
