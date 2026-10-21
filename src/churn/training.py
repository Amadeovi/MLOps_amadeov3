"""Train machine learning models for customer churn prediction."""

from loguru import logger

from .evaluation import get_metrics


def train_model(model, X_train, y_train, X_valid, y_valid):
    """Fit a fresh estimator and return it with train/validation metrics."""
    model.fit(X_train, y_train)

    metrics_train = get_metrics(
        y_train,
        model.predict(X_train),
        model.predict_proba(X_train)[:, 1],
    )
    logger.info(f"Metrics train set: {metrics_train}")

    metrics_valid = get_metrics(
        y_valid,
        model.predict(X_valid),
        model.predict_proba(X_valid)[:, 1],
    )
    logger.info(f"Metrics validation set: {metrics_valid}")

    return model, metrics_train, metrics_valid
