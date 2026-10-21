"""Log results of an experiment."""

import mlflow
import mlflow.sklearn


def track_experiment(model, run_name, metrics_train, metrics_valid, random_seed, test_size):
    """Logging results of a fitted model.

    Args:
        model: Fitted machine learning model.
        run_name: Name of the MLflow run.
        metrics_train: Dictionary of metric names and values for the training set.
        metrics_valid: Dictionary of metric names and values for the validation set.
        random_seed: Random seed used for reproducibility.
        test_size: Proportion of data reserved for evaluation.

    Returns:
        dict: Run ID, model URI, training metrics, and validation metrics.
    """
    with mlflow.start_run(run_name=run_name) as run:
        # Log estimator and pipeline parameters.
        # Strings also accommodate settings such as hidden_layer_sizes=(64, 32).
        params = {
            key: str(value)
            for key, value in model.get_params(deep=True).items()
            if key != "steps" and not hasattr(value, "get_params")
        }

        mlflow.log_params(params)
        mlflow.log_params({"split_seed": random_seed, "test_size": test_size})

        for split, metrics in [("train", metrics_train), ("valid", metrics_valid)]:
            mlflow.log_metrics({f"{split}_{name}": float(value) for name, value in metrics.items()})

        model_info = mlflow.sklearn.log_model(model, name=run_name, serialization_format="cloudpickle")

        return {
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "train_metrics": metrics_train,
            "valid_metrics": metrics_valid,
        }
