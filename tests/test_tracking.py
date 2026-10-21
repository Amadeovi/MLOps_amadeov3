"""Tests for MLflow experiment logging."""

from types import SimpleNamespace

from churn.tracking import track_experiment


def test_track_experiment_logs_parameters_metrics_and_model(mocker):
    """Log supported parameters, split metrics, and the fitted model."""
    model = mocker.Mock()
    nested_estimator = mocker.Mock()
    model.get_params.return_value = {
        "steps": [("classifier", nested_estimator)],
        "classifier": nested_estimator,
        "classifier__max_depth": 3,
        "classifier__hidden_layer_sizes": (64, 32),
        "classifier__class_weight": None,
    }

    start_run = mocker.patch("churn.tracking.mlflow.start_run")
    start_run.return_value.__enter__.return_value = SimpleNamespace(info=SimpleNamespace(run_id="test-run-id"))

    log_params = mocker.patch("churn.tracking.mlflow.log_params")
    log_metrics = mocker.patch("churn.tracking.mlflow.log_metrics")
    log_model = mocker.patch(
        "churn.tracking.mlflow.sklearn.log_model",
        return_value=SimpleNamespace(model_uri="models:/test-model"),
    )

    metrics_train = {"accuracy": 0.95, "auc": 0.98}
    metrics_valid = {"accuracy": 0.85, "auc": 0.90}

    result = track_experiment(
        model=model,
        run_name="test-experiment",
        metrics_train=metrics_train,
        metrics_valid=metrics_valid,
        random_seed=42,
        test_size=0.2,
    )

    start_run.assert_called_once_with(run_name="test-experiment")
    model.get_params.assert_called_once_with(deep=True)

    # Exclude pipeline steps and nested estimators; stringify scalar settings.
    assert log_params.call_args_list == [
        mocker.call(
            {
                "classifier__max_depth": "3",
                "classifier__hidden_layer_sizes": "(64, 32)",
                "classifier__class_weight": "None",
            }
        ),
        mocker.call({"split_seed": 42, "test_size": 0.2}),
    ]

    assert log_metrics.call_args_list == [
        mocker.call({"train_accuracy": 0.95, "train_auc": 0.98}),
        mocker.call({"valid_accuracy": 0.85, "valid_auc": 0.90}),
    ]

    log_model.assert_called_once_with(
        model,
        name="test-experiment",
        serialization_format="cloudpickle",
    )

    assert result == {
        "run_id": "test-run-id",
        "model_uri": "models:/test-model",
        "train_metrics": metrics_train,
        "valid_metrics": metrics_valid,
    }

    start_run.return_value.__exit__.assert_called_once_with(None, None, None)
