"""Run the customer churn model training pipeline."""

import os

import click
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.model_selection import train_test_split

from churn.preprocess import preprocess_data
from churn.tracking import track_experiment
from churn.training import train_model


@click.command()
@click.option("--data-path", default="data/iranian_churn/iranian_churn.parquet", help="Path to data.")
@click.option("--random-seed", default=42, help="Seed used for reproducibility.")
@click.option("--valid-size", default=0.33, help="Fraction of data reserved for validation.")
@click.option("--experiment-name", default="iranian-churn-experiment", help="MLflow experiment name.")
@click.option("--run-name", default="lightgbm_test", help="Name of the model logged in MLflow.")
def main(data_path, random_seed, valid_size, experiment_name, run_name):
    """Split the data and run the tracked LightGBM experiment."""
    logger.info("Reading & Preprocessing the data")
    df = pd.read_parquet(data_path)
    preprocessed_df = preprocess_data(df)
    X = preprocessed_df.drop(columns=["churn"])
    y = preprocessed_df["churn"]

    logger.info("Training the model:")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=valid_size,
        random_state=random_seed,
        stratify=y,
    )

    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=500,
        learning_rate=0.1,
        num_leaves=5,
        min_child_samples=15,
        reg_lambda=1.0,
        random_state=random_seed,
        verbosity=-1,
    )
    fitted_model, metrics_train, metrics_valid = train_model(model, X_train, y_train, X_valid, y_valid)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info("Logging results in MLflow")
    results = track_experiment(fitted_model, run_name, metrics_train, metrics_valid, random_seed, valid_size)

    logger.info(results)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
