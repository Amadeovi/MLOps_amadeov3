"""Predict customer churn from a CSV file."""

import os

import click
import mlflow
import pandas as pd
from loguru import logger

from churn.inference import load_model, predict_customers
from churn.preprocess import preprocess_data


@click.command()
@click.option("--customer-sample", default="data/customer_sample.csv", help="CSV file containing customer features.")
def main(customer_sample):
    """Load customer records and output churn predictions.

    Args:
        customer_sample: Path to the customer CSV file.
    """
    model_uri = os.getenv("MODEL_URI")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    mlflow.set_tracking_uri(tracking_uri)

    logger.info("Loading customer sample:")
    customer_features = preprocess_data(pd.read_csv(customer_sample))

    logger.info("Loading desired model and making predictions:")
    model = load_model(model_uri)
    results = predict_customers(model, customer_features)

    logger.info(results)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
