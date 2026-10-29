"""Provide a Gradio interface for the churn prediction API."""

import json

import gradio as gr
import pandas as pd
import requests


def predict_customer(customer_json):
    """Send customer features to the prediction API.

    Args:
        customer_json: Customer features encoded as JSON.

    Returns:
        The prediction response or an error message.
    """
    try:
        customer = json.loads(customer_json)
    except json.JSONDecodeError as exc:
        raise gr.Error("Enter valid customer JSON.") from exc

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=customer,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise gr.Error("Could not reach the prediction API.") from exc

    if not response.ok:
        raise gr.Error(f"API rejected the request: {response.text}")

    return response.json()


def predict_batch(csv_path):
    """Request churn predictions for customers in a CSV file.

    Args:
        csv_path: Path to the uploaded CSV file.

    Returns:
        Customer records with their churn predictions and probabilities.
    """
    if csv_path is None:
        raise gr.Error("Upload a CSV file first.")

    try:
        customers = pd.read_csv(csv_path)
    except (ValueError, OSError) as exc:
        raise gr.Error("Could not read the CSV file.") from exc

    if customers.empty:
        raise gr.Error("The CSV must contain at least one customer.")

    # Convert missing values to JSON null instead of invalid JSON NaN.
    payload = json.loads(customers.to_json(orient="records"))

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict/batch",
            json=payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise gr.Error("Could not reach the prediction API.") from exc

    if not response.ok:
        raise gr.Error(f"Prediction request failed: {response.text}")

    predictions = pd.DataFrame(response.json())

    if len(predictions) != len(customers):
        raise gr.Error("The API returned an unexpected number of predictions.")

    return pd.concat(
        [customers.reset_index(drop=True), predictions],
        axis=1,
    )


demo = gr.Interface(
    fn=predict_customer,
    inputs=gr.Textbox(
        label="Customer features as JSON",
        lines=15,
    ),
    outputs=gr.JSON(label="Churn prediction"),
    title="Customer Churn Prediction",
    description="Submit customer features to the FastAPI prediction service.",
)


batch_demo = gr.Interface(
    fn=predict_batch,
    inputs=gr.File(
        label="Upload customer CSV",
        file_types=[".csv"],
        type="filepath",
    ),
    outputs=gr.Dataframe(
        label="Customer predictions",
        interactive=False,
    ),
    title="Batch Churn Prediction",
    description="Upload one row per customer to predict churn.",
)


if __name__ == "__main__":
    batch_demo.launch()
