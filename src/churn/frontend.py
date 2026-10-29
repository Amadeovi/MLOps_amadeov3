"""Provide a Gradio interface for the churn prediction API."""

import json

import gradio as gr
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


if __name__ == "__main__":
    demo.launch()
