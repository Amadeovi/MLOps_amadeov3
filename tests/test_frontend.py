"""Test the Gradio prediction callbacks."""

import gradio as gr
import pandas as pd
import pytest
import requests

from churn import frontend


def test_predict_customer_sends_json_and_returns_prediction(mocker):
    """Send parsed customer features and return the API response."""
    prediction = {"churn_prediction": 1, "churn_probability": 0.8}
    post = mocker.patch("churn.frontend.requests.post")
    post.return_value.ok = True
    post.return_value.json.return_value = prediction

    result = frontend.predict_customer('{"call_failure": 7}')

    post.assert_called_once_with(
        "http://127.0.0.1:8000/predict",
        json={"call_failure": 7},
        timeout=30,
    )
    assert result == prediction


def test_predict_customer_rejects_invalid_json(mocker):
    """Reject malformed JSON before calling the API."""
    post = mocker.patch("churn.frontend.requests.post")

    with pytest.raises(gr.Error, match="Enter valid customer JSON"):
        frontend.predict_customer("{broken JSON")

    post.assert_not_called()


def test_predict_customer_handles_unreachable_api(mocker):
    """Show a useful message when the API is unreachable."""
    mocker.patch(
        "churn.frontend.requests.post",
        side_effect=requests.ConnectionError("Connection refused"),
    )

    with pytest.raises(gr.Error, match="Could not reach the prediction API"):
        frontend.predict_customer('{"call_failure": 7}')


def test_predict_batch_preserves_order_and_sends_missing_values_as_null(tmp_path, mocker):
    """Send CSV rows as JSON and attach predictions in the original order."""
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text("call_failure,age\n7,31\n2,\n", encoding="utf-8")

    post = mocker.patch("churn.frontend.requests.post")
    post.return_value.ok = True
    post.return_value.json.return_value = [
        {"churn_prediction": 1, "churn_probability": 0.8},
        {"churn_prediction": 0, "churn_probability": 0.2},
    ]

    result = frontend.predict_batch(str(csv_path))

    post.assert_called_once_with(
        "http://127.0.0.1:8000/predict/batch",
        json=[
            {"call_failure": 7, "age": 31.0},
            {"call_failure": 2, "age": None},
        ],
        timeout=30,
    )
    expected = pd.DataFrame(
        {
            "call_failure": [7, 2],
            "age": [31.0, float("nan")],
            "churn_prediction": [1, 0],
            "churn_probability": [0.8, 0.2],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_predict_batch_reports_api_rejection(tmp_path, mocker):
    """Show the API's error when it rejects a batch."""
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text("call_failure,age\n7,\n", encoding="utf-8")

    post = mocker.patch("churn.frontend.requests.post")
    post.return_value.ok = False
    post.return_value.status_code = 422
    post.return_value.text = '{"detail":"age is required"}'

    with pytest.raises(gr.Error, match="age is required"):
        frontend.predict_batch(str(csv_path))


def test_predict_batch_rejects_wrong_prediction_count(tmp_path, mocker):
    """Reject a response that cannot be matched to every input row."""
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text("call_failure\n7\n2\n", encoding="utf-8")

    post = mocker.patch("churn.frontend.requests.post")
    post.return_value.ok = True
    post.return_value.json.return_value = [
        {"churn_prediction": 1, "churn_probability": 0.8},
    ]

    with pytest.raises(gr.Error, match="unexpected number of predictions"):
        frontend.predict_batch(str(csv_path))


def test_predict_batch_rejects_csv_without_rows(tmp_path, mocker):
    """Reject a header-only CSV before calling the API."""
    csv_path = tmp_path / "customers.csv"
    csv_path.write_text("call_failure,age\n", encoding="utf-8")
    post = mocker.patch("churn.frontend.requests.post")

    with pytest.raises(gr.Error, match="must contain at least one customer"):
        frontend.predict_batch(str(csv_path))

    post.assert_not_called()
