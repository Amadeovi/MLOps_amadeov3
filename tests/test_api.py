"""Tests for the churn prediction API."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from churn.api import app


@pytest.fixture
def client(mocker):
    """Create an API client using a fake prediction model."""
    model = mocker.Mock()

    # Must match the fields expected by your inference validation.
    model.feature_name_ = [
        "call_failure",
        "complains",
        "subscription_length",
        "charge_amount",
        "seconds_of_use",
        "frequency_of_sms",
        "distinct_called_numbers",
        "tariff_plan",
        "status",
        "age",
        "customer_value",
    ]

    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.2, 0.8]])

    mocker.patch("churn.api.load_model", return_value=model)

    with TestClient(app) as test_client:
        yield test_client


def test_predict_returns_prediction(client):
    """Return a churn prediction for a valid customer."""
    customer = {
        "call_failure": 7,
        "complains": 0,
        "subscription_length": 32,
        "charge_amount": 1,
        "seconds_of_use": 4472,
        "frequency_of_sms": 73,
        "distinct_called_numbers": 24,
        "tariff_plan": 1,
        "status": 1,
        "age": 31,
        "customer_value": 471.0,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200
    assert response.json() == {
        "churn_prediction": 1,
        "churn_probability": 0.8,
    }


def test_predict_rejects_missing_fields(client):
    """Reject a request missing required customer features."""
    response = client.post("/predict", json={"age": 31})

    assert response.status_code == 422

    errors = response.json()["detail"]
    assert any(error["loc"] == ["body", "call_failure"] and error["type"] == "missing" for error in errors)
