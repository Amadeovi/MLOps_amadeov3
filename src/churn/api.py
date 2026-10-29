"""FASTAPI SERVING APPLICATION - Production-Ready ML Model Serving.

========================================================================

This application provides a complete serving solution for the Iranian Telecom Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation.
- Pydantic: Data validation and automatic API documentation.
"""

from contextlib import asynccontextmanager
from typing import Literal

import mlflow
import pandas as pd
from fastapi import FastAPI, Request
from loguru import logger
from pydantic import BaseModel, Field

from churn.inference import load_model, predict_customers

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_URI = "models:/m-9f24f420806c464997cd23870e4bc0af"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup and release it at shutdown."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    logger.info("Loading churn model")
    app.state.model = load_model(MODEL_URI)
    logger.info("Churn model loaded successfully")

    try:
        yield
    finally:
        del app.state.model


app = FastAPI(
    title="Iranian Telecom Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Health check endpoint for monitoring and load balancer health checks."""
    return {"status": "ok"}


# Pydantic model for automatic valiation and API documenation
class Customer(BaseModel):
    """Customer features required for churn prediction."""

    call_failure: float = Field(examples=[7])
    complains: Literal[0, 1] = Field(examples=[0])
    subscription_length: float = Field(examples=[32])
    charge_amount: float = Field(examples=[0])
    seconds_of_use: float = Field(examples=[4472])
    frequency_of_sms: float = Field(examples=[73])
    distinct_called_numbers: float = Field(examples=[23])
    tariff_plan: float = Field(examples=[1])
    status: float = Field(examples=[1])
    age: float = Field(examples=[31])
    customer_value: float = Field(examples=[471])


class PredictionResponse(BaseModel):
    """Churn prediction returned by the API."""

    churn_prediction: Literal[0, 1]
    churn_probability: float = Field(ge=0, le=1)


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: Customer, request: Request):
    """Predict churn and its probability for one customer."""
    features = pd.DataFrame([customer.model_dump()])

    model = request.app.state.model
    results = predict_customers(model, features)
    result = results.iloc[0]

    return {
        "churn_prediction": int(result["churn_prediction"]),
        "churn_probability": float(result["churn_probability"]),
    }
