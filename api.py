
import os
import logging
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_DATA_DIR = os.getenv("MLFLOW_DATA_DIR", "/mlflow-data")
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:////mlflow-data/mlflow.db",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# The model is unavailable until startup loading succeeds.
model = None
champion = None


def load_model():
    global model, champion

    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        "server-failure-model",
        "champion",
    )

    model_id = champion.source.split("/")[-1]

    model_path = os.path.join(
        MLFLOW_DATA_DIR,
        "mlruns",
        "1",
        "models",
        model_id,
        "artifacts",
    )

    model = mlflow.sklearn.load_model(model_path)

    logger.info(
        "Loaded server-failure-model version %s",
        champion.version,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except Exception:
        # Keep the API alive so health endpoints can report the problem.
        logger.exception("ML model loading failed")

    yield


app = FastAPI(
    title="Server Failure Prediction API",
    description="Predict server failure using an MLflow model",
    version="1.0.0",
    lifespan=lifespan,
)


class ServerMetrics(BaseModel):
    cpu_usage: float = Field(ge=0, le=100)
    memory_usage: float = Field(ge=0, le=100)
    disk_usage: float = Field(ge=0, le=100)
    request_rate: float = Field(ge=0)


@app.get("/")
def root():
    return {"message": "Server Failure Prediction API is running"}


@app.get("/health/live")
def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
def readiness():
    if model is None or champion is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not ready",
        )

    return {
        "status": "ready",
        "model_name": "server-failure-model",
        "model_version": champion.version,
    }


# Keep the original health endpoint for compatibility.
@app.get("/health")
def health():
    return readiness()


@app.post("/predict")
def predict(metrics: ServerMetrics):
    if model is None or champion is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not ready",
        )

    input_data = pd.DataFrame(
        [
            {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "disk_usage": metrics.disk_usage,
                "request_rate": metrics.request_rate,
            }
        ]
    )

    prediction = model.predict(input_data)[0]

    return {
        "prediction": int(prediction),
        "result": (
            "Server Failure"
            if int(prediction) == 1
            else "No Server Failure"
        ),
        "model_name": "server-failure-model",
        "model_version": champion.version,
    }
