
import os
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

# Use environment variables so the same API works on Windows and Docker.
MLFLOW_DATA_DIR = os.getenv(
    "MLFLOW_DATA_DIR",
    "/mlflow-data"
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:////mlflow-data/mlflow.db"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = MlflowClient()

champion = client.get_model_version_by_alias(
    "server-failure-model",
    "champion"
)

model_id = champion.source.split("/")[-1]

model_path = os.path.join(
    MLFLOW_DATA_DIR,
    "mlruns",
    "1",
    "models",
    model_id,
    "artifacts"
)

model = mlflow.sklearn.load_model(model_path)

app = FastAPI(
    title="Server Failure Prediction API",
    description="Predict server failure using an MLflow model",
    version="1.0.0",
)


class ServerMetrics(BaseModel):
    cpu_usage: float = Field(ge=0, le=100)
    memory_usage: float = Field(ge=0, le=100)
    disk_usage: float = Field(ge=0, le=100)
    request_rate: float = Field(ge=0)


@app.get("/")
def root():
    return {"message": "Server Failure Prediction API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_name": "server-failure-model",
        "model_version": champion.version,
    }


@app.post("/predict")
def predict(metrics: ServerMetrics):
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

