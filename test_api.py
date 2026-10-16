
import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient


def create_test_app():
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([1])

    fake_champion = MagicMock()
    fake_champion.version = "1"
    fake_champion.source = "models:/m-test123"

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.MlflowClient") as mock_client_class,
        patch("mlflow.sklearn.load_model", return_value=fake_model),
    ):
        mock_client_class.return_value.get_model_version_by_alias.return_value = (
            fake_champion
        )

        # Import api only after mocking MLflow initialization.
        sys.modules.pop("api", None)
        api = importlib.import_module("api")

    return api, fake_model


def test_health_endpoint():
    api, _ = create_test_app()
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    api, fake_model = create_test_app()
    client = TestClient(api.app)

    payload = {
        "cpu_usage": 92,
        "memory_usage": 88,
        "disk_usage": 90,
        "request_rate": 850,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    fake_model.predict.assert_called_once()


def test_prediction_rejects_invalid_cpu():
    api, _ = create_test_app()
    client = TestClient(api.app)

    payload = {
        "cpu_usage": 150,
        "memory_usage": 88,
        "disk_usage": 90,
        "request_rate": 850,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
