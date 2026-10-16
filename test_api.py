
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200


def test_prediction_endpoint():
    payload = {
        "cpu_usage": 92,
        "memory_usage": 88,
        "disk_usage": 90,
        "request_rate": 850,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() is not None


def test_prediction_rejects_invalid_cpu():
    payload = {
        "cpu_usage": 150,
        "memory_usage": 88,
        "disk_usage": 90,
        "request_rate": 850,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
