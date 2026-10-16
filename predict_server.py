
import mlflow
import mlflow.sklearn
import pandas as pd

# Connect to the same MLflow tracking database
mlflow.set_tracking_uri(
    "sqlite:///C:/Users/Admin/OneDrive/Desktop/AI/mlflow.db"
)

# Load a specific registered model version
model = mlflow.sklearn.load_model(
    "models:/server-failure-model/1"
)

# Example server metrics
server_metrics = pd.DataFrame(
    [{
        "cpu_usage": 92,
        "memory_usage": 88,
        "disk_usage": 90,
        "request_rate": 850
    }]
)

# Predict server failure risk
prediction = model.predict(server_metrics)[0]

if prediction == 1:
    print("Server failure risk detected!")
else:
    print("Server is operating normally.")
