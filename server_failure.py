import pandas as pd

# Synthetic server monitoring data
data = {
    "cpu_usage": [25, 35, 45, 55, 65, 75, 85, 95, 90, 30,
                  40, 60, 80, 92, 50, 70, 88, 20, 98, 62],
    "memory_usage": [30, 40, 45, 50, 60, 65, 80, 95, 88, 35,
                     45, 55, 75, 92, 48, 70, 85, 25, 96, 58],
    "disk_usage": [20, 30, 40, 45, 55, 60, 75, 90, 85, 25,
                   35, 50, 70, 88, 42, 65, 80, 15, 95, 52],
    "request_rate": [100, 150, 200, 250, 300, 350, 500, 900, 750, 120,
                     180, 280, 450, 850, 230, 400, 700, 80, 1000, 320],
    "server_failure": [0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                       0, 0, 1, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

print("First five rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nFailure distribution:")
print(df["server_failure"].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Define features and target
X = df[[
    "cpu_usage",
    "memory_usage",
    "disk_usage",
    "request_rate"
]]

y = df["server_failure"]

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("\nTraining examples:", len(X_train))
print("Test examples:", len(X_test))

# Create and train the model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Predict test outcomes
y_pred = model.predict(X_test)

# Evaluate the model
print("\nAccuracy:", f"{accuracy_score(y_test, y_pred):.2%}")
print("Precision:", f"{precision_score(y_test, y_pred, zero_division=0):.2%}")
print("Recall:", f"{recall_score(y_test, y_pred, zero_division=0):.2%}")
print("F1-score:", f"{f1_score(y_test, y_pred, zero_division=0):.2%}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Failure", "Failure"],
    zero_division=0
))

import joblib

# Save the trained model
joblib.dump(model, "server_failure_model.joblib")

print("\nModel saved successfully.")

# Load the saved model
loaded_model = joblib.load("server_failure_model.joblib")

# Predict using the loaded model
sample_server = [[92, 88, 90, 850]]

prediction = loaded_model.predict(sample_server)[0]

if prediction == 1:
    print("Prediction: Server failure risk detected")
else:
    print("Prediction: No server failure predicted")