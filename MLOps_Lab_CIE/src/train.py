import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import json
import os
import numpy as np

# Load data
data = pd.read_csv("data/training_data.csv")

X = data.drop("bug_detection_count", axis=1)
y = data["bug_detection_count"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("reviewbot-bug-detection-count")

results = []

models = {
    "LinearRegression": LinearRegression(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

best_mae = float("inf")
best_model_name = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("project_phase", "model_selection")

        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

        if mae < best_mae:
            best_mae = mae
            best_model_name = name

# Save JSON
output = {
    "experiment_name": "reviewbot-bug-detection-count",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "mae",
    "best_metric_value": best_mae
}

os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed")


# ---------------- TASK 2 ----------------
# Save best model separately

import joblib

os.makedirs("models", exist_ok=True)

if best_model_name == "LinearRegression":
    joblib.dump(models["LinearRegression"], "models/best_model.pkl")
else:
    joblib.dump(models["GradientBoosting"], "models/best_model.pkl")

print("Best model saved successfully")