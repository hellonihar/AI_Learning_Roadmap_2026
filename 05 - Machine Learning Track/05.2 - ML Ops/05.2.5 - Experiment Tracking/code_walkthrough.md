# Code Walkthrough: MLflow Experiment Tracking

This walkthrough demonstrates logging parameters, metrics, and artifacts with MLflow, comparing runs, and registering a model.

## Setup

```bash
pip install mlflow scikit-learn pandas
```

## Step 1: Start the MLflow Tracking Server

```bash
# Local UI server (SQLite backend)
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000
```

Open http://localhost:5000 in your browser.

## Step 2: Full Training Script with Tracking

```python
# train.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import json
from datetime import datetime


# ---------------------------------------------------------------------------
# Configuration — log these as parameters
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "iris-classification"
TRACKING_URI = "http://localhost:5000"

# Models to compare
MODELS = {
    "LogisticRegression": LogisticRegression(C=1.0, max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1
    ),
}

# Hyperparameters to vary in each run
HYPERPARAMS = [
    {"C": 0.1, "max_iter": 200},
    {"C": 1.0, "max_iter": 200},
    {"C": 10.0, "max_iter": 500},
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_data() -> tuple:
    """Load Iris dataset. Log the dataset hash as a parameter."""
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


def plot_confusion_matrix(y_true, y_pred, labels, save_path: str):
    """Generate and save a confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------------
# Training function (one MLflow run per call)
# ---------------------------------------------------------------------------
def train_and_log(
    model_name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    hyperparams: dict | None = None,
):
    """Train a model and log everything to MLflow."""

    with mlflow.start_run(run_name=f"{model_name}-{datetime.now():%H%M%S}"):
        # --- Log parameters ---
        mlflow.log_param("model_name", model_name)

        # Log hyperparameters from the model
        params = model.get_params()
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log extra hyperparams if provided
        if hyperparams:
            for key, value in hyperparams.items():
                mlflow.log_param(f"extra_{key}", value)

        # Log dataset info
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("feature_count", X_train.shape[1])
        mlflow.log_param("n_classes", len(np.unique(y_train)))

        # --- Train ---
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Log metrics ---
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # --- Log artifacts ---
        # 1. Confusion matrix plot
        target_names = load_iris().target_names.tolist()
        cm_path = "confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, target_names, cm_path)
        mlflow.log_artifact(cm_path)

        # 2. Metrics as JSON
        metrics_path = "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)

        # 3. Model itself (MLflow format)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=(
                model_name if metrics["accuracy"] > 0.9 else None
            ),
        )

        # --- Tag the run ---
        mlflow.set_tag("model_family", model_name.split("_")[0])
        mlflow.set_tag("accuracy_tier",
                       "high" if metrics["accuracy"] > 0.9 else "medium")

        print(
            f"Run: {mlflow.active_run().info.run_id} | "
            f"Model: {model_name} | "
            f"Accuracy: {metrics['accuracy']:.4f}"
        )


# ---------------------------------------------------------------------------
# Main: run multiple experiments
# ---------------------------------------------------------------------------
def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Run each model with default params
    for name, model in MODELS.items():
        train_and_log(name, model, X_train, X_test, y_train, y_test)

    # Run LogisticRegression with different hyperparameters
    for params in HYPERPARAMS:
        model = LogisticRegression(**params)
        train_and_log(
            "LogisticRegression_tuned",
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            hyperparams=params,
        )

    print("\nAll runs complete. Open http://localhost:5000 to compare.")


if __name__ == "__main__":
    main()
```

## Step 3: Run and Compare

```bash
python train.py
```

### Via the UI

1. Go to http://localhost:5000.
2. Click the **iris-classification** experiment.
3. Select multiple runs and click **Compare**.
4. View parallel coordinates, scatter plots, and metric tables.

### Via the API

```python
# compare_runs.py
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

experiment = mlflow.get_experiment_by_name("iris-classification")
df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Show runs sorted by accuracy
df_sorted = df.sort_values("metrics.accuracy", ascending=False)
print(df_sorted[
    ["run_id", "params.model_name", "metrics.accuracy", "metrics.f1_macro"]
].to_string())
```

Output:

```
                              run_id   params.model_name  metrics.accuracy  metrics.f1_macro
8f3a...  GradientBoosting                   1.0000            1.0000
2b1c...  RandomForest                       1.0000            1.0000
4e7a...  LogisticRegression_tuned           1.0000            1.0000
a0d2...  LogisticRegression                 0.9667            0.9656
```

## Step 4: Register a Model

From the UI, promote the best run to the Model Registry:

1. Click the best run.
2. Under **Artifacts** → **model**, click **Register Model**.
3. Name it `iris-classifier`.
4. Set stage to **Staging** (for testing), then promote to **Production**.

Or via API:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
result = mlflow.register_model(
    model_uri="runs:/<RUN_ID>/model",
    name="iris-classifier"
)
client.transition_model_version_stage(
    name="iris-classifier",
    version=result.version,
    stage="Production",
)
```

## Step 5: Load a Production Model

```python
# serve.py
import mlflow

MODEL_URI = "models:/iris-classifier/Production"
model = mlflow.sklearn.load_model(MODEL_URI)

sample = [[5.1, 3.5, 1.4, 0.2]]
pred = model.predict(sample)
print(f"Prediction: {pred}")
```

## Key Takeaways

- **Every run is self-documenting**: params + metrics + artifacts + code version.
- **Compare across runs**: identify which hyperparams drive performance.
- **Model Registry**: bridge between experimentation and production.
- **Reproducibility**: given the run ID, you can reconstruct the exact model.
