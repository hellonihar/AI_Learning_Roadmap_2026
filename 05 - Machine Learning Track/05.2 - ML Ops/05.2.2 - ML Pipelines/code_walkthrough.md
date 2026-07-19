# Code Walkthrough: ML Pipeline DAG with Prefect

This walkthrough defines a complete ML pipeline as a DAG of functions using Prefect. Each step has explicit inputs/outputs, and the pipeline can be run locally or deployed to a Prefect server.

## Setup

```bash
pip install prefect pandas scikit-learn
```

## Pipeline Overview

```
raw_data ──► featurize ──► train ──► evaluate ──► register_model
```

## Full Code

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import json
import os
from pathlib import Path

from prefect import flow, task
from prefect.cache_policies import INPUTS
from prefect.artifacts import create_table_artifact


# ---------------------------------------------------------------------------
# Step 1: Ingest
# ---------------------------------------------------------------------------
@task
def ingest_data(raw_path: str) -> pd.DataFrame:
    """Load raw CSV data. Fail fast if file is missing."""
    if not Path(raw_path).exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"Ingested {len(df)} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Step 2: Featurize
# ---------------------------------------------------------------------------
@task(cache_policy=INPUTS)
def featurize(df: pd.DataFrame, target_col: str, test_size: float = 0.2):
    """Encode categoricals, split features/target, train/test split."""
    df_clean = df.dropna().copy()

    # Encode categorical columns
    le_dict = {}
    for col in df_clean.select_dtypes(include=["object"]).columns:
        if col != target_col:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            le_dict[col] = le

    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])

    # Encode target if needed
    if y.dtype == object:
        target_le = LabelEncoder()
        y = target_le.fit_transform(y)
        le_dict[target_col] = target_le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(
        f"Featurized: {X_train.shape[0]} train, {X_test.shape[0]} test, "
        f"{X_train.shape[1]} features"
    )
    return X_train, X_test, y_train, y_test, le_dict


# ---------------------------------------------------------------------------
# Step 3: Train
# ---------------------------------------------------------------------------
@task(cache_policy=INPUTS)
def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 10,
) -> RandomForestClassifier:
    """Train a RandomForest and return the model object."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"Trained model: {type(model).__name__}")
    return model


# ---------------------------------------------------------------------------
# Step 4: Evaluate
# ---------------------------------------------------------------------------
@task
def evaluate(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Compute classification metrics and log them as a Prefect artifact."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(
            precision_score(y_test, y_pred, average="weighted"), 4
        ),
        "recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
    }

    create_table_artifact(
        key="evaluation-metrics",
        table=[
            {"metric": k, "value": v}
            for k, v in metrics.items()
        ],
        description="Model evaluation results",
    )

    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    return metrics


# ---------------------------------------------------------------------------
# Step 5: Register
# ---------------------------------------------------------------------------
@task
def register_model(
    model: RandomForestClassifier,
    metrics: dict,
    model_dir: str = "models",
) -> str:
    """Save model to disk and return path. In production, push to registry."""
    os.makedirs(model_dir, exist_ok=True)
    import joblib

    model_path = os.path.join(model_dir, f"model.pkl")
    joblib.dump(model, model_path)

    # Also save metadata
    meta = {"path": model_path, "metrics": metrics}
    meta_path = os.path.join(model_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved to {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Pipeline DAG
# ---------------------------------------------------------------------------
@flow(
    name="ml-pipeline",
    log_prints=True,
)
def ml_pipeline(
    raw_path: str = "data/raw.csv",
    target_col: str = "target",
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 10,
):
    """End-to-end ML pipeline.

    DAG: ingest → featurize → train → evaluate → register
    """
    raw = ingest_data(raw_path)
    X_train, X_test, y_train, y_test, _ = featurize(
        raw, target_col, test_size
    )
    model = train(X_train, y_train, n_estimators, max_depth)
    metrics = evaluate(model, X_test, y_test)
    model_path = register_model(model, metrics)

    print(f"\nPipeline complete. Model at: {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run locally — Prefect handles caching and retries automatically
    ml_pipeline(
        raw_path="data/raw.csv",
        target_col="species",
        n_estimators=200,
    )
```

## Running the Pipeline

```bash
# Create sample data
mkdir -p data
python -c "
import pandas as pd
import numpy as np
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'feature1': np.random.randn(n),
    'feature2': np.random.randn(n),
    'feature3': np.random.randn(n),
    'species': np.random.choice(['setosa', 'versicolor', 'virginica'], n)
})
df.to_csv('data/raw.csv', index=False)
"

# Run the pipeline
python pipeline.py
```

## Key Takeaways

- **Each step is a `@task`**: Prefect tracks the state, timing, and outputs.
- **`cache_policy=INPUTS`**: If inputs match a previous run, the cached output is reused.
- **Artifacts**: `create_table_artifact` logs evaluation metrics for the dashboard.
- **Parameters**: Function arguments at the `@flow` level become configurable run parameters.
- **Extensibility**: Swap `RandomForestClassifier` with any sklearn-compatible model by changing one function.

## Deploying to Prefect Server

```bash
prefect deploy pipeline.py:ml_pipeline --name production --cron "0 6 * * 1"
```

This schedules the pipeline to run every Monday at 6 AM with full observability.
