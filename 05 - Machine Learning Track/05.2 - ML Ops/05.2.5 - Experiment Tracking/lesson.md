# 05.2.5 Experiment Tracking

## Why Track Experiments?

Machine learning is inherently iterative: you try different models, hyperparameters, feature sets, and data preprocessing strategies. Without systematic tracking, teams fall into these traps:

- **Lost history**: "I got 94% accuracy last week — what combination produced that?"
- **Dead-end repetition**: Running the same experiment twice because nobody logged it.
- **No lineage**: Which training data, code commit, and parameters produced model v2.3?
- **Difficult collaboration**: A team member's experiment lives in a notebook on their machine.

Experiment tracking solves all of these by logging every run's parameters, metrics, artifacts, and environment to a central store.

## Key Concepts

| Concept     | Description                                   | Example                                          |
| ----------- | --------------------------------------------- | ------------------------------------------------ |
| **Run**     | A single execution of a training script       | `exp_2024_03_15_001`                             |
| **Experiment** | A logical group of related runs            | `bert_finetuning`, `xgboost_hyperopt`            |
| **Parameter** | Input configuration to the run             | `learning_rate=0.001`, `n_estimators=100`        |
| **Metric**  | Numeric evaluation result                     | `accuracy=0.942`, `loss=0.23`                    |
| **Artifact** | File or directory output from the run        | `model.pkl`, `confusion_matrix.png`, `vocab.txt` |
| **Tag**     | Key-value metadata for filtering              | `{"env": "gpu", "dataset": "v3"}`                |
| **Registry**| Curated, versioned production-ready models   | `model:v1.0` (promoted from run)                 |

## Tools Comparison

### MLflow

The most widely adopted open-source experiment tracker. Four components:

1. **Tracking Server** — logs runs to a SQLite/PostgreSQL/MySQL backend, available via UI and API.
2. **Model Registry** — manage model versions, stages (Staging → Production → Archived), and metadata.
3. **Projects** — Package ML code for reproducible runs.
4. **Models** — Standard model packaging format (MLflow Model) with inference interfaces.

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("xgboost-experiment")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.94)
    mlflow.log_artifact("model.pkl")
    mlflow.sklearn.log_model(model, "model")
```

**Pros**: Open source, large community, integrates with most ML frameworks.  
**Cons**: UI is basic, requires a server for team use.

### Weights & Biases (W&B)

A commercial SaaS platform (with generous free tier) focused on deep learning.

```python
import wandb

wandb.init(project="my-project", config={"lr": 0.01})
wandb.log({"accuracy": 0.94, "loss": 0.23})
wandb.save("model.pkl")
```

**Pros**: Beautiful UI, automatic GPU monitoring, rich hyperparameter sweeps, collaborative dashboards.  
**Cons**: SaaS (data leaves your network unless using self-hosted option), cost at scale.

### Neptune

SaaS platform with strong support for organizing complex experiments. Excellent for large teams.

```python
import neptune

run = neptune.init_run(project="my-project", tags=["transformer", "v2"])
run["params/lr"] = 0.001
run["metrics/accuracy"].append(0.94)
run["model"].upload("model.pkl")
```

**Pros**: Structured hierarchy (params, metrics, artifacts), flexible metadata, good team features.  
**Cons**: Paid, learning curve for advanced features.

### Comet

Another full-featured SaaS platform with automatic logging for many frameworks.

```python
import comet_ml

experiment = comet_ml.Experiment(project_name="my-project")
experiment.log_parameter("lr", 0.001)
experiment.log_metric("accuracy", 0.94)
```

**Pros**: Auto-logging for Keras, PyTorch, XGBoost; works out of the box.  
**Cons**: SaaS, can be noisy with auto-logging.

## Comparison Matrix

| Feature                  | MLflow           | W&B              | Neptune            | Comet             |
| ------------------------ | ---------------- | ---------------- | ------------------ | ----------------- |
| Open source              | Yes              | No (agent open)  | No                 | No                |
| Self-hosted              | Yes              | Yes (paid)       | No                 | Yes (paid)        |
| Hyperparameter sweeps    | Basic            | Excellent        | Good               | Good              |
| Model registry           | Built-in         | Via Artifacts    | Built-in           | Built-in          |
| Collaboration            | Via server       | Real-time        | Team workspaces    | Team workspaces   |
| Deep learning monitoring | Basic            | Excellent        | Good               | Good              |
| Free tier                | Unlimited        | Generous         | Monthly credits    | Generous          |

## Metrics vs Parameters — What to Log

**Always log**:
- Hyperparameters (learning rate, batch size, architecture)
- Framework/library versions (`pip freeze`)
- Dataset version hash or path
- Code commit hash (`git rev-parse HEAD`)
- Training duration and hardware type

**Always log as metrics**:
- Primary evaluation metric (accuracy, F1, RMSE)
- Loss curves (per epoch)
- Training vs validation performance (for overfitting detection)
- Resource utilization (GPU memory, CPU)

**Optionally log as artifacts**:
- Model weights
- Feature importance plots
- Confusion matrix
- Sample predictions on test data
- Environment dump (`pip freeze > requirements.txt`)

## Best Practices

- **One experiment, one folder** — logically group related runs.
- **Name runs systematically** — include data version, model type, and date: `rf_v2_data20240315`.
- **Auto-log when possible** — MLflow autologging: `mlflow.autolog()`.
- **Tag production runs** — mark runs that were deployed with tags like `stage:production`.
- **Clean up failed runs** — delete or tag them as `status:failed` to avoid cluttering dashboards.
- **Set up a server** — a shared MLflow Tracking Server (even on a free-tier VM) is better than file-based logging.

## Summary

Experiment tracking is the scientific record of your ML practice. It eliminates guesswork, enables collaboration, and builds a bridge from training to production. Start with MLflow (open source, self-hosted, broad ecosystem), then evaluate W&B or Neptune if you need richer visualization or team workflows.
