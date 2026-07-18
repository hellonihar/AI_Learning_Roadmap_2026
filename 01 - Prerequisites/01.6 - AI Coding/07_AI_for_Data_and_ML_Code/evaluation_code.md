# Evaluation Code

Generate comprehensive evaluation scripts for model performance.

## Prompt

```
"Write a model evaluation function that takes y_true and y_pred
and returns: accuracy, precision, recall, F1, ROC-AUC, confusion
matrix, classification report. Handle both binary and multi-class.
Use sklearn and return results as a dict."
```

## Example

```python
def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
        classification_report
    )

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred),
    }

    if y_proba is not None and len(np.unique(y_true)) == 2:
        results["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])

    return results
```

## Evaluation Prompts

```
- "Write a function to plot confusion matrix, ROC curve, and precision-recall curve"
- "Generate a model card (model metadata, performance by slice, intended use)"
- "Write bias/fairness evaluation: performance by demographic groups"
- "Create an A/B evaluation framework comparing two model versions"
- "Write a function to compute prediction drift between training and production data"
- "Generate a cost-benefit analysis at different threshold values"
```
