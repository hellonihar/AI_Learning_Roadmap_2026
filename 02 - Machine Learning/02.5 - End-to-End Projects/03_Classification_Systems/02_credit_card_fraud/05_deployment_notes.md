# Deployment Notes — Credit Card Fraud Detection

## Inference Pipeline

```python
def score_transaction(transaction: dict) -> dict:
    df = pd.DataFrame([transaction])
    df["Amount"] = scaler.transform(df[["Amount"]])
    proba = model.predict_proba(df)[0, 1]
    decision = proba >= DEPLOYED_THRESHOLD
    return {"fraud_score": proba, "block": bool(decision)}
```

- Latency: ~5 ms per transaction (LightGBM native, no Python overhead in production).
- Model size: 8 MB (300 trees, depth 6).

## Retraining Strategy
- **Daily rolling window**: train on last 90 days of transactions.
- **Label availability**: chargeback/dispute confirmation takes 30–60 days — use confirmed labels.
- **Semi-supervised**: flag high-confidence predictions as pseudo-labels for next day's training.

## Threshold Management
- Track weekly: PR-AUC, precision@k (top 1% of scores), FP rate per merchant.
- If review team is overwhelmed (FP > 500/day), raise threshold by 0.05.
- If fraud losses exceed target, lower threshold by 0.05.

## Monitoring
- **Model inputs**: drift on each PCA component (Kolmogorov-Smirnov test).
- **Outputs**: daily fraud rate, average fraud score, threshold hit rate.
- **Business**: chargeback rate, false positive cost, customer complaints.

## Infrastructure
- Deploy as sidecar container on the transaction processing pod.
- Model loaded from S3 on startup; hot-reload on `SIGUSR1`.
- Shadow-mode traffic for 24 hours before cutover on each redeploy.
