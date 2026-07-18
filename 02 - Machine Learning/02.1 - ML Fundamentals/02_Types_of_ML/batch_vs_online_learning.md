# Batch Learning vs Online Learning

## Batch Learning

Train on the entire dataset at once. Deploy the model. It stays fixed.

**Pros**: Stable, well-understood, easy to validate
**Cons**: Must retrain from scratch to incorporate new data

**Examples:**
1. **Credit scoring model**: Train on all 2025 loan data, deploy in Jan 2026, update quarterly. The model doesn't change day-to-day — consistent predictions for customers.
2. **Movie recommendation (classic Netflix Prize)**: Train on the full dataset of 100M ratings, produce a fixed model. New ratings aren't incorporated until the next retraining cycle.
3. **Medical diagnosis system**: Train on a fixed dataset of 50K labeled scans. Deploy and validate. New cases are collected and the model retrained infrequently after review.

## Online (Incremental) Learning

Update the model one example (or mini-batch) at a time. The model evolves continuously.

**Pros**: Adapts to new patterns, minimal storage, fast updates
**Cons**: Sensitive to data order, can drift, harder to validate

**Examples:**
1. **Ad click prediction**: Model sees each ad impression and whether it was clicked. Updates weights within milliseconds. Must adapt to changing user interests and new ads every second.
2. **Stock market prediction**: New price data arrives every microsecond. An online model continuously adjusts to regime changes (bull → bear market) without full retraining.
3. **IoT sensor monitoring**: A temperature sensor on a factory floor sends readings every second. An online model detects anomalies in real-time and adapts to seasonal drift (summer vs winter baselines).

## Comparison

| Aspect | Batch | Online |
|---|---|---|
| When to use | Stable patterns, periodic updates | Evolving patterns, real-time needs |
| Training data | All data at once | One sample at a time |
| Storage | Full dataset needed | Can discard after update |
| Adaptation | Only after retraining | Continuous |
| Validation | Standard train/test split | Time-series-aware, concept drift monitoring |
