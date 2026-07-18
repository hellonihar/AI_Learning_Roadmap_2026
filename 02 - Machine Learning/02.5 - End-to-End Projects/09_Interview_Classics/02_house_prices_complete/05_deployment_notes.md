# Deployment Notes — House Prices

## Inference
- Input: 79 raw features → output: predicted sale price (in dollars, back-transformed from log).
- Use `Pipeline` with `ColumnTransformer` for preprocessing.
- Serialize all transformers + models with `joblib`.

```python
import joblib
joblib.dump(stacking_pipeline, "house_price_model.pkl")

# Inference
def predict(features: dict) -> float:
    df = pd.DataFrame([features])
    log_pred = stacking_pipeline.predict(df)[0]
    return np.expm1(log_pred)
```

## Retraining
- Retrain quarterly with new sales data.
- Monitor for:
  - New neighborhoods not seen in training
  - Material/quality rating shifts
  - Economic inflation (adjust target or use relative pricing)

## Monitoring
- Track RMSE by neighborhood — if error spikes in one area, investigate.
- Monitor feature drift: OverallQual distribution, YearBuilt range.
- Log prediction intervals (from GBM quantile regression) to flag high-uncertainty predictions.

## Limitations
- No temporal features — can't model market trends or seasonality beyond YrSold.
- No location coordinates — neighborhood is coarse.
- Model assumes static feature relationships — housing market shifts require retraining.
