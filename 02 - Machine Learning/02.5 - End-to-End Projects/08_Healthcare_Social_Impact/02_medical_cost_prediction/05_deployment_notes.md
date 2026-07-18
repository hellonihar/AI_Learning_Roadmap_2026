# Deployment Notes — Medical Cost Prediction

## Inference
- Input: `{age, sex, bmi, children, smoker, region}` → output: `{predicted_charge, prediction_interval}`
- Use FastAPI endpoint `/predict-cost`
- Return both point estimate and 90% prediction interval (from GBM quantile regression or bootstrapped residuals)

## Retraining
- Retrain annually with new claims data.
- Monitor for changes in:
  - Smoker prevalence (affects intercept)
  - Regional cost trends
  - BMI distribution shifts

## Monitoring
- Track MAE by decile — if error in top decile grows, retrain.
- Monitor mean prediction vs. actual per month.
- Log feature drift (PSI) for `bmi`, `age`, `smoker`.

## Limitations
- No temporal features — can't model medical inflation.
- No diagnosis codes — can't distinguish chronic vs. acute conditions.
- Small dataset — consider transfer learning from larger claims datasets if available.
