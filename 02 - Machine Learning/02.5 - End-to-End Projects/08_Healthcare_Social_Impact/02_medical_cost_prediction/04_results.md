# Results — Medical Cost Prediction

## Metrics (test set)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear (log-charges) | $3,210 | $7,150 | 0.78 |
| Gamma GLM | $3,050 | $6,920 | 0.80 |
| Gradient Boosting | $2,480 | $5,610 | 0.87 |
| **Ensemble (0.4 LR + 0.6 GBM)** | **$2,390** | **$5,420** | **0.88** |

## What Was Learned

- **Smoker status dominates** — smokers cost 3–4x more on average.
- **Log-transform is essential** for linear models; GBM handles skew natively.
- **Ensemble** smooths GBM overpredictions on the highest-cost outliers.
- Gamma GLM is competitive but harder to tune (link function choice matters).

## Failure Cases

- Model underpredicts for young smokers with high BMI (training has few examples).
- Top 5% of charges ( > $40k) have 2x higher error — consider a separate high-cost model.
- Region has weak signal — dropping it doesn't hurt performance.
