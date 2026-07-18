# Results — Disease Risk Prediction

## Metrics (test set)

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.92 |
| Sensitivity | 0.88 |
| Specificity | 0.83 |
| Accuracy | 0.86 |
| F1 Score | 0.87 |

Threshold tuned to 0.42 (lowered from 0.5 to boost sensitivity).

## SHAP Summary

Top 5 drivers (mean |SHAP|):
1. `thal` (7) — thalassemia reversibility
2. `ca` — number of major vessels
3. `oldpeak` — ST depression
4. `thalach` — max heart rate (inverse)
5. `cp` — chest pain type

## What Was Learned

- Logistic regression with tuned threshold matches more complex models on this small dataset.
- SHAP reveals that `ca` and `thal` dominate — these are strong clinical signals.
- Without threshold tuning, sensitivity was only 0.78 — unacceptable for screening.

## Failure Cases

- 3 false negatives had atypical presentation (high thalach but high oldpeak).
- Model struggles when multiple borderline values cancel each other out.
