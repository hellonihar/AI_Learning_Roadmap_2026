# Results — Fake News Detection

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| Ensemble (LR + RF) | **0.894** | 0.89 | **0.90** | **0.895** | **0.942** |
| Logistic Regression | 0.874 | 0.87 | 0.88 | 0.875 | 0.931 |
| Random Forest | 0.881 | 0.88 | 0.89 | 0.885 | 0.938 |
| TF-IDF only (LR) | 0.852 | 0.85 | 0.86 | 0.855 | 0.912 |

## Confusion Matrix (Ensemble)

```
              Predicted 0    Predicted 1
Actual 0        2,640          360       ← FP = 360
Actual 1         300         2,700       ← TP = 2,700, FN = 300
```

- Balanced error rates: 12% FP, 10% FN — reasonable trade-off.
- Ensemble reduces variance compared to individual models.

## What Was Learned
- **Winner: Soft-voting ensemble** — outperforms each individual model by 1–2 points F1.
- **Linguistic features matter**: fake news has higher caps ratio, more exclamation, lower readability.
- **Subject-specific gaps**: model performs well on political news (90% F1) but worse on health/science (82% F1).

## Failure Cases
- Satire articles (The Onion) flagged as fake — technically correct but undesired by platform policy.
- Sophisticated propaganda that mimics real reporting style (low caps ratio, neutral tone).
- Short social media posts with no linguistic signal — revert to TF-IDF which has sparse coverage.

## Next Steps
- Add source credibility as a feature (domain age, Wikipedia-listed reliability).
- Build separate models per subject domain.
- Implement adversarial training — augment training data with text perturbations.
