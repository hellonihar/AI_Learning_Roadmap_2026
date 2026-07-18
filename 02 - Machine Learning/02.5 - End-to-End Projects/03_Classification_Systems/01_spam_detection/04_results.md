# Results — Spam Detection

## Model Comparison

| Model | Precision (Spam) | Recall (Spam) | AUC-ROC | F1 |
|-------|------------------|---------------|---------|-----|
| MultinomialNB (calibrated) | 0.985 | 0.93 | 0.993 | 0.956 |
| LinearSVC | 0.981 | 0.91 | 0.991 | 0.944 |
| Logistic Regression | 0.978 | 0.90 | 0.990 | 0.937 |

## Confusion Matrix Analysis (NB, threshold = 0.65)

```
              Predicted Ham    Predicted Spam
Actual Ham        5,800             22
Actual Spam          63            825
```

- 22 false positives (precision cost) — most were marketing newsletters.
- 63 false negatives — spam with unusual character patterns or very short text.

## What Was Learned
- **Winner: MultinomialNB** — slightly better precision on sparse high-dimensional data.
- **Calibration matters** — without isotonic calibration, precision at default threshold was 0.91.
- **N-grams > unigrams** — catching "act now" and "you've won" bigrams improved recall by 5 points.

## Failure Cases
- Fully image-based spam (no text features) — undetectable by text model.
- Legitimate password-reset emails flagged as spam (contain "click here to verify").
- SMS with heavy leetspeak ("fr33 m0n3y") — tokenizer strips digits, losing signal.

## Next Steps
- Integrate image OCR pipeline for image-only spam.
- Add allowlist for known transactional email domains.
