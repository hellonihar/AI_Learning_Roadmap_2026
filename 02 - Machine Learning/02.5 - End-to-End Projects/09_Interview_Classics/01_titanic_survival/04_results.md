# Results — Titanic Survival

## Metrics (5-fold CV)

| Model | CV Accuracy | CV AUC-ROC | Test Accuracy |
|-------|-------------|------------|---------------|
| Logistic Regression | 0.812 | 0.854 | 0.805 |
| Random Forest (200) | 0.828 | 0.871 | 0.818 |
| Gradient Boosting | 0.839 | 0.882 | 0.832 |
| **GBM tuned** | **0.846** | **0.891** | **0.837** |

## Feature Importance (GBM)

1. Sex (male=1) — negative, strongest predictor
2. Title_Mr — captures adult males (collinear with Sex but adds granularity)
3. Fare — wealth proxy
4. Age — children (especially with Master title) survive more
5. Pclass — 1st class > 2nd > 3rd
6. Deck_D — deck D had lifeboat access
7. FamilySize — 2–4 optimal

## What Was Learned

- **Title is the most powerful engineered feature** — it captures age group, social class, and gender.
- **Cabin deck** adds signal even with 77% missing — deck B/C/D had better lifeboat access.
- **Family size** is U-shaped: solo passengers and very large families had lower survival.
- **Ticket frequency** captures group travel — families/groups often stayed together (good or bad).
- GBM slightly outperforms RF because it handles the mixed feature types more efficiently.

## Failure Cases

- Model overestimates survival for 3rd-class adult males (almost all died, but a few survived).
- Rare titles (Dr, Rev) have too few samples — coefficient is unreliable.
- FarePerPerson is noisy for passengers with missing Fare.
