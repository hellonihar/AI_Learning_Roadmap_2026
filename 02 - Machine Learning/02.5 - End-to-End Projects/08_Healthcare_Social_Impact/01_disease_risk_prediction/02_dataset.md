# Dataset — Heart Disease (UCI)

## Source
UCI Machine Learning Repository — Heart Disease dataset (Cleveland subset).  
[Link](https://archive.ics.uci.edu/ml/datasets/heart+disease)

## Size
- 303 rows, 14 columns
- Train/val/test split: 60/20/20

## Features

| Feature | Type | Description |
|---------|------|-------------|
| age | numeric | Age in years |
| sex | binary | 1 = male, 0 = female |
| cp | categorical | Chest pain type (1–4) |
| trestbps | numeric | Resting blood pressure (mm Hg) |
| chol | numeric | Serum cholesterol (mg/dl) |
| fbs | binary | Fasting blood sugar > 120 mg/dl |
| restecg | categorical | Resting ECG results (0–2) |
| thalach | numeric | Max heart rate achieved |
| exang | binary | Exercise-induced angina |
| oldpeak | numeric | ST depression induced by exercise |
| slope | categorical | Slope of peak exercise ST segment |
| ca | numeric | Number of major vessels (0–3) |
| thal | categorical | Thalassemia (3, 6, 7) |

## Target
- `target`: 0 = no heart disease, 1 = heart disease (merged original values 1–4 into 1)

## Known Challenges
- Small dataset (303 rows) — risk of overfitting, use cross-validation.
- Missing values in `ca` and `thal` (few rows).
- Class imbalance ~ 54% / 46% (mild).
- Features are correlated (e.g., age with thalach).
