# Healthcare & Social Impact

End-to-end ML projects focused on clinical and insurance data. Covers binary classification with interpretability (SHAP) and regression on skewed cost data.

## Projects

| # | Project | Problem Type | Key Techniques | Metric |
|---|---------|-------------|----------------|--------|
| 1 | Disease Risk Prediction | Binary Classification | Logistic Regression, SHAP, threshold tuning | Sensitivity / Specificity |
| 2 | Medical Cost Prediction | Regression (right-skewed) | Gamma GLM, log-transform, ensemble | MAE, RMSE |

## Setup

```bash
pip install pandas numpy scikit-learn shap matplotlib seaborn
```

Each project folder contains 5 files following the same template: problem statement, dataset description, implementation with code, results, and deployment notes.
