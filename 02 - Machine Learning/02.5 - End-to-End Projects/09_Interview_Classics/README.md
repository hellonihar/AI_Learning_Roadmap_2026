# Interview Classics

Comprehensive end-to-end ML projects covering the most common interview challenges. Each project emphasizes feature engineering, model interpretability, and deployment thinking.

## Projects

| # | Project | Problem Type | Key Techniques | Target Metric |
|---|---------|-------------|----------------|---------------|
| 1 | Titanic Survival | Binary Classification | Title extraction, family size, cabin letter, RF/GBM | Accuracy ≥ 0.83 |
| 2 | House Prices | Regression | Elastic Net, GBM, stacking, log-transform | RMSE |
| 3 | Customer Churn | Binary Classification | sklearn Pipeline, experiment tracking, cost-benefit | AUC-ROC |
| 4 | Credit Risk | Binary Classification | WoE binning, Logistic Regression, SHAP, regulatory docs | AUC-ROC |
| 5 | Recommendation System | Collaborative Filtering | SVD (surprise), cold-start, REST API, A/B test | RMSE |

## Setup

```bash
pip install pandas numpy scikit-learn shap matplotlib seaborn xgboost surprise fastapi uvicorn
```
