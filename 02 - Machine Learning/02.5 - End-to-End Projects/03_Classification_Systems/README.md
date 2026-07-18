# Classification Systems — End-to-End Projects

Five binary classification projects spanning text, tabular, and imbalanced datasets.

| # | Project | Problem | Key Technique | Metric Focus |
|---|---------|---------|---------------|-------------|
| 1 | Spam Detection | SMS/Email spam | TF-IDF + MultinomialNB / LinearSVC | Precision |
| 2 | Credit Card Fraud | 0.1% fraud rate | SMOTE + threshold moving | Precision-Recall AUC |
| 3 | Loan Default | Interpretability | Logistic Regression + SHAP | AUC-ROC |
| 4 | Customer Churn | Subscription churn | RF / XGBoost, feature engineering | Recall / F1 |
| 5 | Fake News Detection | Misinformation | TF-IDF + linguistic features + ensemble | Accuracy / F1 |

Each project follows the same five-file structure:
- `01_problem_statement.md` — business context & success metrics
- `02_dataset.md` — source, schema, class balance, challenges
- `03_implementation.md` — step-by-step with code snippets
- `04_results.md` — model comparison & failure-case analysis
- `05_deployment_notes.md` — latency, retraining, monitoring
