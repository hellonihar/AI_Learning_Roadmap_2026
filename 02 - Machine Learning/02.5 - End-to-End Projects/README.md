# 02.5 — End-to-End Projects

Practical, code-driven ML projects spanning 9 application domains. Each project is a self-contained guide with problem statement, dataset walkthrough, implementation code, results analysis, and deployment considerations.

## Categories

| # | Category | Projects | Focus |
|---|---|---|---|
| 01 | [Recommendation Systems](01_Recommendation_Systems/) | 5 | Collaborative filtering, content-based, matrix factorization, ranking |
| 02 | [Prediction & Forecasting](02_Prediction_Forecasting/) | 5 | Regression, time-series, demand forecasting, hierarchical |
| 03 | [Classification Systems](03_Classification_Systems/) | 5 | Spam, fraud, churn, fake news — with imbalance + interpretability |
| 04 | [Computer Vision (Intro)](04_Computer_Vision_Intro/) | 3 | HOG/SIFT + SVC, color histograms, pixel features — no DL |
| 05 | [NLP Systems](05_NLP_Systems/) | 4 | TF-IDF + Naive Bayes/SVM, cosine similarity — no transformers |
| 06 | [Anomaly & Risk Detection](06_Anomaly_Risk_Detection/) | 4 | Isolation Forest, Autoencoders, One-Class SVM, threshold tuning |
| 07 | [Customer & Business Intelligence](07_Customer_Business_Intelligence/) | 4 | Segmentation, market basket, CLV, behavior analytics |
| 08 | [Healthcare & Social Impact](08_Healthcare_Social_Impact/) | 2 | Clinical prediction, medical cost — with SHAP explainability |
| 09 | [Interview Classics](09_Interview_Classics/) | 5 | Titanic, house prices, churn, credit risk, recommendation — end-to-end |

**Total**: 37 projects across 9 categories

## Project File Structure

Each project contains 5 files:

| File | What It Covers |
|---|---|
| `01_problem_statement.md` | Business context, problem type, success metrics, constraints |
| `02_dataset.md` | Source, size, features, target, known challenges |
| `03_implementation.md` | Step-by-step code walkthrough with key snippets |
| `04_results.md` | Metrics, findings, failure cases, what was learned |
| `05_deployment_notes.md` | Inference latency, retraining, monitoring, API design |

## Stack

All projects use only **basic ML libraries**: pandas, numpy, scikit-learn, imbalanced-learn, surprise (recommendation). No deep learning frameworks — those are reserved for later sections.
