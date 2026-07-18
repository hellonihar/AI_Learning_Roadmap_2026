# End-to-End ML Workflow

The full lifecycle of an ML project — from idea to deployment.

```
1. Problem Definition
       ↓
2. Data Collection
       ↓
3. Data Understanding & Exploration
       ↓
4. Data Preparation (cleaning, splitting, preprocessing)
       ↓
5. Modeling → Train → Evaluate → Iterate ←┐
       ↓                                       │
6. Model Selection (compare, choose best) ─────┘
       ↓
7. Final Evaluation (on untouched test set)
       ↓
8. Deployment Awareness
       ↓
9. Monitoring & Maintenance (post-deployment)
```

## 1. Problem Definition

- What business problem are we solving?
- Is it a classification, regression, clustering, or ranking problem?
- What metric defines success? (business metric, not ML metric)
- What's the baseline to beat?

**Example**: "Reduce customer churn by 20%. Predict which customers will cancel in the next 30 days. Success = revenue saved minus cost of retention offers."

## 2. Data Collection

- What data exists? (databases, logs, APIs, third-party)
- How much do we need? (minimum viable data)
- Is the data representative of the deployment scenario?
- Any privacy/legal constraints?

**Example**: 6 months of subscription logs, support tickets, usage patterns, payment history. Need at least 10K churned + 10K retained customers.

## 3. Data Understanding & Exploration

- Visualize distributions, correlations, missing values
- Identify data quality issues
- Generate hypotheses about which features matter
- Detect potential leakage sources

## 4. Data Preparation

- Handle missing values, outliers
- Feature engineering (create useful features from raw data)
- Encode categorical variables
- Scale/normalize features
- Train/validation/test split

## 5. Model Training & Iteration

- Start with simple baselines (mean predictor, linear model)
- Gradually increase complexity (tree-based → neural nets)
- Tune hyperparameters using validation set
- Track experiments (what was tried, what worked)

## 6. Model Selection

- Compare models on validation metrics
- Consider not just accuracy but: inference speed, memory, interpretability, deployability
- Choose the simplest model that meets requirements

## 7. Final Evaluation

- Run selected model on the **unseen test set** — exactly once
- Compute final metrics
- If performance is unacceptable, go back to step 5 (not step 7 — the test set is sacred)

## 8. Deployment Awareness

- Inference latency requirements (real-time vs batch)?
- Model size and memory constraints?
- How will the model be served? (REST API, embedded, mobile)
- Retraining strategy: how often to update?

**Example**: The churn model needs predictions within 100ms per customer, served as a REST API. Must fit in 200MB RAM. Retrain weekly with fresh subscription data.

## 9. Monitoring & Maintenance

- Track prediction distributions (data drift)
- Track model performance (concept drift)
- Set up alerts when metrics degrade
- Plan for retraining cycles

## Examples

1. **Retail demand forecasting**: Problem → "Predict weekly demand for 10K products across 500 stores." Data → 2 years of historical sales, promotions, weather, holidays. Baseline → last year's same week sales. Model → gradient boosting. Monitor → tracking forecast error by store, trigger retraining if error increases by 20%.
2. **Medical image classification**: Problem → "Flag potential pneumonia from chest X-rays." Data → 100K labeled X-rays, 5% positive. Baseline → radiologist's accuracy. Model → ResNet with AUC. Deployment → assistant to radiologist (not replacement). Monitor → AUC by hospital quarterly, retrain with new cases annually.
3. **LLM-powered customer support**: Problem → "Automate responses to common customer questions." Data → 50K past support tickets + resolutions. Baseline → keyword matching (40% resolution rate). Model → fine-tuned LLM. Deployment → AI drafts response, human approves (human-in-the-loop). Monitor → resolution rate, escalation rate, user satisfaction.
