# Interpretability & Explainability

## Model Interpretability vs Explainability

These are related but distinct concepts:

| | Interpretability | Explainability |
|---|---|---|
| **Definition** | Degree to which a human can understand the model's **inner workings** | Degree to which a human can understand **why a specific prediction** was made |
| **Scope** | The model itself | Individual predictions |
| **Example** | Linear regression: "coefficient 2.5 means each sqft adds \$2.50" | SHAP: "This loan was denied because income is low and debt ratio is high" |
| **Models** | Linear, Decision Trees, kNN | Black-box models (NN, XGBoost) with explanation tools |

Interpretable models are **inherently explainable**. Black-box models need **post-hoc explainability**.

## Global vs Local Explanations

| | Global | Local |
|---|---|---|
| **What** | How the model behaves **overall** | Why a **single prediction** was made |
| **Answers** | "Which features matter most?" | "Why was this customer denied?" |
| **Methods** | Feature importance, Partial dependence | SHAP, LIME |
| **Audience** | Data scientists, stakeholders | Regulators, end users, auditors |

## Feature Importance

How much each feature contributes to the model's predictions.

### Built-in Importance (Tree Models)
Random Forest and Gradient Boosting track how much each feature reduces impurity (Gini/MSE) across all splits. Features used near the top of many trees get higher scores.

**Example**: Customer churn model — top features: `login_frequency` (0.35), `support_tickets_30d` (0.28), `account_age` (0.15), `payment_method` (0.10). The message: "Login frequency and support tickets are by far the strongest churn indicators."

### Permutation Importance
Shuffle one feature's values, measure how much performance drops. If shuffling a feature causes accuracy to plummet, it's important. If accuracy stays the same, it's irrelevant.

**Advantages**: Model-agnostic (works for any model), captures feature interactions naturally.

**Example**: Medical diagnosis model — shuffle `age` → AUC drops from 0.92 to 0.75. Shuffle `hair_color` → AUC stays at 0.92. Conclusion: age is critical, hair color is irrelevant.

## Partial Dependence Plots (PDP)

Show how a feature affects predictions **on average**, marginalizing over all other features.

```
Plot: temperature (x-axis) vs predicted ice cream sales (y-axis)
Shape: upward slope up to 30°C, then downward — captures the non-linear relationship.
```

**When to use**: Understanding feature relationships, detecting non-linear patterns, checking if model behavior aligns with domain knowledge.

**Example**: House price model — PDP for `sqft` shows a steep slope up to 3,000 sqft, then plateau. This matches reality: adding square footage adds value up to a point, but mega-mansions have diminishing returns.

## SHAP (SHapley Additive exPlanations)

Based on **Shapley values** from cooperative game theory. Each feature gets a "fair" contribution score for each prediction.

**How it works**: Consider each feature as a "player" in a game (the prediction). SHAP computes how much each player contributed, averaging over all possible coalitions.

**Output example**: For a single loan denial prediction:
```
Base value (average prediction): 0.35 (35% default risk)
SHAP contributions:
    income = $30K:     +0.25 (increases risk)
    debt_ratio = 0.6:  +0.20 (increases risk)
    credit_score = 720: -0.10 (decreases risk)
    has_collateral = 1: -0.05 (decreases risk)
Final prediction: 0.35 + 0.25 + 0.20 - 0.10 - 0.05 = 0.65 (65% → denied)
```

**Why SHAP is powerful**: Consistent (same feature importance across models), locally accurate (sum of contributions equals prediction), and globally interpretable (aggregate SHAP values show global feature importance).

**Examples:**
1. **Loan denial explanation (regulatory)**: A denied applicant asks "why?" SHAP shows: "Your debt ratio of 0.6 added +0.20 to default risk. Your income of \$30K added +0.25. If your debt ratio were 0.3 instead, your risk would drop to 45% — still high, but combined with other improvements..." This is what regulators require.
2. **Medical diagnosis**: A patient is flagged as high-risk for diabetes. SHAP shows: "BMI of 32 (+0.15), family history (+0.10), age 55 (+0.08), but active lifestyle (-0.05) and normal blood pressure (-0.03)."
3. **Fraud investigation**: A transaction flagged as fraud. SHAP shows: "Amount \$2,500 (unusual for this user, +0.30), time 3 AM (+0.20), merchant_category 'wire transfer' (+0.15), distance_from_home 500 miles (+0.25)."

## LIME (Local Interpretable Model-agnostic Explanations)

Explains individual predictions by fitting a **simple interpretable model** (linear model) locally around the prediction.

**How it works**: Perturb the input (turn features on/off), observe how predictions change, and fit a weighted linear model in that neighborhood.

**Output**: A list of features with weights: "This review was classified as negative because it contains words 'terrible' (+2.3), 'boring' (+1.8), 'waste' (+1.5)."

**Comparison with SHAP**:

| | SHAP | LIME |
|---|---|---|
| **Theoretical foundation** | Game theory (Shapley values) | Local linear approximation |
| **Consistency** | Consistent (same features = same contribution) | Unstable (different runs may differ) |
| **Speed** | Slower (computes all coalitions) | Faster (local neighborhood only) |
| **Global explanations** | Yes (aggregate SHAP values) | No (local only) |
| **Feature interactions** | Captured | Captured (but local) |

## When to Use Which

| Goal | Method |
|---|---|
| "Which features matter most globally?" | Feature importance (built-in or permutation) |
| "How does feature X affect predictions?" | Partial dependence plot |
| "Why was THIS prediction made?" | SHAP (preferred) or LIME |
| "I need model-agnostic, mathematically rigorous attribution" | SHAP |
| "I need fast, simple explanations for non-experts" | LIME |
| "My model is linear/logistic — I already have coefficients" | Coefficients (most interpretable) |

## Examples

1. **Regulatory compliance (banking)**: Bank must explain each loan denial. SHAP provides per-customer explanations that satisfy regulations. The explanation must be "specific to the individual's circumstances" — global feature importance doesn't suffice.
2. **Model debugging**: XGBoost shows unexpected behavior on certain customer segments. Partial dependence plots reveal that the model learned an unintuitive pattern (e.g., older customers with high income are predicted HIGHER risk — a data artifact). SHAP on specific mispredictions confirms the issue.
3. **Doctor trust**: A black-box model predicts sepsis risk. SHAP shows which vital signs drove the prediction: "heart rate 112 (abnormal), temperature 38.5°C (fever), white blood cell count 14K (elevated)." The doctor sees the model is using clinically sensible features → trusts the alert.
