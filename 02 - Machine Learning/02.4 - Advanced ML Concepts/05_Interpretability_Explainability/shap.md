# SHAP (SHapley Additive exPlanations)

SHAP values explain **individual predictions** by fairly distributing the prediction among features. Based on Shapley values from cooperative game theory.

*A focused deep-dive. For the broader context of interpretability, see the [Interpretability overview](interpretability.md).*

## How SHAP Works

Imagine a prediction as a game where each feature "plays" to achieve the outcome. SHAP calculates each feature's contribution by considering all possible combinations of features and averaging their marginal contributions.

For a model with 3 features (income, credit_score, debt_ratio), SHAP asks:
- What does the model predict with NO features? (base value)
- What does it predict with just income?
- What does it predict with income + credit_score?
- What does it predict with all three?
- How much did adding each feature change the prediction, averaged over all possible ordering?

## SHAP Output

For a single prediction, SHAP produces:

```
base_value (average prediction) + sum(feature_contributions) = prediction

Example — Loan denied:
base_value:             0.35  (35% default risk — population average)
+ income = $30K:       +0.25
+ debt_ratio = 0.6:    +0.20
+ credit_score = 720:  -0.10
+ has_collateral = 1:  -0.05
= prediction:          0.65  (65% → denied)
```

## Global SHAP

SHAP values for individual predictions can be aggregated to show global importance:

- **SHAP summary plot**: Bar chart of mean absolute SHAP values per feature — ranks feature importance globally
- **SHAP beeswarm plot**: Each point is a SHAP value for a single feature × single prediction. Color shows feature value. Reveals direction and distribution of effects.

```
              Feature Value
beeswarm:     Low ●●●●○○○ High
income        ●●●●○○○○○○○○○   (most important)
credit_score  ○○●●●●●●●●○○○
debt_ratio    ●●●●○○○○○○○○
has_collateral ○○○○●●●●●○○○
```

## SHAP vs LIME

| Aspect | SHAP | LIME |
|---|---|---|
| **Consistency** | Guaranteed (same features = same contribution) | Can be unstable across runs |
| **Speed** | Slower (exact SHAP is exponential; TreeSHAP is fast) | Faster |
| **Global explanations** | Yes (aggregate SHAP values) | No |
| **Feature interactions** | Captured naturally | Captured locally |
| **Theoretical basis** | Game theory (Shapley values — unique fair solution) | Heuristic local approximation |

## When to Use SHAP

- **Regulated industries** — need rigorous, consistent explanations (banking, healthcare, insurance)
- **Model debugging** — understand why specific predictions are wrong
- **Feature engineering feedback** — see which features actually drive predictions
- **Stakeholder trust** — showing concrete per-case explanations builds confidence

## Examples

1. **Credit denial explanation (ECOA compliance)**: US Equal Credit Opportunity Act requires lenders to explain adverse actions. SHAP generates a legally defensible explanation for each denied applicant: "Your application was denied primarily due to debt-to-income ratio (0.55), which contributed +18 percentage points to default risk. Your credit score of 680 was favorable (−5 points) but insufficient to offset."
2. **Medical alert debugging**: A sepsis prediction model flagged a patient who didn't develop sepsis. SHAP reveals: "The model was most influenced by heart rate (abnormal, +0.12) and respiratory rate (elevated, +0.10). However, white blood cell count (normal, −0.08) and lactate (normal, −0.05) contradicted the sepsis signal. The model over-weighted vitals over labs." This identified a calibration issue.
3. **Fraud model tuning**: Fraud team reviews 100 false positives per day. SHAP on each case shows which features triggered the alert. Pattern emerges: 40% of false positives were driven by "distance_from_home" flagging business trips. The team adds a "known_travel_pattern" feature, reducing false positives by 30%.
