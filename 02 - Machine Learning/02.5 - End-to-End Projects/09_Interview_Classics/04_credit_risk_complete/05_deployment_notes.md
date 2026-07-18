# Deployment Notes — Credit Risk

## Inference
- Input: applicant data → output: `{credit_score, risk_tier, decision, top_factors}`
- Risk tiers: Excellent (> 700), Good (650–700), Fair (600–650), Poor (< 600)
- Decision: Approve (≥ 650), Manual review (600–650), Decline (< 600)

## Regulatory Documentation
- **Model card**: purpose, population, performance, limitations
- **Feature documentation**: WoE bins, IV values, rationale for each bin boundary
- **Fairness audit**: disparate impact analysis by sex, age group
- **Monitoring plan**: AUC-ROC, KS, population stability index (PSI)

## Retraining
- Retrain quarterly or when PSI > 0.1.
- Recalculate WoE bins if feature distributions shift.
- Revalidate scorecard cutoffs with business stakeholders.

## Monitoring
- Track AUC-ROC and KS monthly.
- Monitor approval rate by demographic group for fairness.
- Log score distribution — if average score drifts, recalibrate.

## Limitations
- Small dataset (1,000 rows) — not sufficient for production credit scoring.
- No macroeconomic features (interest rates, unemployment).
- Scorecard assumes stable relationships — economic downturns require recalibration.
