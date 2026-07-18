# ML vs Statistics vs Rules-Based Systems

## Comparison Table

| Aspect | Rules-Based | Statistics | Machine Learning |
|---|---|---|---|
| **Approach** | Hand-coded logic (`if-then-else`) | Mathematical equations, hypothesis testing | Data-driven pattern discovery |
| **Human effort** | High: every edge case needs a rule | Medium: feature engineering + model selection | Low for inference, high for data prep |
| **Handles complexity** | Poor (explodes in rules) | Moderate (linear assumptions) | High (non-linear, high-dim) |
| **Interpretability** | Trivial (you wrote the rules) | High (coefficients, p-values) | Varies (linear models = high, deep nets = low) |
| **Data need** | None | Moderate | Large |
| **Generalization** | None (only coded cases) | Limited (assumptions matter) | Strong (if data is representative) |

## When Each Shines

### Rules-Based
- Tax calculation: `if income < 10L: 0% tax elif income < 50L: 5%` — deterministic, legally required to be explicit
- Authentication: `if password == stored_hash: grant access` — no learning needed
- Medical device alerts: `if heart_rate > 120: beep` — must be 100% predictable

### Statistics
- A/B testing: "Does the new button increase conversion?" — hypothesis test with p-values
- Epidemiology: "Is smoking correlated with lung cancer?" — logistic regression with confidence intervals
- Survey analysis: "What factors predict customer satisfaction?" — linear regression with interpretable coefficients

### Machine Learning
- Recommendation: "What should this user watch next?" — collaborative filtering on 100M interactions
- Speech recognition: "Transcribe this audio" — deep learning on 10K hours of labeled speech
- Fraud detection: "Is this transaction fraudulent?" — gradient boosting on 1M transactions with 200 features
