# Linear vs Non-Linear Models (Conceptual)

## Linear Models

Output is a **weighted sum** of the inputs: `y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`

- Decision boundary: a straight line (2D), plane (3D), or hyperplane (nD)
- **Interpretability**: Easy — you can read the coefficient for each feature
- **Limitation**: Can't capture interactions or non-linear patterns unless you manually add interaction features

**Examples:**
1. **Linear regression for house prices**: `price = 250*sqft + 30K*bedrooms — 50K*location_penalty + 100K`. Each coefficient directly tells you the impact of each feature.
2. **Logistic regression for spam detection**: Score = 0.5*"viagra_count" + 0.3*"exclamation_count" — 2.0*"known_sender_score". A positive score → spam. Each feature contributes independently.
3. **Credit scoring**: Score = age*0.1 + income*0.05 — debt*0.2 + 300. Simple, auditable, explainable to regulators.

## Non-Linear Models

Output can have **complex relationships** with inputs: interactions, thresholds, hierarchies.

- Decision boundary: curved, twisted, can separate complex patterns
- **Interpretability**: Hard to impossible (especially deep neural networks)
- **Power**: Can automatically learn interactions without manual feature engineering

**Examples:**
1. **Decision tree for loan approval**: `IF income > 50K AND debt_ratio < 0.3: approve. ELSE IF income > 50K AND debt_ratio > 0.3: review.` — non-linear because the same income threshold leads to different outcomes depending on debt_ratio.
2. **Random forest for fraud detection**: Combines hundreds of trees, each capturing different non-linear patterns (e.g., "small amount + overseas + 3 AM" vs "large amount + domestic + business hours"). The ensemble captures complex interactions automatically.
3. **Deep neural network for image recognition**: Pixels → edges → shapes → objects → concepts. Each layer learns increasingly abstract non-linear transformations. No linear model can learn "cat" from raw pixels.

## When to Use Which

| Scenario | Linear | Non-Linear |
|---|---|---|
| Need interpretability | ✅ | ❌ |
| Small dataset (<1000 samples) | ✅ | ⚠️ (risk of overfitting) |
| Huge dataset (>100K) | Tuning | ✅ (deep learning) |
| Feature interactions are simple | ✅ | Overkill |
| Feature interactions are complex | Needs manual engineering | ✅ (automatic) |
| Real-time / low-latency | ✅ (fast inference) | ⚠️ (slower) |

**Practical tip**: Always start with a linear model as a baseline. If it performs poorly, add non-linear models.
