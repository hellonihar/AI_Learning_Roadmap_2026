# When NOT to Use ML

## Anti-Patterns

### 1. You Have a Simple, Deterministic Rule
- **Bad**: ML to check if a number is even or to validate email format
- **Better**: `if x % 2 == 0` or a regex
- Why: ML adds complexity, latency, and a failure surface for zero benefit

### 2. You Have No Data (or Terrible Data)
- **Bad**: ML to predict customer churn with only 50 samples
- **Better**: Domain-expert rules or a simple heuristic
- Why: ML needs data to learn — with too little, it will overfit or fail to generalize

### 3. You Need Perfect Accuracy
- **Bad**: ML for a medical diagnosis system expected to be 100% correct
- **Better**: Combine ML as a screening tool with human verification
- Why: ML is probabilistic — it will always have some error rate. In safety-critical systems, use ML as an aid, not a decision-maker

### 4. The Problem Changes Constantly
- **Bad**: ML to detect fraud patterns that evolve every week, with no retraining pipeline
- **Better**: Rules for fast-changing patterns + periodic model retraining for stable ones
- Why: A static model deteriorates (concept drift). If you can't retrain frequently, ML will degrade

### 5. Interpretability Is Legally Required
- **Bad**: Deep learning for loan approval decisions where regulators demand exact "why this was denied" logic
- **Better**: Logistic regression or a credit scoring rulebook
- Why: Black-box models make it impossible to explain individual decisions. Regulations (GDPR, ECOA) often require explanation

### 6. Cost of Mistakes Is Catastrophic
- **Bad**: ML-only self-driving car with no safety fallback
- **Better**: ML + explicit safety layer (e.g., hard-coded collision avoidance)
- Why: ML can fail in unpredictable ways. Safety-critical systems need deterministic backups

## Decision Flowchart

```
Can you write a simple rule? ──yes──→ Use rule. No ML needed.
        ↓ no
Do you have enough quality data? ──no──→ Collect data or use heuristics.
        ↓ yes
Is perfect accuracy required? ──yes──→ ML as aid + human in loop.
        ↓ no
Does the problem change weekly? ──yes──→ Need automated retraining pipeline.
        ↓ no
Is interpretability legally required? ──yes──→ Choose interpretable model.
        ↓ no
Go ahead with ML ── but always have a baseline!
```
