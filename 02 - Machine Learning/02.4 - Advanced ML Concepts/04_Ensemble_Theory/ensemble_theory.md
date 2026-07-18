# Ensemble Theory (Beyond Algorithms)

Ensembles combine multiple models to produce a stronger predictor. The key insight: if models make **different errors**, averaging them cancels the errors out.

## Bagging vs Boosting

The two foundational ensemble approaches:

| | Bagging | Boosting |
|---|---|---|
| **Goal** | Reduce **variance** | Reduce **bias** |
| **Training** | Models trained in **parallel** | Models trained **sequentially** |
| **Each model** | High bias, low variance (deep trees) | Low bias, high variance (shallow trees) |
| **Data** | Bootstrap samples (random subsets) | Weighted data (focus on mistakes) |
| **Example** | Random Forest | XGBoost |
| **Error pattern** | Uncorrelated errors → averaging cancels | Errors of earlier models get fixed by later ones |

### Why Bagging Reduces Variance

Individual models overfit in different ways. Their errors are uncorrelated. Averaging uncorrelated errors with mean 0 reduces variance by ~1/n.

```
Model 1: predicts 100, actual 95 → error = +5
Model 2: predicts 92, actual 95 → error = -3
Model 3: predicts 97, actual 95 → error = +2
Ensemble (avg): 96.3, error = +1.3  (less than any individual)
```

### Why Boosting Reduces Bias

Each new model focuses on the mistakes of the ensemble so far. The ensemble progressively captures patterns the earlier models missed. The combined model can represent much more complex functions than any individual weak learner.

## Stacking

Train **different types** of models, then train a **meta-model** to combine their predictions.

**How it works:**
1. Split training data into k folds
2. For each fold, train base models on the other k-1 folds, predict on the held-out fold
3. Collect all out-of-fold predictions as **meta-features**
4. Train a **meta-model** (usually Logistic Regression or a simple linear model) on the meta-features
5. At test time: base models predict → meta-model combines them into final prediction

```
Base models:    [Random Forest] [XGBoost] [CatBoost] [kNN]
                       ↓            ↓         ↓        ↓
Meta-features:  [RF_pred]  [XGB_pred] [Cat_pred] [kNN_pred]
                       ↓            ↓         ↓        ↓
Meta-model:          Logistic Regression (or any simple model)
                       ↓
                Final prediction
```

**Why it works**: Different algorithms capture different patterns. RF captures non-linear interactions, XGBoost captures gradient-focused patterns, kNN captures local structure. The meta-model learns who to trust when.

**Examples:**
1. **Kaggle competition winner**: Many winning solutions stack 5-10 diverse models. A simple logistic regression or ridge regression meta-model is often best — prevents overfitting to meta-features.
2. **Credit risk**: Stack logistic regression (interpretable), random forest (robust), gradient boosting (accurate). The meta-model combines their strengths. If RF and XGBoost agree but LR disagrees, the meta-model learns to follow the complex models for this case.
3. **Medical diagnosis**: Stack CNN (image features), XGBoost (lab results), Logistic Regression (patient history). Each base model processes a different data modality. The meta-model fuses all modalities.

## Blending

Similar to stacking but uses a **hold-out validation set** instead of cross-validation. Simpler but uses less data.

1. Split data into train + hold-out
2. Train base models on train
3. Predict on hold-out → these are meta-features
4. Train meta-model on hold-out predictions

**Stacking vs Blending**: Stacking uses cross-validation (more robust, more computation). Blending is simpler (single hold-out, less robust).

## Voting Classifiers

Combine predictions from multiple models by **voting** — no meta-model needed.

### Hard Voting
Each model votes for a class. The class with the most votes wins.

```
RF → "fraud", XGB → "fraud", kNN → "legitimate", LR → "fraud"
Result: 3-1 vote → "fraud"
```

### Soft Voting
Each model outputs class probabilities. Average the probabilities. Predict the class with the highest average probability.

```
          Fraud    Legit
RF:       0.85     0.15
XGB:      0.92     0.08
kNN:      0.30     0.70
LR:       0.75     0.25
Average:  0.71     0.30  → "fraud"
```

Soft voting is generally better — it accounts for confidence, not just final labels.

**When to use voting**: Simple, robust, needs no extra training. Use when you have 3+ diverse models with similar performance.

## Comparison Table

| Technique | Meta-model? | Complexity | When to Use |
|---|---|---|---|
| Voting | No | Low | Quick ensemble, baseline |
| Bagging | No | Low-Medium | Variance reduction |
| Boosting | No | Medium | Bias reduction |
| Stacking | Yes | High | Maximum performance, diverse models |
| Blending | Yes | Medium | Simpler alternative to stacking |
