# Model Selection & Comparison

Choosing the right model involves balancing performance, interpretability, speed, and business constraints.

## Bias vs Variance Driven Selection

The fundamental trade-off guides model choice:

| | High Bias (Underfitting) | High Variance (Overfitting) |
|---|---|---|
| **Training error** | High | Very low |
| **Validation error** | High | High (gap with training) |
| **Symptoms** | Model too simple, misses patterns | Model too complex, memorizes noise |
| **Fix** | More complex model, more features, reduce regularization | Simpler model, more data, increase regularization |

### Decision Flow

```
Training error high?
  ├── Yes → Underfitting (high bias)
  │     └── Increase model complexity, add features, reduce regularization
  └── No → Check validation error
        ├── Validation error also low? → Good fit! ✅
        └── Validation error significantly higher? → Overfitting (high variance)
              └── Simplify model, add regularization, more data, reduce features
```

**Example**: Training a decision tree:
- Depth=3: Train Acc=0.72, Val Acc=0.71 → high bias (both low). Increase depth.
- Depth=10: Train Acc=0.98, Val Acc=0.80 → high variance (large gap). Decrease depth or prune.
- Depth=6: Train Acc=0.88, Val Acc=0.86 → good balance.

## Metric-Driven Model Choice

The business problem determines the metric, and the metric determines the best model.

| Business Goal | Primary Metric | Best Model Type |
|---|---|---|
| Maximize overall accuracy | Accuracy | Balanced models (RF, XGBoost) |
| Minimize false negatives | Recall | Models with class weighting, lower threshold |
| Minimize false positives | Precision | Conservative models, higher threshold |
| Rank correctly | NDCG, MAP | Ranking models (LambdaRank, XGBoost ranker) |
| Explain decisions | Coefficient interpretability | Linear/Logistic Regression, Decision Tree |
| Minimize error magnitude | MAE / RMSE | Regression models (GBM, RF) |
| Fast inference (<1ms) | Latency | Linear models, shallow trees, distilled NNs |

### Model Comparison Table

| Model | Interpretability | Accuracy (tabular) | Training Speed | Inference Speed | Handles Non-Linear | Handles High-Dim |
|---|---|---|---|---|---|---|
| Logistic Regression | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ✅ |
| Decision Tree | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ❌ |
| Random Forest | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ✅ |
| XGBoost/LightGBM | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ✅ |
| SVM (RBF) | ⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | ✅ | ✅ |
| Neural Network | ⭐ | ⭐⭐⭐⭐⭐ (large data) | ⭐⭐ | ⭐⭐⭐ | ✅ | ✅ |

## Statistical Significance Testing

A model that scores 0.835 vs 0.832 might not be truly better — the difference could be random noise.

### When to Test

- Comparing two models on the same test set
- Deciding whether an "improvement" is real or due to chance
- Reporting results in research or production benchmarks

### Common Tests

| Test | When to Use |
|---|---|
| **McNemar's test** | Comparing classification models (paired nominal) |
| **Wilcoxon signed-rank** | Comparing regression models (paired, non-parametric) |
| **5×2 CV paired t-test** | Comparing algorithms across multiple cross-validation runs |
| **Paired bootstrap** | Comparing metrics on a test set (resample with replacement) |

### Practical Advice

- For most production work: run 5-fold CV, report mean ± std. If the means differ by > 2× the std, the difference is likely real.
- For rigorous comparison: use nested cross-validation with a statistical test.
- Always ask: "Is the improvement worth the added complexity?" A 0.1% accuracy gain that adds 10× inference cost is rarely worth it.

## Practical Selection Framework

```
1. Define the metric that matters to the business (not just ML metrics)
2. Establish a simple baseline (mean predictor, linear model)
3. Train 3-5 diverse candidates (e.g., Linear, RF, XGBoost, LightGBM)
4. Compare on validation metric + training time + inference time + interpretability
5. Check if the best model is significantly better than the second-best
6. Choose the simplest model that meets requirements
```

## Examples

1. **Real-time fraud detection**: Business needs: <10ms inference, catch 80%+ fraud, explain each flag to investigators. Winner: XGBoost with SHASH + Decision Tree as backup. XGBoost hits accuracy target at 2ms inference. SHAP explains each prediction. The decision tree (slower, less accurate) serves as fallback if the primary model fails.
2. **Medical diagnosis assistant**: Business needs: high recall (don't miss disease), interpretable, regulator-approved. Logistic Regression achieved 88% recall vs XGBoost's 91%. But LR's coefficients are directly interpretable by doctors. The 3% recall gain doesn't justify the black-box risk. LR is selected.
3. **Ad click-through rate**: Business needs: 1M predictions/second, any accuracy gain directly increases revenue. LightGBM trains in 2 hours on 100M rows, inferes in 50μs per sample. XGBoost is 2× slower to infer. LightGBM's 0.01% accuracy loss is worth millions in saved compute costs.
