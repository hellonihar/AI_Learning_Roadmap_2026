# Ignoring Baseline Models Pitfall

## The Mistake

Jumping directly to complex models (neural networks, gradient boosting) without establishing a simple baseline.

## Why Baselines Matter

Without a baseline:
- You don't know if your complex model is actually better than simple alternatives
- You can't quantify the value added by complexity
- You might have a bug in your pipeline (a broken model can be worse than predicting the mean)

## Common Baselines

| Problem | Baseline |
|---|---|
| Regression | Predict the **mean** of the training target |
| Classification (balanced) | Predict the **most frequent class** |
| Classification (imbalanced) | Predict **always negative** (or always positive) |
| Time-series | **Last value** (persistence model) |
| Time-series (seasonal) | **Same season last year** |
| Recommendation | **Most popular items** (globally) |
| Ranking | **Sort by popularity** |

## Examples

1. **House price prediction**: The naive baseline (predict average price) gives MAE = $120K. Your complex gradient boosting model gives MAE = $45K. That's a $75K improvement — worth the complexity. Without the baseline, you wouldn't know if $45K is good or just OK.
2. **Fraud detection**: Baseline (predict "not fraud" always) gives 99.9% accuracy but 0% recall. Your deep learning model gives 95% recall at 80% precision. The baseline reveals that accuracy is meaningless — you must use precision-recall metrics.
3. **LLM chatbot for customer support**: Baseline (keyword rules) resolves 40% of tickets. RAG with GPT-4 resolves 75%. Is the improvement worth the 100x cost increase? Baseline quantifies the tradeoff: 35% more resolutions for 100x more cost. Maybe 60% with a smaller model is the sweet spot.

## Rule of Thumb

Always implement the simplest possible baseline first. If the baseline is competitive with your complex model, the complex model isn't adding value. If the baseline is terrible, you have a clear target to beat. If the baseline beats your complex model, you have a bug.
