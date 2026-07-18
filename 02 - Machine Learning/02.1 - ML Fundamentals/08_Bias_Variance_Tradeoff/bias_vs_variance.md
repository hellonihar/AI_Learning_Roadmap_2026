# Bias vs Variance Intuition

## The Core Tradeoff

**Bias**: Error from **overly simplistic assumptions**. The model can't capture the true pattern.
**Variance**: Error from **excessive sensitivity to training data**. The model learns noise as if it were signal.

## Visual Intuition (Target Analogy)

Bias = how far your shots are from the bullseye (on average)
Variance = how spread out your shots are

```
High Bias, Low Variance   Low Bias, High Variance   Low Bias, Low Variance
(Tight cluster, far away) (Scattered, on average)   (Tight cluster, on target)
      X X                      X                           X
       X                       X X X X                   X   X
                               X                           X X
```

## Underfitting vs Overfitting

| | Underfitting | Overfitting |
|---|---|---|
| **What** | Model is too simple | Model is too complex |
| **Bias** | High | Low |
| **Variance** | Low | High |
| **Training error** | High | Very low (near zero) |
| **Test error** | High | High |
| **Fix** | More complex model, more features | Regularization, more data, simpler model |

## Model Complexity vs Data Size

```
                    ↑
Test Error          |  🟢 (sweet spot)
                    |   \
                    |    \    ↗ Overfitting region
                    |     \  /
                    |      \/
 Underfitting ←     |      /\
   region           |     /  \
                    |    /    \
                    |   /      ↘ Overfitting worsens
                    |  /
                    | /
                    |/
                    +———————————————→ Model Complexity
```

- Too simple: high bias, underfits
- Too complex: high variance, overfits
- Sweet spot: balanced complexity

## Examples

1. **House price prediction**: A linear model (high bias) says price is strictly proportional to sqft — misses neighborhood effects, school quality, renovation quality. A 20th-degree polynomial (high variance) perfectly fits every training house but predicts wildly for new houses. The sweet spot is something like a random forest with depth 10.
2. **Facial recognition**: A template-matching approach (high bias) only recognizes faces if they're perfectly centered and lit. A deep network with no regularization (high variance) memorizes all 10K training faces but fails on a person wearing sunglasses. The sweet spot: a regularized CNN that learns general face features.
3. **Stock prediction**: A model that always predicts "up 0.1%" (high bias, underfits) is stable but useless. A model that perfectly fits past price movements (high variance, overfits) fails catastrophically in live trading because it learned random noise patterns. The balance is a model that captures genuine trends but ignores noise.
