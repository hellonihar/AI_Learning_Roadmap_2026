# What Is a Loss Function?

The loss function measures **how wrong** the model's predictions are. Training = minimizing the loss.

## Loss vs Error vs Metric

| Term | What It Is | Example |
|---|---|---|
| **Loss** | Differentiable function used for **training** | Mean squared error |
| **Error** | Raw difference between prediction and actual | Predicted 100, actual 90 → error = 10 |
| **Metric** | Human-interpretable score for **evaluation** | Accuracy, F1-score |

- **Loss** is for the optimizer (needs gradients). Train with loss.
- **Error** is the building block of loss.
- **Metric** is for humans (doesn't need to be differentiable).

## Why We Minimize Loss

The loss is a single number summarizing how bad the model is. By making it smaller, we push the model toward better predictions.

```
loss = average of (how wrong each prediction is)
training = keep adjusting model parameters to make this number as small as possible
```

## Key Intuition

We choose a loss function that matches our problem:

- **Regression** (predict a number): use MAE or MSE
- **Classification** (predict a category): use log loss (cross-entropy)

The choice matters because different loss functions penalize errors differently.
