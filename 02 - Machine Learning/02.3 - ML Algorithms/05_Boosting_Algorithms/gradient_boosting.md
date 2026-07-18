# Gradient Boosting

Trains a sequence of trees where **each tree predicts the residual errors** of the previous ensemble.

## How It Works

1. Start with a simple prediction (mean of target)
2. Compute residuals (actual — prediction)
3. Train a shallow tree to predict the **residuals**
4. Add the tree's prediction (scaled by learning rate) to the ensemble
5. Recompute residuals
6. Repeat

```
Round 1: predict mean → residuals = actual — mean
Round 2: tree learns residuals → new prediction = mean + η × tree_prediction
Round 3: tree learns new residuals → ...
```

## Learning Rate (η)

The most important hyperparameter. Controls how much each tree contributes.

| Learning Rate | Trees Needed | Effect |
|---|---|---|
| High (0.1-0.3) | Few (100-500) | Fast, risk of overfitting |
| Low (0.01-0.05) | Many (1000-5000) | Slower, better generalization |

**Rule**: Lower learning rate + more trees almost always beats higher learning rate + fewer trees.

## Key Hyperparameters

| Parameter | Effect |
|---|---|
| n_estimators | Number of trees (increase if LR is low) |
| learning_rate (η) | Step size for each tree |
| max_depth | Usually 3-6 (deeper = more interaction) |
| subsample | Row sampling (0.5-0.8 = stochastic GB) |
| min_samples_leaf | Prevents overfitting |

## Examples

1. **Kaggle tabular competitions**: Gradient Boosting (especially XGBoost/LightGBM) dominates structured data competitions. It consistently outperforms random forests on medium-to-large datasets.
2. **Insurance claim prediction**: Features = driver age, vehicle type, location, history. Gradient Boosting captures complex interactions (young driver + sports car + urban area = high risk) better than linear models.
3. **Web click-through rate**: Hundreds of features (user, ad, context). Gradient Boosting handles mixed feature types, missing values, and non-linear relationships without extensive preprocessing.
