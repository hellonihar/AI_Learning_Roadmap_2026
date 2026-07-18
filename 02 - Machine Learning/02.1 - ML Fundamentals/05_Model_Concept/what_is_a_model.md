# What Is a Model?

A model is a simplified representation of reality that maps **inputs → outputs** using learned parameters.

## Parameters vs Hyperparameters

### Parameters
- Learned **automatically** from data during training
- Internal to the model (weights, biases)
- Examples: coefficients in linear regression, weights in a neural network

### Hyperparameters
- Set **manually** before training
- Control how training works
- Examples: learning rate, number of trees in random forest, network depth

| | Parameters | Hyperparameters |
|---|---|---|
| Who sets them? | Training algorithm | Human (or search) |
| When set? | During training | Before training |
| Examples | Weight w₁₁, bias b₂ | Learning rate, k in k-NN |
| How many? | Millions–billions | 5–50 |

## Hypothesis Space Intuition

The hypothesis space is the **set of all possible models** your algorithm can learn.

- Linear regression hypothesis space: all possible straight lines
- Neural network hypothesis space: all possible functions the network architecture can represent

More complex models = larger hypothesis space = more flexible but harder to search.

**Example**: Trying to fit data with a line (small hypothesis space) vs a 10th-degree polynomial (large hypothesis space). The line is limited but stable. The polynomial can fit perfectly but may be unrealistic.

## What Training Does

Training searches the hypothesis space to find the model that best fits the data. The algorithm minimizes a loss function, which measures how wrong the model's predictions are. Starting from random parameters, the model iteratively adjusts to reduce error — moving through the hypothesis space toward better solutions.

**Example**: Training a linear regression model: start with random slope and intercept, compute prediction error, adjust slope/intercept to reduce error, repeat. After convergence, the final line is your model: `price = 250 * sqft + 50,000`.
