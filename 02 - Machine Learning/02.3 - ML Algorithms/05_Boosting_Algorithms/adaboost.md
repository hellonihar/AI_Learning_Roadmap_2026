# AdaBoost

**Adaptive Boosting** — trains a sequence of weak learners, each focusing on the mistakes of the previous one.

## How It Works

1. Train a weak model (usually a shallow tree / stump) on the data
2. Increase the weight of **misclassified samples**
3. Train the next model on the reweighted data
4. Repeat
5. Final prediction = weighted vote of all models

## Key Intuition

Each new model focuses on the hard cases the previous models got wrong. The ensemble becomes progressively better at the difficult edge cases.

## Weak Learners

Typically **decision stumps** (depth-1 trees). Each stump looks at one feature and makes a simple decision. Alone they're weak (~51% accuracy), but boosting combines them into a strong model.

## Learning Rate vs Number of Estimators

| High learning rate | Low learning rate |
|---|---|
| Few stumps needed | Many stumps needed |
| Faster training | Slower training |
| Overfitting risk | Better generalization |

## Examples

1. **Face detection (Viola-Jones)**: The original real-time face detector used AdaBoost with Haar-like features. Each weak learner detected a simple pattern (e.g., "eye region is darker than cheek region"). Allowed the first real-time face detection on consumer cameras.
2. **Customer churn**: Shallow decision stumps each capture one churn signal (e.g., "logins < 5 → churn risk"). AdaBoost refines by focusing on customers who were misclassified — the hard-to-predict cases.
3. **Text classification**: Each stump tests presence of a single word. AdaBoost learns which words matter most and combines them into a strong classifier.
