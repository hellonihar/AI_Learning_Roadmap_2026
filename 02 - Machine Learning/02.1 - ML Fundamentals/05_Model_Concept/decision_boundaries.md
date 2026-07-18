# Decision Boundaries (Visual Intuition)

The decision boundary is the "line" (or surface) that separates different classes in the feature space.

## 2D Intuition

Imagine two features (x-axis, y-axis) and two classes (red dots, blue dots):

```
Linear Boundary (Logistic Regression):
    |
    |  red  red  red
    |  red  red  red
    |____\___________   ← straight line separating red from blue
    |      \  blue blue
    |       \ blue blue
    |________________

Non-Linear Boundary (Decision Tree / Neural Net):
    |
    |  red  red  blue
    |  red  |    blue
    |______|___ blue
    |         |  blue
    |  blue   |  red
    |_________|________
                 ↑ wiggly boundary that twists to separate classes
```

- **Linear model**: draws a straight line — works when classes are roughly separable by a line
- **Non-linear model**: draws a curved/irregular boundary — can separate complex patterns

## Examples

1. **Spam vs not-spam**: 2D features (word count, exclamation count). A linear boundary might separate clear cases. But real spam detection needs a highly non-linear boundary (spammers use sophisticated patterns).
2. **Medical diagnosis**: 2 features (age, blood pressure). Healthy vs at-risk. The boundary might be non-linear because risk increases sharply after certain thresholds.
3. **Handwritten digit recognition**: Features are 784 dimensions (28×28 pixels). The decision boundary is a 783-dimensional hyperplane or curved surface impossible to visualize — but the intuition is the same: separating "3"s from "8"s in pixel space.

## Why It Matters

- **Simple boundary**: Interpretable, stable, low variance, but underfits complex patterns
- **Complex boundary**: Flexible, fits complex patterns, but overfits noise and is harder to interpret

The goal: find a boundary complex enough to capture the true pattern but simple enough to generalize.
