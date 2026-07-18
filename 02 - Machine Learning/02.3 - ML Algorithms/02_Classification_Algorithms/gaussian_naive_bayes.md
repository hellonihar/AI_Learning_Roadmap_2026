# Gaussian Naive Bayes

Variant of Naive Bayes for **continuous features** that are assumed to follow a **normal (Gaussian) distribution** within each class.

## How It Works

For each class, the model learns the mean and variance of each feature. At prediction time, it computes the probability of the feature value under the Gaussian distribution for each class.

```
P(feature | class) = Gaussian(feature; μ_class, σ_class)
```

## When to Use

- Continuous features that are roughly normally distributed within each class
- Low-to-moderate number of features
- When speed matters more than accuracy

## Examples

1. **Iris flower classification**: Features (sepal length, sepal width, petal length, petal width) are roughly normally distributed per species. GaussianNB achieves ~95% accuracy on this classic dataset.
2. **Medical screening**: Continuous vitals (heart rate, blood pressure, temperature, oxygen level) for classifying "healthy" vs "at risk." Fast enough for real-time ICU monitoring.
3. **Sensor fault detection**: Continuous sensor readings (vibration, temperature, pressure) assumed normally distributed under normal operation. Deviations from expected Gaussian indicate potential faults.
