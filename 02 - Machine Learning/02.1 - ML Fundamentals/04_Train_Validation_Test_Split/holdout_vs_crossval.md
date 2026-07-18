# Hold-Out Validation vs Cross-Validation

## Hold-Out Validation

Split data **once** into train / validation / test sets. Simple and fast.

**When to use**: Large datasets (>1M samples), training is expensive, quick iteration needed.

**Example**: Training a large neural network on 10M images. A single train/val/test split (80/10/10) is practical — cross-validating 5 times would mean training 5 models, which is prohibitively expensive.

## Cross-Validation (k-Fold)

Split data into **k folds**. Train on k-1 folds, validate on the remaining fold. Repeat k times, each time holding out a different fold. **Average** the k results.

```
Fold 1: [val] [train] [train] [train] [train]
Fold 2: [train] [val] [train] [train] [train]
Fold 3: [train] [train] [val] [train] [train]
Fold 4: [train] [train] [train] [val] [train]
Fold 5: [train] [train] [train] [train] [val]
```

**When to use**: Small datasets (<10K samples), when you need a reliable estimate of performance, or when comparing algorithms.

**Example**: You have 500 samples. A single 80/20 hold-out gives only 100 validation points — noisy estimate. 5-fold cross-validation uses each sample for validation exactly once, giving a much more reliable performance estimate.

## Comparison

| Aspect | Hold-Out | Cross-Validation |
|---|---|---|
| **Speed** | Fast (train once) | Slow (train k times) |
| **Variance of estimate** | Higher | Lower |
| **Best for** | Large datasets, deep learning | Small datasets, model comparison |
| **Risk** | Unlucky split → misleading results | More computationally expensive |
| **Common values** | 80/10/10 or 70/15/15 | k=5 or k=10 |
