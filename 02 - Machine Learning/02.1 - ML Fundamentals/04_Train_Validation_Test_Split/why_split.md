# Why Splitting Data Matters

If you train and evaluate on the same data, you get an **overly optimistic** estimate of performance — the model memorized the answers instead of learning to generalize.

## The Three Sets

### Training Set (~60-80%)
- The model **learns** from this data
- Sees every example multiple times during training
- Loss is computed and weights are updated

### Validation Set (~10-20%)
- Used to tune **hyperparameters** (learning rate, model size, regularisation)
- Evaluated after each training epoch to check for overfitting
- The model never trains on this data, but it influences decisions
- **Leakage risk**: if you tune too many hyperparameters based on this set, you implicitly overfit to it

### Test Set (~10-20%)
- Used **once** at the very end to report final performance
- The model should never see this data before final evaluation
- It's held back until you're done with all experimentation

## The Core Rule

**The test set should be touched as few times as possible — ideally once.**

Every time you look at test set performance and then change your model, you're leaking information from the test set into your training decisions.

## Examples

1. **Medical diagnosis model**: You train on 80% of patient records, validate on 10% (tune threshold), and keep 10% hidden. When the model finally sees the test set, its 94% accuracy is your best estimate of real-world performance. If you'd instead used all 100% for training and tested on the same data, you'd report 99% but fail in the clinic.
2. **House price prediction**: Training on 2020–2024 data, validating on early 2025, testing on late 2025. Time-based split (not random) prevents the model from learning future patterns. If you randomly shuffled, the model could cheat by seeing 2025 trends during training.
3. **LLM benchmark contamination**: When a model is trained on internet data that contains benchmark answers, its "test set" performance is inflated. This is why new benchmarks are constantly created — the test set must truly be unseen.
