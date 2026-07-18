# Classification Metrics (Intuition)

## Accuracy

`(correct predictions) / (total predictions)`

**Example**: Out of 100 emails, model correctly classifies 95 as spam or not-spam. Accuracy = 95%.

**Problem**: Accuracy breaks on imbalanced data.

## The Imbalanced Data Problem

**Example**: Fraud detection. 99.9% of transactions are legitimate. A model that predicts "not fraud" for every transaction gets 99.9% accuracy but catches **zero** fraud. Useless.

## Precision

"Of the positives I predicted, how many were correct?"

`precision = true_positives / (true_positives + false_positives)`

**Example**: Spam filter flags 20 emails as spam. Only 12 are actually spam (8 are important emails misflagged). Precision = 12/20 = 60%.

## Recall

"Of the actual positives, how many did I catch?"

`recall = true_positives / (true_positives + false_negatives)`

**Example**: There were 15 actual spam emails in the inbox. Model caught 12. Recall = 12/15 = 80%.

## F1-Score

Harmonic mean of precision and recall: `2 * (precision * recall) / (precision + recall)`

A single number that balances both. Favors models that are good at both precision and recall.

**Example**: Model A has precision=90%, recall=50% → F1 = 0.64. Model B has precision=70%, recall=75% → F1 = 0.72. Model B is more balanced, so F1 prefers it.

## Examples

1. **Cancer screening**: 1% of patients have cancer. A model with 99% accuracy might miss 50% of cancer cases (low recall). For screening, high recall is critical — you'd rather have 100 false positives (biopsies) than miss one cancer. Precision matters less at the screening stage.
2. **YouTube recommendation**: Showing a bad video to a user is mildly annoying (low precision cost). Missing a great video they'd love is a lost opportunity (high recall value). The recommender can afford lower precision to improve recall.
3. **Self-driving pedestrian detection**: Both precision and recall are critical. False positive (braking for a ghost): jarring but safe. False negative (missing a real pedestrian): catastrophic. The system needs high recall, but also reasonable precision to avoid constant false braking.
