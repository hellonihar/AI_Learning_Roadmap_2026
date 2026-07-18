# Semi-Supervised Learning

Mix of **small labeled** data + **large unlabeled** data.

## Why Use It?

Labeling data is expensive (requires human experts). Unlabeled data is cheap or free. Semi-supervised learning leverages both.

## How It Works

1. Train a model on the small labeled set
2. Use that model to make **pseudo-labels** on unlabeled data
3. Keep the high-confidence predictions as new labeled data
4. Retrain on the augmented dataset
5. Repeat until convergence

## Examples

1. **Medical imaging**: A radiologist labels 500 X-rays as "pneumonia" or "healthy". You have 50,000 unlabeled X-rays. The model learns from the 500 labeled + uses structure in the 50K unlabeled to improve accuracy.
2. **Web page classification**: Hand-label 1,000 pages as "sports" / "finance" / "tech". Crawl 1M unlabeled pages — the model uses similarity in text embeddings to propagate labels.
3. **Speech recognition**: Transcribe 10 hours of audio (expensive). Use 10,000 hours of unlabeled audio — the model learns acoustic patterns from unlabeled data, improving recognition on the labeled subset.

## Common Algorithms

- Self-training (pseudo-labeling)
- Co-training (multiple views of data)
- Label propagation (graph-based)
- Consistency regularization (FixMatch, MixMatch)
