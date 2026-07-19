# NLP with Deep Learning — Cheatsheet

## Tokenization Types

| Type | Vocabulary | OOV | Sequence Length | Example |
|------|-----------|-----|-----------------|---------|
| Word | ~50k | Yes | Short | `["Hello", "world"]` |
| Character | ~100 | No | Long | `["H", "e", "l", "l", "o"]` |
| Subword (BPE) | ~10-50k | No | Medium | `["Hello", "world"]` or `["Hel", "lo"]` |

## nn.Embedding

```python
nn.Embedding(vocab_size, embed_dim, padding_idx=0)
# Forward: (B, T) → (B, T, E)
# padding_idx=0 zeros out gradient for padding positions
```

## LSTM for Text

```python
nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=False, dropout=0.0, batch_first=True)

# Packing for variable length:
packed = pack_padded_sequence(emb, lengths, batch_first=True)
out, (h_n, c_n) = lstm(packed)
out, _ = pad_packed_sequence(out, batch_first=True)

# h_n shape: (num_layers * num_directions, B, H)
# out shape: (B, T, H * num_directions)
```

## Sampling Methods

| Method | Parameter | Behavior |
|--------|-----------|----------|
| Greedy | — | Always argmax → repetitive |
| Temperature | τ ∈ (0, ∞) | Scale logits: `logits / τ`; low τ = greedy, high τ = random |
| Top-k | k (10-100) | Sample from top-k tokens only |
| Top-p (nucleus) | p (0.85-0.95) | Sample from smallest set with cumulative prob ≥ p |
| Beam search | beam_width | Maintain k candidates; best for constrained generation |

## CRF Overview

- **Purpose**: Model dependencies between adjacent labels in sequence labeling.
- **Training**: Maximize log-likelihood of the correct label sequence using forward algorithm (O(T·L²)).
- **Decoding**: Viterbi algorithm finds the highest-scoring path (O(T·L²)).
- **Transition matrix**: Learned, shape (L, L); illegal transitions get −∞.
- **Without CRF**: Per-token softmax ignores label dependencies (e.g., I-PER after B-LOC is possible).

## Text Generation Loop

```python
for _ in range(max_len):
    logits = model(tokens)          # (1, T, V)
    next_logits = logits[:, -1, :]  # last timestep
    probs = softmax(next_logits / temperature)
    next_token = multinomial(probs)
    tokens = cat([tokens, next_token], dim=1)
```

## Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN) / (TP+TN+FP+FN) | Balanced classification |
| Precision | TP / (TP+FP) | Minimize false positives |
| Recall | TP / (TP+FN) | Minimize false negatives |
| F1 Score | 2·P·R / (P+R) | Imbalanced classification |
| Span-F1 | F1 on exact entity spans | NER evaluation |
| Perplexity | exp(CrossEntropyLoss) | Language model quality |
| Cross-entropy | -Σ y·log(ŷ) | Training objective |

## HuggingFace Quick Start

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer("Text here", return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
logits = outputs.logits
```

## Common Hyperparameters

| Parameter | Typical Range |
|-----------|--------------|
| Embedding dim | 100-300 |
| RNN hidden dim | 128-512 |
| RNN layers | 1-3 |
| Dropout | 0.2-0.5 |
| Learning rate | 1e-4 - 1e-3 (Adam) |
| Gradient clip norm | 1.0-5.0 |
| Batch size | 16-128 |
| Max sequence length | 128-512 |
