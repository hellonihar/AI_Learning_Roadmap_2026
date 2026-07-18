# Implementation — Session‑Based Music Recommender

## 1. Session Padding + Negative Sampling

```python
import numpy as np

def pad_session(tracks, max_len=20):
    if len(tracks) > max_len:
        tracks = tracks[-max_len:]
    return tracks + [0] * (max_len - len(tracks))

def sample_negatives(pos_track, all_tracks, n_neg=100):
    negs = np.random.choice(all_tracks, n_neg, replace=False)
    return [t for t in negs if t != pos_track][:n_neg]
```

## 2. GRU4Rec Model (Simplified)

```python
import tensorflow as tf

class GRU4Rec(tf.keras.Model):
    def __init__(self, n_items, emb_dim=100, gru_units=100):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(n_items + 1, emb_dim, mask_zero=True)
        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=False)
        self.out = tf.keras.layers.Dense(n_items, activation='softmax')

    def call(self, x):
        x = self.emb(x)
        x = self.gru(x)
        return self.out(x)
```

## 3. Top‑K Evaluation (Recall@k)

```python
def recall_at_k(model, test_seqs, k=20):
    hits = 0
    total = 0
    for seq, next_track in test_seqs:
        probs = model.predict(seq[np.newaxis, :], verbose=0)[0]
        top_k = np.argsort(probs)[-k:][::-1]
        if next_track in top_k:
            hits += 1
        total += 1
    return hits / total

print(f'Recall@20: {recall_at_k(model, test_data, k=20):.3f}')
```
