# RNN for Text Classification

## Architecture Overview

Text classification with RNNs follows a simple blueprint:

```
Input tokens → Embedding → RNN/LSTM → Hidden state → Classifier → Output
```

The RNN processes the sequence token by token, producing a hidden state at each timestep. For classification, we typically use the **final hidden state** (or a pooled version) as a fixed-size representation of the entire sequence, then pass it through a linear classifier.

```python
import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)                # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.rnn(packed)           # hidden: (1, B, H)
        hidden = hidden.squeeze(0)                  # (B, H)
        return self.classifier(hidden)               # (B, C)
```

## Why LSTM over Plain RNN?

A vanilla RNN computes `h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)`. It suffers from **vanishing gradients** — gradients diminish exponentially with sequence length, making it unable to learn long-range dependencies (the "long-term dependency problem").

LSTM introduces a **gating mechanism** (input, forget, output gates) and a **cell state** that can carry information unchanged across hundreds of timesteps:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     # forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     # input gate
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     # output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_c · [h_{t-1}, x_t] + b_c)
h_t = o_t ⊙ tanh(c_t)
```

GRU is a simpler alternative with only two gates (reset and update). In practice, LSTM and GRU perform similarly; LSTM is slightly more expressive.

## Bidirectional RNN

A standard RNN only sees past context. A **bidirectional RNN** runs two RNNs: left-to-right (forward) and right-to-left (backward). Hidden states are concatenated at each timestep:

```python
self.rnn = nn.LSTM(embed_dim, hidden_dim // 2, bidirectional=True)
# hidden shape: (2, B, H/2) → concatenated to (B, H)
```

For classification, final hidden is usually the concatenation of the forward and backward final states: `h = [h_fwd(T); h_bwd(1)]`.

## Packing Sequences for Variable Length

Batched sequences have different lengths due to padding. Without packing, the RNN processes padding tokens, wasting computation and corrupting the hidden state.

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 1. Sort by length (descending) — required by pack_padded_sequence
lengths, perm_idx = lengths.sort(descending=True)
x = x[perm_idx]

# 2. Pack
packed = pack_padded_sequence(embedded, lengths, batch_first=True)

# 3. Run RNN on packed sequence
packed_out, (hidden, cell) = self.rnn(packed)

# 4. Unpack (optional — needed if using all timestep outputs)
output, _ = pad_packed_sequence(packed_out, batch_first=True)

# 5. Restore original order
_, unperm_idx = perm_idx.sort()
output = output[unperm_idx]
```

`pack_padded_sequence` tells the RNN to ignore padding by tracking the actual lengths. Internally, it removes padding tokens and stores them in a `PackedSequence` object. This ensures the RNN only processes real tokens.

## Training Loop

```python
import torch.optim as optim

model = RNNClassifier(vocab_size=10000, embed_dim=100, hidden_dim=256, num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for batch in dataloader:
    texts, labels, lengths = batch
    logits = model(texts, lengths)
    loss = criterion(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Important Design Choices

- **Embedding dimension**: 100–300 for most tasks. 50 for small datasets.
- **Hidden dimension**: 128–512. Larger = more capacity but slower.
- **Number of layers**: 1–2. Deeper RNNs are harder to train and prone to overfitting.
- **Dropout**: Apply dropout to embeddings and between RNN layers (set `dropout=0.3` in nn.LSTM).
- **Padding index**: The `padding_idx` argument in `nn.Embedding` zeroes out the gradient for padding positions.
- **Gradient clipping**: Clip gradients to `max_norm` to prevent explosion in long sequences.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Key Takeaways

- RNN/LSTM processes sequences left-to-right; the final hidden state encodes the entire sequence.
- LSTM's gating mechanism solves the vanishing gradient problem, enabling long-range dependencies.
- Bidirectional RNN captures context from both directions.
- `pack_padded_sequence` is essential for efficient variable-length batch processing.
- Always pair padding with `padding_idx` in Embedding to avoid learning spurious patterns.
