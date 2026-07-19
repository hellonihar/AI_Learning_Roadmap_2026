# Sequence Labeling

## Task Definition

Sequence labeling assigns a label to each token in a sequence. Common tasks:

- **Named Entity Recognition (NER)**: Person, Organization, Location, etc.
- **Part-of-Speech (POS) Tagging**: Noun, Verb, Adjective, etc.
- **Chunking**: Base noun phrases, verb phrases.

```
Token:  [John]  [lives]  [in]  [New]  [York]  [.]
NER:    B-PER   O        O     B-LOC  I-LOC   O
POS:    NNP     VBZ      IN    NNP    NNP     .
```

Labels use **BIO encoding** (Begin, Inside, Outside) to mark span boundaries. Without BIO, the model cannot distinguish adjacent entities of the same type (e.g., "New York" vs "New" and "York" separately).

## BiLSTM-CRF Architecture

The BiLSTM-CRF is the classic deep learning architecture for sequence labeling.

```
      [B-PER]   [O]      [O]      [B-LOC]  [I-LOC]  [O]        ← CRF layer
         ↑       ↑        ↑         ↑        ↑        ↑
      [  h_1  ] [ h_2 ]  [ h_3 ]  [ h_4 ]  [ h_5 ]  [ h_6 ]   ← BiLSTM outputs
         ↑       ↑        ↑         ↑        ↑        ↑
      [  x_1  ] [ x_2 ]  [ x_3 ]  [ x_4 ]  [ x_5 ]  [ x_6 ]   ← Embeddings
```

### BiLSTM Encoder

A bidirectional LSTM reads the sequence in both directions. The output at each position is the concatenation of forward and backward hidden states, enriched with context from the entire sentence.

```python
class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.bilstm(emb)     # (B, T, H)
        return out
```

The BiLSTM produces emissions `E ∈ R^(T × L)` where `E[t][l]` is the score of label `l` at position `t`. A naive approach applies softmax independently per token — but this ignores dependencies between adjacent labels.

## Role of the CRF Layer

A **Conditional Random Field** (CRF) layer models the transition between labels. It learns that "I-LOC" cannot follow "B-PER", or that "I-LOC" must follow "B-LOC" or "I-LOC". This is the key advantage over per-token softmax.

The CRF defines a **score** for a label sequence `y = (y_1, ..., y_T)` given emissions `E`:

```
score(y) = Σ_t E[t][y_t] + Σ_t T[y_{t-1}][y_t]
```

Where `T[i][j]` is a learned transition score from label i to label j.

**Training**: Maximize the log-likelihood of the correct label sequence:

```
log P(y|E) = score(y) - log Σ_{y'} exp(score(y'))
```

The denominator sums over all possible label sequences, computed efficiently with the **forward algorithm** (dynamic programming, O(T · L²)).

**Decoding**: Find the highest-scoring label sequence with **Viterbi decoding** (also O(T · L²)):

```python
def viterbi_decode(emissions, transitions):
    # emissions: (T, L), transitions: (L, L)
    # Returns: best label sequence
    ...
```

Without a CRF, the model might predict "B-LOC I-PER" — a valid per-token argmax but an invalid entity. The CRF's transition matrix penalizes such illegal transitions during both training and decoding.

## Implementation Notes

```python
class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels):
        super().__init__()
        self.encoder = BiLSTMEncoder(vocab_size, embed_dim, hidden_dim)
        self.emission = nn.Linear(hidden_dim, num_labels)
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        self.transitions.data[START_TAG, :] = -10000  # cannot transition from START
        self.transitions.data[:, STOP_TAG] = -10000    # cannot transition to STOP

    def forward(self, x, labels=None):
        emissions = self.emission(self.encoder(x))
        if labels is not None:  # training: return negative log-likelihood
            return -self.log_likelihood(emissions, labels)
        else:                   # inference: return best path
            return self.viterbi_decode(emissions)
```

## Evaluation: Span-Based F1

For NER, accuracy at the token level is misleading (most tokens are "O"). The standard metric is **span-based F1**:

1. Extract predicted entity spans (type + start/end position).
2. Compare with ground-truth spans.
3. A span is correct only if both the type and boundaries match exactly.

```python
def extract_spans(labels):
    spans = []
    i = 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            entity_type = labels[i][2:]
            j = i + 1
            while j < len(labels) and labels[j] == f"I-{entity_type}":
                j += 1
            spans.append((entity_type, i, j - 1))
            i = j
        else:
            i += 1
    return spans

def span_f1(pred_spans, gold_spans):
    correct = len(set(pred_spans) & set(gold_spans))
    precision = correct / len(pred_spans) if pred_spans else 0
    recall = correct / len(gold_spans) if gold_spans else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"precision": precision, "recall": recall, "f1": f1}
```

## Key Takeaways

- Sequence labeling assigns a label per token; BIO encoding marks entity boundaries.
- BiLSTM captures bidirectional context; CRF models label dependencies.
- CRF's transition matrix enforces valid label sequences (e.g., I-X must follow B-X or I-X).
- The forward algorithm computes the partition function; Viterbi decodes the best path.
- Evaluation uses span-based F1, not token accuracy, to measure entity-level performance.
