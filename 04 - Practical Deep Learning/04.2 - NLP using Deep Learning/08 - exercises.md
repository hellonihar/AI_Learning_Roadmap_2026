# Exercises

## Exercise 1: Packing Variable-Length Sequences

Implement a collate function that sorts a batch of sequences by length, pads them, and returns the lengths for use with `pack_padded_sequence`. Then write the forward pass of an LSTM classifier using packing.

<details>
<summary>Answer</summary>

```python
def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
    sorted_idx = torch.argsort(lengths, descending=True)
    texts = [texts[i] for i in sorted_idx]
    labels = torch.stack([labels[i] for i in sorted_idx])
    lengths = lengths[sorted_idx]
    padded = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return padded, labels, lengths

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=True)
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden.squeeze(0))
```
</details>

## Exercise 2: Word vs Subword Tokenization

Given the sentence `"I love playing with uncharted territories"`, tokenize it with:
1. Whitespace word tokenizer
2. A character tokenizer
3. Conceptual subword tokenizer (assume merges: "un", "ed", "territor", "ies")

Compare vocabulary sizes and discuss how each handles the rare word "uncharted".

<details>
<summary>Answer</summary>

- Word: `["I", "love", "playing", "with", "uncharted", "territories"]` — vocab ~50k, "uncharted" unlikely in training → `<unk>`.
- Character: `["I", " ", "l", "o", "v", "e", ...]` — vocab ~100, no OOV but very long sequence (38 tokens).
- Subword: `["I", "love", "play", "ing", "with", "un", "chart", "ed", "territories"]` — vocab ~10k, "uncharted" is decomposed into known pieces ["un", "chart", "ed"]. No OOV, reasonable sequence length.

Subword tokenization balances the tradeoff: frequent words stay intact, rare words are handled by composition.
</details>

## Exercise 3: CBOW Embeddings

Implement a simple CBOW model in PyTorch: given a context window of 4 words (2 before, 2 after), predict the target word. Use an embedding layer, average the context embeddings, and project to the vocabulary.

<details>
<summary>Answer</summary>

```python
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, context):
        emb = self.embeddings(context)          # (B, 4, E)
        avg = emb.mean(dim=1)                   # (B, E)
        return self.proj(avg)                   # (B, V)
```

Training uses cross-entropy loss. Negative sampling is used in practice for efficiency.
</details>

## Exercise 4: Role of the CRF Layer

Explain why a per-token softmax classifier fails on NER compared to a CRF. Give a concrete example.

<details>
<summary>Answer</summary>

Per-token softmax makes independent predictions: each label is chosen by argmax, ignoring neighboring labels. Consider the sequence "New York":

- Token-level softmax might predict: New → B-LOC, York → I-PER (valid individually, but I-PER after B-LOC is illegal).
- A CRF learns a transition matrix T where T[B-LOC][I-PER] has a very negative score, making this impossible.

The CRF also learns patterns like: "I-ORG" cannot occur after "O" (must follow "B-ORG"), and "I-LOC" can only follow "B-LOC" or "I-LOC". This structural constraint is essential for span-based evaluation.
</details>

## Exercise 5: Temperature Sampling

Implement a function that given logits of shape `(batch_size, vocab_size)`, applies temperature scaling and samples a token. Compare outputs with τ=0.5, τ=1.0, τ=2.0.

<details>
<summary>Answer</summary>

```python
def temperature_sample(logits, temperature):
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)

# τ=0.5: peaked distribution, almost deterministic, likely argmax token
# τ=1.0: standard softmax, moderate diversity
# τ=2.0: flattened distribution, more uniform, higher diversity

logits = torch.tensor([[10.0, 5.0, 0.0, -5.0]])
print(temperature_sample(logits, 0.5))  # almost always token 0
print(temperature_sample(logits, 1.0))  # mostly token 0, sometimes token 1
print(temperature_sample(logits, 2.0))  # token 0 still most likely, others possible
```
</details>

## Exercise 6: Top-p (Nucleus) Sampling

Implement top-p sampling. Given logits and p=0.9, select the smallest set of tokens whose cumulative probability ≥ p, renormalize, and sample.

<details>
<summary>Answer</summary>

```python
def top_p_sample(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    mask = cumsum - probs > p
    sorted_logits[mask] = float("-inf")
    renormalized = F.softmax(sorted_logits, dim=-1)
    idx = torch.multinomial(renormalized, 1)
    return sorted_indices.gather(-1, idx)
```

Top-p adaptively includes more tokens when the distribution is flat and fewer when it's peaked, giving more control than fixed top-k.
</details>

## Exercise 7: Attention Weight Analysis

Given a trained BiLSTM attention classifier, write code to extract attention weights for a sample sentence and visualize which words the model focused on.

<details>
<summary>Answer</summary>

```python
def visualize_attention(model, text, tokenizer, vocab):
    model.eval()
    tokens = tokenizer.encode(text)
    ids = torch.tensor([[vocab.get(t, vocab["<unk>"]) for t in tokens]])
    with torch.no_grad():
        _, attn_weights = model(ids, torch.tensor([len(tokens)]))
    weights = attn_weights.squeeze(0).squeeze(0).numpy()
    import matplotlib.pyplot as plt
    plt.bar(range(len(tokens)), weights)
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.show()
```

For the sentence "I absolutely loved this movie, it was fantastic!", the attention should assign high weight to "absolutely", "loved", "fantastic" — the sentiment-bearing words.
</details>

## Exercise 8: Span-Based F1 Metric

Write a function that computes span-based F1 for NER predictions. Include span extraction, matching, and precision/recall/F1 computation.

<details>
<summary>Answer</summary>

```python
def extract_spans(label_seq):
    spans = set()
    i = 0
    while i < len(label_seq):
        if label_seq[i].startswith("B-"):
            typ = label_seq[i][2:]
            j = i + 1
            while j < len(label_seq) and label_seq[j] == f"I-{typ}":
                j += 1
            spans.add((typ, i, j - 1))
            i = j
        else:
            i += 1
    return spans

def span_f1(pred_labels, gold_labels):
    pred = extract_spans(pred_labels)
    gold = extract_spans(gold_labels)
    correct = len(pred & gold)
    prec = correct / len(pred) if pred else 0.0
    rec = correct / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}

# Example
pred = ["B-PER", "O", "O", "B-LOC", "I-LOC"]
gold = ["B-PER", "O", "O", "B-LOC", "I-LOC"]
print(span_f1(pred, gold))  # {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
```
</details>
