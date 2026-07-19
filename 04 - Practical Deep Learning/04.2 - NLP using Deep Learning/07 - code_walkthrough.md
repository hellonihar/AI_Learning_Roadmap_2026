# Code Walkthrough: BiLSTM Sentiment Classifier with Attention

A complete PyTorch implementation of a sentiment classifier using subword tokenization and a BiLSTM with attention.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import math
```

## 1. BPE Subword Tokenizer (from scratch)

```python
class BPE:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.tokens = set()

    def get_stats(self, words):
        pairs = {}
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def merge_vocab(self, pair, words):
        new_words = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        for word, freq in words.items():
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = freq
        return new_words

    def fit(self, corpus):
        words = {}
        for text in corpus:
            word = " ".join(list(text.lower())) + " </w>"
            words[word] = words.get(word, 0) + 1

        self.tokens = set(c for w in words for c in w.split())

        for i in range(self.vocab_size - len(self.tokens)):
            pairs = self.get_stats(words)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges[best] = i
            words = self.merge_vocab(best, words)

        self.tokens = set(c for w in words for c in w.split())

    def encode(self, text):
        text = text.lower()
        symbols = list(text) + ["</w>"]
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            mergeable = {p: self.merges.get(p, float("inf")) for p in pairs}
            best = min(mergeable, key=mergeable.get)
            if mergeable[best] == float("inf"):
                break
            i = pairs.index(best)
            symbols = symbols[:i] + ["".join(best)] + symbols[i + 2:]
        return [s for s in symbols if s != "</w>"]
```

## 2. Dataset

```python
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.texts[idx])
        ids = [self.vocab.get(t, self.vocab["<unk>"]) for t in tokens]
        ids = ids[:self.max_len]
        length = len(ids)
        ids = ids + [self.vocab["<pad>"]] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long), length
```

## 3. BiLSTM with Attention

```python
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs):
        scores = self.attn(lstm_outputs).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=-1).unsqueeze(1)  # (B, 1, T)
        context = torch.bmm(weights, lstm_outputs).squeeze(1)  # (B, H)
        return context, weights


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim // 2,
                              bidirectional=True, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))                       # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True)                           # (B, T, H)
        context, attn_weights = self.attention(lstm_out)            # (B, H)
        logits = self.classifier(self.dropout(context))             # (B, C)
        return logits, attn_weights
```

## 4. Training

```python
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for texts, labels, lengths in train_loader:
            texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
            logits, _ = model(texts, lengths)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        preds, true = [], []
        with torch.no_grad():
            for texts, labels, lengths in val_loader:
                texts, labels, lengths = texts.to(device), labels.to(device), lengths.to(device)
                logits, _ = model(texts, lengths)
                preds.extend(logits.argmax(dim=-1).cpu().tolist())
                true.extend(labels.cpu().tolist())
        print(f"  Val Acc: {sum(p==t for p,t in zip(preds,true))/len(true):.4f}")
```

## 5. Evaluation

```python
def evaluate(model, loader):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    preds, true = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for texts, labels, lengths in loader:
            texts, labels = texts.to(device), labels.to(device)
            lengths = lengths.to(device)
            logits, _ = model(texts, lengths)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            true.extend(labels.cpu().tolist())
    return {
        "accuracy": accuracy_score(true, preds),
        "f1": f1_score(true, preds, average="binary")
    }
```

## 6. Full Pipeline

```python
def run():
    texts = [
        "I absolutely loved this movie, it was fantastic!",
        "Terrible film, worst I have ever seen, complete waste of time.",
        "Great performances and a compelling story.",
        "Boring and predictable, I fell asleep halfway through.",
    ] * 50  # small synthetic dataset
    labels = [1, 0, 1, 0] * 50

    # Build BPE and vocabulary
    bpe = BPE(vocab_size=200)
    bpe.fit(texts)

    vocab = {"<pad>": 0, "<unk>": 1}
    for t in bpe.tokens:
        vocab[t] = len(vocab)

    dataset = SentimentDataset(texts, labels, bpe, vocab)
    train_size = int(0.8 * len(dataset))
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size])

    def collate_fn(batch):
        texts, labels, lengths = zip(*batch)
        texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
        return texts, torch.stack(labels), torch.tensor(lengths)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn)

    model = BiLSTMClassifier(vocab_size=len(vocab), embed_dim=128, hidden_dim=128)
    train_model(model, train_loader, val_loader, epochs=5)
    print(evaluate(model, val_loader))

if __name__ == "__main__":
    run()
```

## Key Components Breakdown

1. **BPE tokenizer**: Learns merge rules from the corpus, encodes any text into subword tokens.
2. **SentimentDataset**: Encodes text → subword IDs → padded tensor with length tracking.
3. **Attention**: Weighted sum of BiLSTM outputs, learns which tokens are most informative.
4. **BiLSTM**: Bidirectional LSTM captures full context; `pack_padded_sequence` ignores padding.
5. **Classifier**: Linear layer on the attention-weighted context vector.
6. **Training**: AdamW optimizer, gradient clipping, CrossEntropyLoss.
7. **Evaluation**: Accuracy and binary F1 via sklearn.
