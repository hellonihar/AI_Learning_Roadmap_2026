# Text Generation

## Autoregressive Generation

Language models generate text **autoregressively**: each token is predicted conditioned on the previous tokens.

```
P(w_1, w_2, ..., w_T) = Π_t P(w_t | w_1, ..., w_{t-1})
```

At inference, we sample from the predicted distribution over the vocabulary at each step and feed the sampled token back as input.

```python
def generate(model, start_tokens, max_len=50):
    model.eval()
    tokens = start_tokens
    for _ in range(max_len):
        logits = model(tokens)                    # (1, T, V)
        probs = torch.softmax(logits[:, -1, :], dim=-1)  # last timestep
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens
```

## Sampling Strategies

Greedy decoding (always pick argmax) produces dull, repetitive text. Stochastic sampling strategies introduce diversity.

### Temperature Sampling

Scale logits before applying softmax:

```
P(w) = exp(logit(w) / τ) / Σ exp(logit' / τ)
```

- **τ → 0**: Greedy (deterministic, repetitive).
- **τ = 1**: Standard softmax.
- **τ → ∞**: Uniform (random).

```python
def sample_with_temperature(logits, temperature=1.0):
    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

Low temperature (0.5–0.7) produces more focused, coherent text. High temperature (1.2–1.5) produces more creative, unpredictable output.

### Top-k Sampling

Sample only from the k tokens with the highest probabilities:

```python
def top_k_sampling(logits, k=40):
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(top_k_values, dim=-1)
    sampled_index = torch.multinomial(probs, num_samples=1)
    return top_k_indices.gather(-1, sampled_index)
```

This prevents sampling from the long tail of improbable tokens. Typical k values: 40–100.

### Top-p (Nucleus) Sampling

Instead of a fixed k, select the smallest set of tokens whose cumulative probability mass exceeds p:

```python
def top_p_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    mask = cumsum - probs > p
    sorted_logits[mask] = float("-inf")
    probs = torch.softmax(sorted_logits, dim=-1)
    sampled_index = torch.multinomial(probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_index)
```

Top-p adaptively chooses the number of candidates. If the distribution is peaked, few tokens are considered; if it's flat, many are. Typical p: 0.85–0.95.

### Beam Search

Beam search maintains k candidate sequences (beams) and expands them in parallel:

```
Step 1: ["the"] (score -2.1), ["a"] (score -2.5)
Step 2: ["the cat"] (-4.0), ["the dog"] (-4.2), ["a cat"] (-4.5), ["a dog"] (-4.7)
         → keep top-2: ["the cat"] (-4.0), ["the dog"] (-4.2)
```

Beam search is better for tasks like machine translation where the output should be high-probability. It is worse for open-ended generation (produces generic, repetitive text).

```python
def beam_search(model, start_tokens, beam_width=5, max_len=50):
    beams = [(start_tokens, 0)]  # (sequence, log_prob)
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            logits = model(seq.unsqueeze(0))[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_scores, topk_tokens = torch.topk(log_probs.squeeze(0), beam_width)
            for s, t in zip(topk_scores, topk_tokens):
                candidates.append((torch.cat([seq, t.unsqueeze(0)]), score + s))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    return beams[0][0]
```

## Training a Word-Level Language Model with LSTM

```python
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)              # (B, T, E)
        out, _ = self.lstm(emb)              # (B, T, H)
        logits = self.fc(out)                # (B, T, V)
        return logits
```

**Objective**: cross-entropy between predicted next-token distribution and actual next token.

```python
def train_step(model, batch, optimizer, criterion):
    # batch shape: (B, T) — tokens include <bos> at position 0
    input_seq = batch[:, :-1]   # (B, T-1)
    target_seq = batch[:, 1:]   # (B, T-1): shifted right
    logits = model(input_seq)   # (B, T-1, V)
    loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    return loss.item()
```

**Perplexity**: `exp(loss)`. Lower is better. Perplexity measures how "surprised" the model is by the data — a perplexity of 50 means the model is as confused as if it were choosing uniformly among 50 tokens.

## Key Takeaways

- Language models generate text autoregressively: each token depends on previous ones.
- Temperature, top-k, and top-p control the randomness-diversity tradeoff.
- Beam search is best for constrained tasks; sampling is best for open-ended generation.
- LSTM language models predict the next token from all previous tokens via the hidden state.
- Perplexity measures generation quality (lower is better).
- Always use gradient clipping when training LSTM-based LMs.
