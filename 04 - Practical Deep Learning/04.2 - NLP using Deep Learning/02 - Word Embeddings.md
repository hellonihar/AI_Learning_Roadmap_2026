# Word Embeddings

## From One-Hot to Dense Vectors

One-hot vectors are sparse (size = vocabulary), orthogonal (no notion of similarity), and suffer from the curse of dimensionality. Dense word embeddings address all three: they map each word to a low-dimensional, real-valued vector where semantic similarity corresponds to cosine similarity.

```python
# One-hot: shape (vocab_size,) — 99.9% zeros
# Embedding: shape (embed_dim,) — e.g., 300 dense floats
```

## nn.Embedding — The Learnable Lookup Table

PyTorch's `nn.Embedding` is a simple lookup table that stores trainable vectors:

```python
import torch.nn as nn

embedding = nn.Embedding(num_embeddings=10000, embedding_dim=300)
# Forward: (batch, seq_len) indices → (batch, seq_len, 300)
```

Under the hood, `nn.Embedding` is equivalent to `nn.Linear(vocab_size, embed_dim, bias=False)` with one-hot input — but far more efficient. Gradients flow back to the embedding matrix during training, allowing the model to learn task-specific representations.

## Word2Vec

Mikolov et al. (2013) introduced Word2Vec, a family of shallow neural networks that learn embeddings from local context.

### CBOW (Continuous Bag of Words)

Predict the target word from its surrounding context words:

```
Input: context words [w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}]
Output: target word w_t
```

The context embeddings are averaged, then projected through a softmax over the vocabulary. CBOW is fast and works well for frequent words.

### Skip-gram

Predict context words from the target word:

```
Input: target word w_t
Output: context words [w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}]
```

Skip-gram is slower but produces better embeddings for rare words. Both use **negative sampling** (NCE) to avoid computing the full softmax over the vocabulary.

```python
# Conceptual Skip-gram loss (simplified)
# Maximize log σ(u_o · v_c) + Σ log σ(-u_j · v_c) for j ∈ negative_samples
# where v_c = target embedding, u_o = context embedding, u_j = negative sample
```

## GloVe (Global Vectors)

Pennington et al. (2014) argued that Word2Vec underuses corpus statistics. GloVe factorizes the **co-occurrence matrix** (counts of word pairs within a window) using weighted least-squares regression:

```
J = Σ f(X_ij) * (w_i · w̃_j + b_i + b̃_j - log X_ij)^2
```

Where X_ij is the co-occurrence count, f is a weighting function, and w_i, w̃_j are the target/context vectors. The final embedding is often w_i + w̃_j.

GloVe typically produces more consistent embeddings than Word2Vec and is available in pre-trained dimensions (50, 100, 200, 300) trained on Common Crawl or Wikipedia/Gigaword.

## FastText

Bojanowski et al. (2016) extended Word2Vec's Skip-gram to incorporate subword information. Each word is represented as a bag of character n-grams (e.g., "where" → "<wh", "whe", "her", "ere", "re>").

```python
# "where" with n=3: ["<wh", "whe", "her", "ere", "re>"]
# Final embedding = sum of all n-gram embeddings
```

**Advantage**: Can generate embeddings for OOV words by summing their n-gram vectors. Particularly effective for morphologically rich languages.

## Loading Pre-trained Embeddings

```python
import torch
import torch.nn as nn

# GloVe via torchtext
from torchtext.vocab import GloVe
glove = GloVe(name="6B", dim=100)  # 400k words, 100d

# Look up a word
glove["king"]  # tensor of shape (100,)

# Build embedding layer with pre-trained weights
vocab_size, embed_dim = 10000, 100
pretrained = nn.Embedding.from_pretrained(glove.vectors[:vocab_size], freeze=True)
# freeze=True: no gradient updates during training
# freeze=False: fine-tune embeddings for the task
```

For custom loading:
```python
import numpy as np

def load_glove(path, vocab, dim=300):
    embeddings = np.random.randn(len(vocab), dim) * 0.1
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word, *vec = line.split()
            if word in vocab:
                embeddings[vocab[word]] = np.array(vec, dtype=np.float32)
    return torch.tensor(embeddings)
```

## Embedding Visualization

High-dimensional embeddings are visualized by reducing to 2D/3D:

- **t-SNE**: Preserves local structure; good for clusters.
- **PCA**: Linear; captures global variance; faster.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

words = ["king", "queen", "man", "woman", "prince", "princess"]
vectors = torch.stack([glove[w] for w in words]).numpy()
tsne = TSNE(n_components=2, perplexity=5)
coords = tsne.fit_transform(vectors)
plt.scatter(coords[:, 0], coords[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, coords[i])
```

You should see "king" near "queen" and "man" near "woman", with the famous vector analogy: king - man + woman ≈ queen.

## Key Takeaways

- `nn.Embedding` is a differentiable lookup table, the entry point for all neural NLP.
- Word2Vec (CBOW/Skip-gram) uses local context windows with negative sampling.
- GloVe factorizes the global co-occurrence matrix.
- FastText adds subword n-grams for OOV handling and morphological awareness.
- Pre-trained embeddings (GloVe, FastText) are easily loaded via `torchtext` or custom parsers.
- Visualization with t-SNE/PCA reveals semantic and syntactic structure.
