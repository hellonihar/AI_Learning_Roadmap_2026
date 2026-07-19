# Text Preprocessing for Deep Learning

## Overview

Raw text is unstructured and high-dimensional. Deep learning models require numerical, fixed-size inputs. Text preprocessing bridges this gap by converting strings into tensors. The pipeline typically involves: normalization → tokenization → vocabulary building → numericalization → padding/truncation.

## Tokenization

### Word Tokenization

The simplest approach: split text on whitespace and punctuation. Libraries like `spacy` and `nltk` provide rule-based tokenizers.

```python
import spacy
nlp = spacy.load("en_core_web_sm")
tokens = [tok.text for tok in nlp("Deep learning is fun.")]
# ["Deep", "learning", "is", "fun", "."]
```

**Pros**: Intuitive, preserves word boundaries. **Cons**: Large vocabulary (50k+), OOV (out-of-vocabulary) words, morphological variations ("run", "running", "ran" are distinct tokens).

### Character Tokenization

Split text into individual characters (or byte pairs for Unicode).

```python
list("hello")  # ["h", "e", "l", "l", "o"]
```

**Pros**: Tiny vocabulary (~100), no OOV. **Cons**: Long sequences, weak semantic signal per timestep.

### Subword Tokenization

The dominant approach in modern NLP. Combines the vocabulary efficiency of character-level with the semantic richness of word-level. The key idea: frequent words stay as units, rare words are decomposed into smaller pieces.

#### Byte-Pair Encoding (BPE)

BPE is a data compression algorithm adapted for tokenization. It iteratively merges the most frequent pair of adjacent characters/subwords.

**Algorithm:**
1. Start with character vocabulary (all unique bytes/characters).
2. Count all adjacent pairs in the corpus.
3. Merge the most frequent pair into a new token.
4. Repeat until desired vocabulary size is reached.

```python
# Conceptual BPE merge: ("e", "s") occurs 1000 times → merge to "es"
# Subsequent merges may produce "es", "th", "the", "ing"...
```

BPE is used by GPT models. Since merges are deterministic, any string can be segmented. The algorithm is greedy and does not consider probability — it always picks the highest-frequency merge.

#### WordPiece

Similar to BPE but merges based on likelihood gain rather than frequency:

```
score = P(merged) / (P(left) * P(right))
```

WordPiece is used by BERT. It uses a greedy merge criterion: at each step, merge the pair that maximizes the likelihood of the training data. This often produces slightly more linguistically meaningful subwords than BPE.

#### SentencePiece

A language-independent tokenizer that treats the input as a raw Unicode stream (including spaces). It implements both BPE and Unigram Language Model tokenization. Key features:
- **No pretokenization**: Space is treated as a regular character (tokenized as `▁`).
- **Directly models raw text**: No dependency on language-specific tokenizers.
- Used by T5, ALBERT, XLNet.

SentencePiece with the **Unigram** algorithm: starts with a large seed vocabulary and iteratively removes tokens that minimize the likelihood loss.

```python
import sentencepiece as spm
spm.SentencePieceTrainer.train("--input=data.txt --model_prefix=m --vocab_size=8000")
sp = spm.SentencePieceProcessor(model_file="m.model")
sp.encode("Hello world", out_type=str)  # ["▁Hello", "▁world"]
```

## Vocabulary Building

After tokenization, a vocabulary maps tokens to unique integer IDs.

```python
from collections import Counter

def build_vocab(tokenized_corpus, min_freq=2, specials=["<pad>", "<unk>", "<bos>", "<eos>"]):
    counter = Counter()
    for tokens in tokenized_corpus:
        counter.update(tokens)
    vocab = {token: idx for idx, token in enumerate(specials)}
    for token, freq in counter.items():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab
```

Special tokens:
- `<pad>` (0): Padding to equalize sequence lengths.
- `<unk>`: Unknown/OOV tokens during inference.
- `<bos>`/`<eos>`: Beginning/End of sequence markers for generation.

## Padding, Truncation, and Bucketing

Batched training requires equal-length sequences.

- **Padding**: Short sequences are extended with `<pad>` tokens (usually ID 0).
- **Truncation**: Long sequences are clipped to `max_len` (e.g., 512 for BERT).
- **Bucketing**: Sort sequences by length and create batches of similar-length sequences to minimize padding waste.

```python
from torch.nn.utils.rnn import pad_sequence

# sequences is a list of 1D LongTensors
batch = pad_sequence(sequences, batch_first=True, padding_value=0)
# shape: (batch_size, max_len_in_batch)
```

## torchtext

`torchtext` provides built-in tokenizers, vocabularies, and dataloaders:

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(map(tokenizer, corpus), specials=["<unk>"], max_tokens=10000)
vocab.set_default_index(vocab["<unk>"])
```

## Custom Preprocessing Pipeline

A complete preprocessing function:

```python
def preprocess(text, tokenizer, vocab, max_len=128):
    tokens = tokenizer(text.lower())
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    ids = ids[:max_len] + [0] * (max_len - len(ids))  # truncate + pad
    return torch.tensor(ids, dtype=torch.long)
```

## Key Takeaways

- Subword tokenization (BPE/WordPiece/SentencePiece) is the standard for deep learning NLP.
- Word tokenization is simple but suffers from large vocabularies and OOV issues.
- Character tokenization avoids OOV but produces very long sequences.
- Vocabulary building, padding, and truncation are essential for batching.
- Tools like `torchtext` and `sentencepiece` simplify the preprocessing pipeline.
- Always handle `<unk>` tokens gracefully during inference.
