# Text Feature Engineering (Intro Level)

## Why Text Needs Special Handling

Models process numbers, not words. Text must be converted to numerical vectors while preserving meaning. Unlike structured data (where each column has a consistent meaning), text is unstructured, variable-length, and context-dependent.

## Text Cleaning Basics

Before any numerical encoding, clean the text to reduce noise.

**Common steps:**
1. **Lowercasing**: "Hello" → "hello", "HELLO" → "hello". Reduces vocabulary size.
2. **Remove punctuation**: "hello!" → "hello". Punctuation usually doesn't add signal for basic models.
3. **Remove stop words**: "the", "a", "is", "and" — common words that appear in every document and add little signal.
4. **Stemming**: "running", "runner", "ran" → "run". Reduces words to their root form.
5. **Remove special characters**: URLs, HTML tags, emojis (or handle separately).

**Example**: Raw review: "This product is AMAZING!!! I've been using it for 3 months and it's soo goood :)"
Cleaned: "this product amazing i ve been using it for 3 month and it soo goood"

## Tokenization (Conceptual)

Split text into individual units (tokens), usually words or subwords.

- **Word tokenization**: "I love ML" → ["I", "love", "ML"]
- **Subword tokenization (BPE/WordPiece)**: Breaking rare words into common subwords — "unbelievably" → ["un", "believe", "ably"]. Used in modern LLMs to handle any word.

**Example**: In spam detection, tokenization turns each email into a list of words. The model learns which words are spammy ("free", "winner", "guaranteed").

## Bag of Words (BoW) Intuition

Create a vector where each position represents a unique word in the vocabulary, and the value is the count (or binary presence) of that word in the document.

```
Document 1: "I love cats"     → [1, 1, 1]  (assuming vocab = ["cats", "I", "love"])
Document 2: "I love dogs"     → [1, 0, 1]
```

- **Pro**: Simple, fast, interpretable. A weight of +2.5 on "fraud" means the word "fraud" is strongly associated with the target class.
- **Con**: Loses word order ("not good" and "good not" look identical), creates huge sparse vectors (vocabulary of 50K words), treats all words equally.

**Examples:**
1. **Spam detection**: Bag of words works well — "congratulations", "winner", "free" are strong indicators regardless of sentence structure. A spam model needs just a few hundred words to be effective.
2. **Sentiment analysis**: "amazing" vs "terrible" — word presence alone captures general sentiment. But "not amazing" gets misclassified as positive. Order matters here, which BoW can't capture.
3. **Topic classification (sports vs politics)**: Word counts work great. "goal", "match", "score" → sports. "election", "policy", "vote" → politics. Order doesn't matter for topic detection.

## TF-IDF Intuition

Term Frequency — Inverse Document Frequency: downweights common words, upweights rare but informative words.

**Formula intuition:**
- **TF**: How often does this word appear in this document? (higher = more important in *this* doc)
- **IDF**: How rare is this word across all documents? (rarer = more discriminative)
- **TF-IDF = TF × IDF**: Words that appear frequently in THIS document but rarely in OTHER documents get the highest score.

**Examples:**
1. In a set of news articles, "the" appears in every document (low IDF) → near-zero TF-IDF score. "election" appears in only 12% of documents (higher IDF), and 8 times in one article (high TF) → high TF-IDF score. The model learns "election" is important for classifying that article.
2. In customer reviews, "good" appears in 80% of reviews (low IDF) → not very discriminative. But "flimsy" appears in only 5% of reviews (high IDF), and 3 times in one review (high TF) → strong signal for negative reviews.

## When Feature Engineering Gives Way to Embeddings

For complex text understanding, word meaning matters more than word counts. Embeddings (word2vec, GloVe, BERT) map words to dense vectors where similar words are close together.

```
"king" — "man" + "woman" ≈ "queen"  (word2vec relationships)
```

- BoW/TF-IDF: Good for simple classification, fast, interpretable
- Embeddings: Better for semantics, context, rare words; needed for complex tasks

**Example**: "The movie was not bad" — BoW sees ["not", "bad"], both negative words → predicts negative. BERT understands "not bad" means "good" → predicts positive. Only embeddings capture negation and context.

## What Is Deferred to NLP Section

This section covers basic text feature engineering for tabular ML tasks. The following topics belong to a dedicated NLP section and are not covered here:
- Word embeddings (word2vec, GloVe)
- Transformer models (BERT, GPT)
- Sequence models (LSTM, GRU)
- Named entity recognition (NER)
- Part-of-speech tagging
- Language-specific preprocessing
- Advanced tokenization (BPE, SentencePiece)
- Document similarity and semantic search
