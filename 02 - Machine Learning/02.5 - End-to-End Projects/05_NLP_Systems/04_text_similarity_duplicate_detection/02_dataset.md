# Text Similarity / Duplicate Detection — Dataset

## Source
[Quora Question Pairs](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) — 400,000 labelled question pairs. 37% are duplicates.

## Size
| Split | Pairs | Duplicates |
|-------|-------|------------|
| Train | 300,000 | ~110,000 |
| Test  | 100,000 | ~37,000 |

## Features
- **Vocabulary size**: ~100,000 unique tokens across both questions
- **Similarity features** (engineered, 3 main groups):
  - **TF-IDF cosine similarity**: each question vectorised, cosine computed
  - **Jaccard similarity**: (intersection / union) of token sets
  - **Word Mover's Distance (WMD)**: using word2vec embeddings (GoogleNews 300-d)
  - **Basic features**: length diff, common word count, longest common substring ratio
- **Target**: 0 (not duplicate) / 1 (duplicate)

## Challenges
- **Paraphrasing**: "How to learn Python?" vs "Best way to learn Python programming"
- **Word order matters**: "What is AI?" vs "AI, what is it?" — same bag of words, different structure
- **Tricky negatives**: "What is the best phone?" vs "What is the best laptop?" — high lexical overlap, different intent
- WMD is computationally expensive (O(n³) naive; O(p² log p) with RWMD)
