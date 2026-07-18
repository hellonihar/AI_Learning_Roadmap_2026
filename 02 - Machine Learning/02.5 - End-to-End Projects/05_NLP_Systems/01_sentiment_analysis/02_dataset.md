# Sentiment Analysis — Dataset

## Source
[Large Movie Review Dataset (IMDB)](https://ai.stanford.edu/~amaas/data/sentiment/) — 50,000 reviews split evenly between positive and negative.

## Size
| Split | Samples |
|-------|---------|
| Train | 25,000 |
| Test  | 25,000 |

## Features
- **Vocabulary size**: ~75,000 unique tokens after cleaning
- **TF-IDF unigrams**: 75k dimensions (sparse)
- **TF-IDF bigrams**: ~1.2M dimensions (sparse)
- **Document length**: 50–2,500 words per review
- **Target**: 0 (negative) / 1 (positive)

## Challenges
- Sarcasm and negations ("not bad" = positive, but contains "bad")
- Domain-specific slang
- Long-range dependencies missed by bag-of-words
- Class balance is perfect (50/50) in this dataset but rarely in production
