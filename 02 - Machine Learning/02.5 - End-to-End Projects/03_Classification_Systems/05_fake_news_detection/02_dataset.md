# Dataset — Fake News Detection

## Source
- **LIAR Dataset** (William Yang Wang) — 12,836 short statements labeled for truthfulness.
- **Kaggle Fake News** (~20,000 articles) — collected from 244 news sites.
- Combined and binarized: `true/mostly-true` → 0, `false/pants-fire` → 1.

## Size
- ~30,000 labeled articles after deduplication and binarization.

## Features
- `text`: article body or statement (varies by source).
- `subject` (optional): politics, health, science, etc.

## Target
- `label`: 0 (real/trustworthy) / 1 (fake/untrustworthy).

## Class Balance
- Real: ~53%
- Fake: ~47%
- Nearly balanced — no resampling needed.

## Known Challenges
- **LIAR statements are short** (mean 20 words); Kaggle articles are long (mean 800 words).
- **Domain mismatch**: LIAR is political; Kaggle has broader topics.
- **Temporal generalization**: model trained on 2016 data fails on COVID-era misinformation.
- **Adversarial examples**: changing "not" to "never" or adding qualifiers can flip predictions.

## Linguistic Feature Candidates
- Sentiment polarity, subjectivity, readability scores (Flesch-Kincaid).
- Count of ALL-CAPS words, exclamation marks, quotes per sentence.
- Ratio of assertive vs. hedging language.
