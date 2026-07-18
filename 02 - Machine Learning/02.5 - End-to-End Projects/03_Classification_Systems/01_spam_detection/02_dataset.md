# Dataset — SMS & Email Spam

## Source
- **SMS Spam Collection** (UCI) — 5,574 SMS messages, labeled ham/spam.
- **Enron Email Dataset** — ~33,000 emails, hand-labeled by researchers.
- Combined: ~38,000 documents after deduplication.

## Features
- Raw message text, no pre-structured fields.
- Target: `label` — 0 (ham), 1 (spam).

## Class Balance
- Ham: ~86%
- Spam: ~14%
- Mildly imbalanced but not extreme.

## Known Challenges
- Vocabulary drift — spam tactics evolve weekly (new URLs, obfuscated words).
- Short messages (SMS mean = 15 tokens) vs. long emails (mean = 200+ tokens).
- Encoding issues in the SMS corpus (some non-ASCII characters).
- Overlap with legitimate marketing emails — ambiguous cases.

## Preprocessing
- Lowercase, strip punctuation, remove non-alphanumeric tokens.
- Lemmatization (optional; no strong gain in short-text tasks).
- Custom tokenizer that preserves URLs, email addresses, and numbers as single tokens.
