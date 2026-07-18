# News Article Categorisation — Results

## Performance

| Classifier | Features | Test Accuracy | Macro F1 |
|------------|----------|--------------|----------|
| Linear SVC | TF-IDF (10k unigrams) | 84.3% | 0.83 |
| Linear SVC | TF-IDF (10k unigrams+bigrams) | **86.1%** | **0.85** |
| MultinomialNB | TF-IDF (10k unigrams) | 80.5% | 0.79 |

## Per-Class F1 Highlights

| Best Classes | F1 | Worst Classes | F1 |
|-------------|-----|--------------|-----|
| rec.sport.hockey | 0.96 | talk.religion.misc | 0.71 |
| rec.motorcycles | 0.95 | misc.forsale | 0.73 |
| sci.space | 0.93 | comp.sys.mac.hardware | 0.76 |

## What Was Learned

- Linear SVC is the go-to for high-dimensional text classification — fast, accurate, sparse-friendly
- Bigrams help but diminishing returns beyond 10k features
- LDA reveals coherent topics: Topic 3 = "game team players season" (sports), Topic 7 = "key encryption chip clipper" (crypto)
- Removing headers/footers is critical — without it, accuracy artificially inflates to 95%+ (model learns metadata, not content)

## Failure Cases

- **comp.sys.ibm.pc.hardware ↔ comp.sys.mac.hardware**: both discuss drivers, RAM, monitors — hard to separate without brand names
- **talk.religion.misc ↔ alt.atheism**: overlapping vocabulary (god, belief, christian)
- **Short posts** (< 50 words): sparse vectors, often misclassified as "misc.forsale" (default catch-all)
- LDA topics show some overlapping; 20 topics for 20 newsgroups is too many — 15 is more coherent
