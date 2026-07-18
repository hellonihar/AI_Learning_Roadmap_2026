# Text Similarity / Duplicate Detection — Results

## Performance

| Feature Set | ROC AUC | Note |
|-------------|---------|------|
| TF-IDF cosine only | 0.745 | Best single feature |
| Jaccard only | 0.682 | |
| WMD only | 0.731 | On 5k subset |
| All 6 features | **0.831** | Full 300k train set |
| WMD baseline (no WMD) | 0.818 | With 5 other features on 300k |

## Feature Importance (LR Coefficients)

| Feature | Weight | Impact |
|---------|--------|--------|
| TF-IDF cosine | +1.42 | Strongest |
| Common words count | +0.89 | |
| Jaccard | +0.65 | |
| LCS ratio | +0.41 | |
| Length difference | −0.52 | Inversely correlated |
| WMD | −0.38 | Inversely correlated (lower distance = more similar) |

## What Was Learned

- TF-IDF cosine alone captures most of the signal; other features add ~8% relative AUC improvement
- WMD is theoretically appealing but adds only ~1.3% AUC over the cheaper features — costly for marginal gain
- Length difference and common word counts are cheap features that boost recall for long-form duplicates
- "How do I X?" vs "How to X?" — high TF-IDF overlap but word order varies; cosine handles this well

## Failure Cases

- **Opposite meaning, similar phrasing**: "Why is Python slow?" vs "Why is Python fast?" — high similarity, not duplicate
- **Short questions** (< 5 words): "iPhone?" vs "iPhone 14?" — sparse vectors, near-zero cosine
- **Cross-language**: English vs transliterated Hindi (common on Quora) — vocabulary mismatch
- **Synonym-heavy**: "automobile" vs "car" — no lexical overlap, TF-IDF cosine = 0, but they are duplicates
