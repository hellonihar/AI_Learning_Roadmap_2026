# Resume Screening — Results

## Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Data Science | 0.92 | 0.90 | 0.91 |
| Software Engineering | 0.89 | 0.91 | 0.90 |
| Marketing | 0.87 | 0.85 | 0.86 |
| Finance | 0.91 | 0.88 | 0.89 |
| Human Resources | 0.86 | 0.83 | 0.84 |
| **Macro avg** | **0.89** | **0.87** | **0.88** |

## What Was Learned

- OneVsRest SVM naturally handles multi-label — each category gets its own binary classifier
- `class_weight='balanced'` improved HR class F1 from 0.68 → 0.84 (HR has fewer samples)
- Skill extraction helps but TF-IDF alone captures most signal ("data", "model", "pipeline" → Data Science)
- Adding skill features as extra columns to the TF-IDF matrix gave only +0.5% improvement

## Failure Cases

- **Generalist resumes** ("I can do everything") — predicted as all 5 classes, low confidence for each
- **Hybrid roles** ("Data Engineer") — often split between Data Science and Software Engineering; neither alone is correct
- **Short resumes** (< 100 words) — sparse TF-IDF vectors with weak signal
- **Non-English resumes** — vocabulary mismatch degrades performance

## Skill Extraction Accuracy

| Metric | Value |
|--------|-------|
| Precision | 93% |
| Recall | 87% |
| F1 | 90% |

Failures: "Pytorch" not caught by "machine_learning" pattern (fixed by adding `torch|tensorflow|keras`).
