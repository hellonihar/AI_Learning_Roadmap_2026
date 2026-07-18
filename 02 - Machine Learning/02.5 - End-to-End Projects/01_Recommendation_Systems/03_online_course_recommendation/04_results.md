# Results — Online Course Recommendation

## Expected Metrics
| Strategy                     | Precision@5 | Cold‑Start P@5 | Coverage |
|------------------------------|-------------|----------------|----------|
| Popularity only              | 22.4%       | 22.4%          | 6.1%     |
| Skill‑TF-IDF only            | 31.8%       | 35.2%          | 52.3%    |
| Two‑tower (collaborative)    | 47.2%       | 18.5%          | 28.7%    |
| **Hybrid (skill + two‑tower)** | **49.5%** | **38.1%**      | **48.6%** |

## Key Findings
- Skill‑TF‑IDF dramatically improves cold‑start precision and coverage.  
- Two‑tower alone fails on new users but excels on warm users (P@5 > 55%).  
- Blending weights (0.7 skill / 0.3 collaborative) needed tuning per user segment.

## Failure Cases
- **Overly broad skills** (e.g., "Data Science") match too many courses, diluting precision.  
- **Noisy onboarding skills** — users check random tags; elicitation quality matters.  
- **Course duplication** — multiple skill‑identical courses (same topic, different university) cause redundancy in top‑k.
