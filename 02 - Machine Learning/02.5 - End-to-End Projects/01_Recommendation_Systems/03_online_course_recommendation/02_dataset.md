# Dataset — Coursera Course Reviews / Kaggle

## Source
Kaggle — [Coursera Course Reviews Dataset](https://www.kaggle.com/datasets/khusheekapoor/coursera-courses-dataset-2021) + synthetic enrollment logs.

## Size
| Entity      | Count |
|-------------|-------|
| Courses     | ~3,500 |
| Users       | ~50,000 |
| Enrollments | ~800,000 |
| Skills      | ~200 (unique tags) |

## Features
- **Course:** `course_id`, `title`, `skills` (list), `difficulty`, `rating`, `enrollment_count`.  
- **User:** `user_id`, `enrolled_courses` (sequence), `stated_skills` (from onboarding survey).  
- **Interaction:** `user_id`, `course_id`, `rating` (1–5), `completed` (bool), `timestamp`.

## Target
Binary relevance — `completed == True` or `rating >= 4`.

## Known Challenges
- **Cold‑start >>> 50%** of users have no history (new sign‑ups).  
- **Course churn:** courses appear / disappear every semester.  
- **Skill ambiguity:** "Python" and "Python programming" as separate tags.  
- **Selection bias:** high‑rated courses get more enrollments, inflating popularity.
