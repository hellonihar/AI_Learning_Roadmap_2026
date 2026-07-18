# Problem Statement — Online Course Recommendation (Skill‑Based + Popularity)

## Business Context
A MOOC platform wants to guide learners to the next best course. Many users are new (cold‑start) with no history. The platform needs a skill‑aware recommender that maps a user's stated goals to relevant courses while also handling returning users via past enrollments.

## Problem Type
**Cold‑start + sequential multi‑class recommendation.**  
- **Input:** User ID, past enrollments, skill tags, course metadata.  
- **Output:** Top‑N courses for each user.  
- **Approach:** Two‑tower — (a) popularity + skill‑match for cold users, (b) collaborative embedding for warm users.

## Success Metrics
- **Precision@k** (k = 5) — relevant courses in top‑5. Target ≥ 45%.  
- **Coverage** — % of course catalog ever recommended (combat popularity bias).  
- **Cold‑start Hit Rate** — same metric computed only for users with < 3 enrollments.
