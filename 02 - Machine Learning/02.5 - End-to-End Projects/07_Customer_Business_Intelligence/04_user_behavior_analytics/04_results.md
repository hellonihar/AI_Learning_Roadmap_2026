# User Behavior Analytics — Results

## Funnel Conversion

| Step | Users | Conversion | Drop-off |
|------|-------|------------|----------|
| Visit | 48,200 | 100% | — |
| Signup | 31,330 | 65% | 35% |
| Onboarding Start | 22,080 | 46% | 19% |
| Onboarding Complete | 8,850 | 18% | **28%** |
| First Feature Use | 7,610 | 16% | 2% |
| First Payment | 2,850 | 6% | 10% |

## Weekly Cohort Retention (select cohorts)

| Cohort Week | W1 | W2 | W3 | W4 | W8 | W12 |
|-------------|----|----|----|----|----|-----|
| Week 1 | 100% | 42% | 35% | 28% | 18% | 14% |
| Week 2 | 100% | 40% | 32% | 26% | 17% | 12% |
| Week 5 | 100% | 38% | 30% | 22% | 15% | 10% |

## Engagement Scoring

| Segment | % Users | Sessions/Week | Feature Depth | 30-Day Retention |
|---------|---------|---------------|---------------|------------------|
| High (Heads) | 15% | 7.2 | 5.8 | 91% |
| Medium (Hearts) | 35% | 3.1 | 3.2 | 62% |
| Low (Tails) | 50% | 0.8 | 1.1 | 18% |

## What Was Learned

- Biggest drop-off is onboarding completion (28% drop in conversion) — users who finish onboarding are 4x more likely to convert to paid
- Using 3+ features in the first session correlates with 70% Week-4 retention (vs. 20% for 0-1 features)
- "Feature active" has only 2% drop-off from onboarding — users who finish onboarding almost always try a feature
- Early cohorts (Jan) retained better than later cohorts (Mar) — likely due to seasonal effects or product changes

## Failure Cases

- **Identified anonymous users** — 22% of events have no user_id (no-cookie browsers); funnel analysis misses these entirely
- **Session timeout choice** — 30-min gap vs. 15-min changes session count by ~15%; arbitrary choice impacts metrics
- **Self-serve vs. sales-assisted** — Users who start with a demo call have a completely different funnel; mixing them obscures insights
