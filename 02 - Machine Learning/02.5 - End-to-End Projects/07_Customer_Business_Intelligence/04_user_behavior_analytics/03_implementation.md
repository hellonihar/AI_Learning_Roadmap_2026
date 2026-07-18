# User Behavior Analytics — Implementation

## Steps

1. **Sessionization** — Sort events by user + timestamp; mark new session if gap > 30 min; compute session-level metrics
2. **Funnel Analysis** — Define activation funnel steps; compute conversion and drop-off per step
3. **Cohort Analysis** — Group users by registration week; compute retention (active users in each subsequent week)
4. **Engagement Scoring** — Aggregate feature usage, session frequency, depth; score users as High/Medium/Low
5. **Actionable Insights** — Identify features correlated with retention

## Key Code

```python
# Sessionization
events = events.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

events['prev_timestamp'] = events.groupby('user_id')['timestamp'].shift(1)
events['gap_seconds'] = (events['timestamp'] - events['prev_timestamp']).dt.total_seconds()

# New session if gap > 30 minutes or first event
session_timeout = 30 * 60
events['new_session'] = (
    events['prev_timestamp'].isna() |
    (events['gap_seconds'] > session_timeout)
).astype(int)
events['session_id'] = events.groupby('user_id')['new_session'].cumsum()

# Session-level features
sessions = events.groupby(['user_id', 'session_id']).agg(
    start_time=('timestamp', 'min'),
    end_time=('timestamp', 'max'),
    event_count=('event_name', 'count'),
    unique_features=('feature_name', 'nunique'),
    pages_viewed=('page_url', 'nunique')
).reset_index()

sessions['duration_sec'] = (sessions['end_time'] - sessions['start_time']).dt.total_seconds()
```

```python
# Funnel analysis
funnel_steps = [
    ('signup_visit', 'visit'),
    ('signup_complete', 'signup'),
    ('onboarding_step_1', 'onboarding_start'),
    ('onboarding_step_5', 'onboarding_complete'),
    ('feature_active', 'first_feature_use'),
    ('payment', 'first_payment'),
]

funnel_data = []
for col, label in funnel_steps:
    # % of users who reached this step
    reached = events[events['event_name'] == col]['user_id'].nunique()
    funnel_data.append({'step': label, 'users': reached})

funnel_df = pd.DataFrame(funnel_data)
funnel_df['conversion'] = funnel_df['users'] / funnel_df['users'].iloc[0] * 100
funnel_df['dropoff'] = funnel_df['conversion'].diff().abs().fillna(0)

print(funnel_df)
```

```python
# Weekly cohort retention
users['registration_week'] = users['signup_date'].dt.isocalendar().week
cohorts = users.groupby('registration_week')['user_id'].nunique()

cohort_data = []
for week, cohort_users in cohorts.items():
    for w in range(week, 13):
        active = active_weeks[
            (active_weeks['user_id'].isin(cohort_users)) &
            (active_weeks['week'] == w)
        ]['user_id'].nunique()
        cohort_data.append({
            'cohort_week': week, 'week': w,
            'retention': active / cohort_users * 100
        })

cohort_pivot = pd.pivot_table(
    pd.DataFrame(cohort_data),
    index='cohort_week', columns='week',
    values='retention'
).round(1)
print(cohort_pivot)
```
