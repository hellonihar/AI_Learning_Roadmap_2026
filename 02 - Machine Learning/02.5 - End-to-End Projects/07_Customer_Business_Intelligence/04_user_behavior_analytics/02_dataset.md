# User Behavior Analytics — Dataset

## Source
Synthetic event-log data (simulates real SaaS product) or [Google Analytics Sample](https://support.google.com/analytics/answer/7586738) (GA4 public dataset on BigQuery).

## Size & Shape
- **Total**: 2M+ event records, ~50K users, 3 months (Jan–Mar 2026)
- **Event schema**: user_id, timestamp, event_name, page_url, session_id, device_type, feature_name, duration_sec
- **Event types**: page_view, signup, onboarding_step_1-5, feature_click, share, payment, upgrade, cancel, login
- **Derived features**: session_count, total_duration, features_used, days_active, conversion_flag

## Challenges
- **Session boundary detection** — No explicit session_start; need to define session timeout gap (30 min)
- **Bot traffic** — ~8% of events are from automated scripts; need user-agent + rate filtering
- **Event name sparsity** — 200+ unique event names; many are legacy and unused
- **Incomplete data** — Users who block analytics cookies (GDPR) appear as anonymous; can't track cross-session
