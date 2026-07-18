# AI-Relevant SQL Patterns

```sql
-- Sampling
SELECT * FROM training_data
ORDER BY RANDOM() LIMIT 10000;               -- simple random
SELECT * FROM training_data
TABLESAMPLE BERNOULLI(1);                      -- 1% sample (PostgreSQL)

-- Stratified sampling via window
WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY label ORDER BY RANDOM()
    ) AS rn
    FROM training_data
) SELECT * FROM ranked WHERE rn <= 1000;       -- 1k per class

-- Deduplication (keep latest)
WITH deduped AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY user_id ORDER BY event_timestamp DESC
    ) AS rn
    FROM user_events
) SELECT * FROM deduped WHERE rn = 1;

-- Feature engineering: rolling aggregations
SELECT user_id, event_timestamp,
    COUNT(*) OVER (PARTITION BY user_id ORDER BY event_timestamp
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS rolling_30d_count,
    AVG(value) OVER (PARTITION BY user_id ORDER BY event_timestamp
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS rolling_30d_avg
FROM user_events;

-- Time-based features
SELECT user_id,
    EXTRACT(HOUR FROM created_at) AS hour_of_day,
    EXTRACT(DOW FROM created_at) AS day_of_week,
    CASE WHEN EXTRACT(DOW FROM created_at) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend
FROM orders;

-- User profile aggregation (feature table)
SELECT user_id,
    COUNT(*) AS total_orders,
    AVG(amount) AS avg_order_value,
    MAX(created_at) AS last_order_date,
    DATEDIFF('day', MAX(created_at), NOW()) AS days_since_last_order,
    SUM(amount) AS lifetime_value
FROM orders
GROUP BY user_id;

-- Pivot / cross-tab (status counts per user)
SELECT user_id,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_orders,
    COUNT(*) FILTER (WHERE status = 'cancelled') AS cancelled_orders,
    COUNT(*) FILTER (WHERE status = 'refunded') AS refunded_orders
FROM orders
GROUP BY user_id;

-- JSON extraction from LLM logs
SELECT
    data->>'model' AS model_name,
    data->'usage'->>'total_tokens' AS tokens,
    data->>'response' AS output,
    data->>'latency_ms' AS latency
FROM llm_logs
WHERE data->>'model' = 'gpt-4'
  AND created_at > NOW() - INTERVAL '7 days';

-- Training / validation split
WITH split AS (
    SELECT *, ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn,
           COUNT(*) OVER () AS total
    FROM labeled_data
) SELECT *,
    CASE WHEN rn <= total * 0.8 THEN 'train' ELSE 'val' END AS split
FROM split;
```
