# Window Functions

```sql
-- ROW_NUMBER (deduplication)
SELECT *,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn
FROM events;

-- RANK vs DENSE_RANK
SELECT score,
    RANK() OVER (ORDER BY score DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY score DESC) AS dense_rank
FROM leaderboard;

-- LAG / LEAD (previous / next row)
SELECT user_id, amount, created_at,
    LAG(amount) OVER (PARTITION BY user_id ORDER BY created_at) AS prev_amount,
    LEAD(amount) OVER (PARTITION BY user_id ORDER BY created_at) AS next_amount
FROM orders;

-- Running total
SELECT user_id, amount, created_at,
    SUM(amount) OVER (PARTITION BY user_id ORDER BY created_at) AS running_total
FROM orders;

-- Moving average (last 7 rows)
SELECT date, revenue,
    AVG(revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma_7d
FROM daily_revenue;

-- FIRST_VALUE / LAST_VALUE
SELECT user_id, date, amount,
    FIRST_VALUE(amount) OVER (PARTITION BY user_id ORDER BY date) AS first_purchase
FROM orders;
```
