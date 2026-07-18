# SQL Quick Reference

## Core Query
```sql
SELECT col1, COUNT(*) as cnt
FROM table
WHERE col2 > 100
GROUP BY col1
HAVING COUNT(*) > 5
ORDER BY cnt DESC
LIMIT 10;
```

## Joins
```sql
SELECT *
FROM users u
JOIN orders o ON u.id = o.user_id;
LEFT JOIN, RIGHT JOIN, FULL JOIN -- same syntax
```

## CTE
```sql
WITH user_stats AS (
    SELECT user_id, COUNT(*) as order_count
    FROM orders GROUP BY user_id
)
SELECT * FROM user_stats WHERE order_count > 10;
```

## Window Functions
```sql
ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)
RANK()       OVER (ORDER BY score DESC)
LAG(amount)  OVER (PARTITION BY user_id ORDER BY date)
SUM(sales)   OVER (PARTITION BY region ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
```

## Aggregates
```sql
COUNT(DISTINCT col), SUM(col), AVG(col), MIN(col), MAX(col)
```

## Conditional
```sql
CASE WHEN score > 90 THEN 'A'
     WHEN score > 80 THEN 'B'
     ELSE 'C' END AS grade
```

## AI Patterns
```sql
-- Sample
SELECT * FROM data ORDER BY RANDOM() LIMIT 1000;

-- Deduplicate
WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn
    FROM events
) SELECT * FROM ranked WHERE rn = 1;

-- Time features
SELECT user_id,
       COUNT(*) OVER (PARTITION BY user_id ORDER BY ts ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as rolling_30d_actions
FROM events;

-- JSON extract
SELECT data->>'name' AS name, data->>'score' AS score
FROM logs WHERE data @> '{"status": "completed"}';
```

## Indexes
```sql
CREATE INDEX idx_user_id ON orders(user_id);
CREATE INDEX idx_created_at ON events(created_at DESC);
```
