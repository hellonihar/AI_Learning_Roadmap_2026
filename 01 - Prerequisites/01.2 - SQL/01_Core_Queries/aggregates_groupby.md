# Aggregates & GROUP BY

```sql
-- Basic aggregates
SELECT COUNT(*), SUM(amount), AVG(amount),
       MIN(amount), MAX(amount)
FROM orders;

-- GROUP BY
SELECT status, COUNT(*) AS cnt, AVG(amount) AS avg_amount
FROM orders
GROUP BY status;

-- HAVING (filter after aggregation)
SELECT user_id, COUNT(*) AS order_count
FROM orders
GROUP BY user_id
HAVING COUNT(*) > 5
ORDER BY order_count DESC;

-- GROUP BY with multiple columns
SELECT category, status, COUNT(*)
FROM orders
GROUP BY category, status;
```
