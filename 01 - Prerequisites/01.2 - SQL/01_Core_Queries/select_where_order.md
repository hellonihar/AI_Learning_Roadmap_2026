# SELECT, WHERE, ORDER BY, LIMIT

```sql
-- Basic SELECT
SELECT * FROM users;
SELECT id, name, email FROM users;

-- Filtering
SELECT * FROM orders
WHERE status = 'completed'
  AND amount > 100
  AND created_at >= '2025-01-01';

-- Sorting
SELECT * FROM products
ORDER BY price DESC, name ASC;

-- Limiting
SELECT * FROM events
LIMIT 100;

-- DISTINCT
SELECT DISTINCT category FROM products;

-- Aliases
SELECT COUNT(*) AS total_users FROM users;
```
