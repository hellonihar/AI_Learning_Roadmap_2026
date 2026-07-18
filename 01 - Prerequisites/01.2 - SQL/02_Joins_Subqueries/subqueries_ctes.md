# Subqueries & CTEs

## Subqueries

```sql
-- Scalar subquery (single value)
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);

-- IN subquery
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE amount > 100);

-- EXISTS subquery
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.user_id = u.id AND o.amount > 500
);

-- Subquery in FROM (derived table)
SELECT dept, avg_salary
FROM (
    SELECT department_id AS dept, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
) sub
WHERE avg_salary > 80000;
```

## CTEs (WITH)

```sql
-- Basic CTE
WITH high_value_users AS (
    SELECT user_id, SUM(amount) AS total_spent
    FROM orders
    GROUP BY user_id
    HAVING SUM(amount) > 1000
)
SELECT u.name, h.total_spent
FROM users u
JOIN high_value_users h ON u.id = h.user_id;

-- Multiple CTEs
WITH
user_orders AS (
    SELECT user_id, COUNT(*) AS cnt FROM orders GROUP BY user_id
),
user_revenue AS (
    SELECT user_id, SUM(amount) AS rev FROM orders GROUP BY user_id
)
SELECT u.name, o.cnt, r.rev
FROM users u
JOIN user_orders o ON u.id = o.user_id
JOIN user_revenue r ON u.id = r.user_id;
```
