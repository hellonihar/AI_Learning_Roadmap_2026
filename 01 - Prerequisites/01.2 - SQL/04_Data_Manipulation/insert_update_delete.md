# Data Manipulation

```sql
-- INSERT
INSERT INTO users (name, email, created_at)
VALUES ('Alice', 'alice@email.com', NOW());

INSERT INTO logs (event, payload)
SELECT 'signup', json_build_object('user_id', id)
FROM users WHERE created_at > NOW() - INTERVAL '1 day';

-- UPDATE
UPDATE users
SET status = 'inactive', updated_at = NOW()
WHERE last_login < NOW() - INTERVAL '1 year';

-- DELETE
DELETE FROM temp_events
WHERE created_at < NOW() - INTERVAL '30 days';

-- MERGE (upsert)
MERGE INTO users u
USING (VALUES ('bob@email.com', 'Bob')) AS s(email, name)
ON u.email = s.email
WHEN MATCHED THEN UPDATE SET name = s.name
WHEN NOT MATCHED THEN INSERT (email, name) VALUES (s.email, s.name);

-- Transactions
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
-- ROLLBACK on error
```
