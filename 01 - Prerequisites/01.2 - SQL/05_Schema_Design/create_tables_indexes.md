# Schema Design

```sql
-- CREATE TABLE
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Foreign Key
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    amount NUMERIC(10,2) NOT NULL,
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Constraints
ALTER TABLE users ADD CONSTRAINT chk_status
    CHECK (status IN ('active', 'inactive', 'banned'));

-- Indexes
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_created_at ON orders(created_at DESC);
CREATE INDEX idx_users_status ON users(status) WHERE status = 'active';
CREATE INDEX idx_orders_lookup ON orders(user_id, created_at DESC);

-- Normalization (3NF summary)
-- 1NF: atomic columns, no repeating groups
-- 2NF: 1NF + all non-key cols depend on full PK
-- 3NF: 2NF + no transitive dependencies

-- JSON queries (PostgreSQL)
SELECT data->>'name' AS name,
       data->'address'->>'city' AS city,
       data @> '{"verified": true}' AS is_verified
FROM profiles
WHERE data->>'plan' = 'premium';
```
