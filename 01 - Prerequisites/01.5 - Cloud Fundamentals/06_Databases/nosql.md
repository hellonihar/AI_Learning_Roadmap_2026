# NoSQL Databases

## When to Use (vs Relational)
- Flexible / evolving schema
- High throughput at scale
- Simple key-value or document lookups
- Horizontal scaling needed

## Types
| Type | Example | Use Case |
|---|---|---|
| Key-Value | DynamoDB, Redis | Caching, session store, vector metadata |
| Document | Firestore, Cosmos DB | User profiles, content management |
| Wide-Column | Bigtable, Cassandra | Time-series, large-scale event data |
| Graph | Neptune, Neo4j | Knowledge graphs, recommendation |

## For AI Engineers
- **DynamoDB**: store prompt-response logs at high throughput
- **Firestore**: real-time sync for AI app state
- **Redis**: cache embeddings, rate limit counters, ephemeral queues
- **Bigtable**: store large volumes of feature data for real-time inference
