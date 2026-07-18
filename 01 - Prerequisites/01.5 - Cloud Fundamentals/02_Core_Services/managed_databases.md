# Managed Databases

## Relational Databases
| AWS | GCP | Azure |
|---|---|---|
| RDS (PostgreSQL, MySQL, SQL Server, Oracle) | Cloud SQL (PostgreSQL, MySQL, SQL Server) | Azure SQL / MySQL / PostgreSQL |

**Key features**: automated backups, read replicas, multi-AZ, auto-scaling

## NoSQL Databases
| AWS | GCP | Azure |
|---|---|---|
| DynamoDB (key-value + document) | Firestore (document) / Bigtable (wide-column) | Cosmos DB (multi-model) |

**Key features**: serverless, auto-scale, single-digit ms latency

## Cache
| AWS | GCP | Azure |
|---|---|---|
| ElastiCache (Redis / Memcached) | Memorystore (Redis) | Cache for Redis |

Used to cache frequent queries, reduce DB load, store session state.

## For AI Engineers
- Store experiment metadata in PostgreSQL
- Cache embedding vectors in Redis for fast retrieval
- Use DynamoDB / Cosmos DB for high-throughput prompt/response logging
- Read replicas to avoid impacting production DB during ML data extraction
