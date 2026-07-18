# Relational Databases

## When to Use
- Structured data with relationships
- ACID transactions required
- Complex queries with JOINs
- Schema is known and stable

## Managed Services
| Provider | Service | Engines |
|---|---|---|
| AWS | RDS | PostgreSQL, MySQL, SQL Server, Oracle, MariaDB |
| GCP | Cloud SQL | PostgreSQL, MySQL, SQL Server |
| Azure | Azure SQL / MySQL / PostgreSQL | SQL Server, MySQL, PostgreSQL |

## Key Features
- **Multi-AZ**: synchronous standby replica in another AZ for HA
- **Read replicas**: async copies for read scaling (max 15 per instance)
- **Automated backups**: daily snapshots with PITR (point-in-time recovery)
- **Auto-scaling storage**: grows as needed

## For AI Engineers
- Store experiment tracking metadata (parameters, metrics, artifacts)
- Log user feedback on model predictions
- Read replicas for extracting training data without impacting production
