# Managed Database Services

## Serverless Databases
| Service | Type | Good For |
|---|---|---|
| Aurora Serverless | Relational (MySQL/PostgreSQL) | Variable workloads, infrequent usage |
| DynamoDB On-Demand | NoSQL key-value | Unknown or bursty traffic |
| Firestore | NoSQL document | Mobile/web apps |

## Backup & Restore
- **Automated snapshots**: daily, retained for configurable period (7-35 days)
- **Point-in-Time Recovery (PITR)**: restore to any second within retention window
- **Manual snapshots**: permanent, not auto-deleted

## Read Replicas
- Async replica for read-heavy workloads
- Can be cross-region (for disaster recovery)
- Each replica has its own endpoint
- **Use**: offload ML data extraction to a replica

## For AI Engineers
- Use Aurora Serverless v2 for ML experiment database (auto-scales to zero)
- Cross-region replicas for global dataset access
- PITR to recover accidentally deleted experiment records
