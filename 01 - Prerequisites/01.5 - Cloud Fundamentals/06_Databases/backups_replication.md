# Backups & Replication

## Backup Types
| Type | Description | RPO |
|---|---|---|
| Snapshots | Full point-in-time copy to object storage | Minutes to hours |
| Continuous backup (PITR) | Transaction log streaming | Seconds |
| Read replicas | Async copy for read scaling | Sub-second lag |

## Replication Strategies
| Strategy | Description | Use Case |
|---|---|---|
| Same-region, multi-AZ | Sync replica in another AZ | HA within region |
| Cross-region | Async replica in different region | Disaster recovery, geo-proximity |
| Multi-master | Write in multiple regions | Global apps, low-latency writes |

## For AI Engineers
- Cross-region replication for global model serving (data close to inference)
- Enable PITR on databases used for experiments (cost of losing data > storage cost)
- Snapshot DB before running large training jobs that modify tables
