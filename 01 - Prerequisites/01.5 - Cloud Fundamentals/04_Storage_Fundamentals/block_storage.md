# Block Storage (EBS / Persistent Disk)

## Key Concepts
- Attached to a single VM at a time
- Survives instance stop/start (persistent)
- Treated as a block device (you can partition, format, mount)

## Volume Types
| Type | Use Case | Max IOPS |
|---|---|---|
| gp3 | General purpose, boot volumes | 16,000 |
| io2 | High-performance DB, low latency | 256,000 |
| st1 | Throughput-optimized (cold HDD) | 500 MB/s |
| sc1 | Cold, infrequent access | 250 MB/s |

## Key Features
- **Snapshots**: backup to S3, incremental
- **Encryption**: at-rest encryption with KMS
- **Multi-attach** (io2 only): attach same volume to multiple instances (clustered DBs)

## For AI Engineers
- Use gp3 for boot volumes
- Use io2 Block Express for high-throughput training data loading
- Snapshot volumes before destructive experiments
- Detach and delete temp volumes when not in use to save cost
