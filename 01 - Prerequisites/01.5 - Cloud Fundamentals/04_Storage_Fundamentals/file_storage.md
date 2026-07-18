# File Storage (EFS / Filestore)

## Key Concepts
- Shared NFS volume accessible by multiple VMs simultaneously
- POSIX-compliant (works with standard file I/O)
- Scales automatically as files are added

## Use Cases
- Shared datasets across a training cluster
- Home directories shared across team instances
- Web content served from multiple web servers

## Performance Modes
| Mode | Use Case |
|---|---|
| General Purpose | Latency-sensitive (web serving, content management) |
| Max I/O | High-throughput, big data, media processing |

## For AI Engineers
- Mount shared training data across multi-node cluster
- Store experiment logs and checkpoints accessible from any node
- Not ideal for high-throughput training data loading (use FSx for Lustre instead)
