# Storage Overview

## Object Storage (S3, GCS, Blob)
- **What**: files stored as objects in buckets with unique URLs
- **Use for**: datasets, model artifacts, images, logs, backups
- **Key features**: unlimited scale, tiered pricing (hot/cold/archive), lifecycle policies
- **AI use**: store training data, checkpoints, model registry

## Block Storage (EBS, Persistent Disk)
- **What**: virtual hard drives attached to a VM
- **Use for**: OS volumes, database storage, low-latency scratch space
- **Key features**: snapshot backups, encryption, io1/gp3 volume types
- **AI use**: fast reads for training data loading, checkpoint writes

## File Storage (EFS, Filestore)
- **What**: shared NFS volume mountable by multiple VMs
- **Use for**: shared datasets across a cluster, home directories
- **Key features**: POSIX-compliant, scalable, concurrent access
- **AI use**: shared dataset mount across multi-node training cluster
