# Object Storage (S3 / GCS / Blob)

## Key Concepts
- **Bucket**: container for objects (globally unique name)
- **Object**: file + metadata + unique key
- **URL pattern**: `https://bucket.s3.region.amazonaws.com/key`

## Storage Tiers
| Tier | Use Case | Cost |
|---|---|---|
| Standard | Active data | Higher |
| Infrequent Access | Backups, older data | Lower storage, higher retrieval |
| Glacier / Archive | Compliance, long-term | Lowest |

## Key Features
- **Versioning**: keep all versions of an object (protect against deletes)
- **Lifecycle rules**: auto-move to colder tiers after N days
- **Pre-signed URLs**: generate time-limited access URLs
- **Event notifications**: trigger Lambda on upload

## For AI Engineers
- Store datasets in buckets (organized by project / date)
- Use Pre-signed URLs for secure dataset download by training instances
- Version model artifacts so you can roll back
- Never store credentials or secrets in buckets
