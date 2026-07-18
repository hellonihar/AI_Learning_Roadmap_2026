# Serverless Computing

Run code without provisioning or managing servers. Pay only for execution time.

## Key Services
| Provider | Service | Use Case |
|---|---|---|
| AWS | Lambda | Event-driven functions |
| GCP | Cloud Functions | Lightweight APIs |
| Azure | Functions | Enterprise workflows |

## Characteristics
- **Event-driven**: triggered by HTTP, queues, cron, S3 events
- **Auto-scales**: from 0 to thousands of concurrent executions instantly
- **Cold starts**: first invocation after idle may take 1-2s extra
- **Time limit**: typically 15 min max (AWS Lambda)
- **Stateless**: no local filesystem persistence between calls

## For AI Engineers
- Pre-process data on upload: S3 trigger → Lambda → clean → save
- Lightweight inference: wrap a small model (distilbert, ONNX) in Lambda
- Async processing: SQS → Lambda → process → write results
- **Not suitable**: GPU inference, >500MB model, long-running training
