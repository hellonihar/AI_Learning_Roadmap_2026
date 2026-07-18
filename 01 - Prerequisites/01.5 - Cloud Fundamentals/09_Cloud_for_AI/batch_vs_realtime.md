# Batch vs Real-time Inference

## Batch Inference
- Process large amounts of data at once (thousands to millions of records)
- No immediate response needed
- Scheduled (hourly, daily) or triggered by data arrival

| Attribute | Batch | Real-time |
|---|---|---|
| Latency | Minutes to hours | Milliseconds to seconds |
| Cost | Lower (use spot instances) | Higher (always-on infra) |
| Compute | Scale up massively, then down to 0 | Auto-scale but min running |
| Tools | SageMaker Batch, AWS Batch, GCP Batch | SageMaker Endpoints, Cloud Run |

## Real-time Inference
- Single request → immediate prediction
- Always-on endpoints with auto-scaling
- Trade-off: cost vs latency

## Hybrid Approach
```
Input → Queue (SQS) → Lightweight router → 
  ├─ Fast path: cached or simple model  (real-time)
  └─ Slow path: complex ensemble model  (async callback)
```

For most AI engineers: start with batch inference, add real-time only when latency requirements demand it.
