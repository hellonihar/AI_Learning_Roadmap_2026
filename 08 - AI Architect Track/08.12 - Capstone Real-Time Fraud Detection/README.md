# Capstone: Real-Time Fraud Detection & Prevention Platform

Design a high-throughput, low-latency ML-powered fraud detection platform that processes thousands of transactions per second with sub-100ms decision latency.

## Context

A financial technology company processes millions of transactions daily (payments, transfers, card authorizations). Fraud costs the company $50M+ annually. They need a real-time ML system that scores every transaction in under 100ms while adapting to evolving fraud patterns.

- **Scale**: 5k–20k transactions/second peak, 200M+ transactions/month
- **Latency SLA**: p99 < 100ms end-to-end (including feature computation + model inference)
- **Data sources**: real-time transaction streams, user profile DB, merchant history, device fingerprinting, geolocation
- **Model types**: gradient-boosted trees (XGBoost/LightGBM), deep learning (TabNet), graph neural networks for network analysis
- **Constraints**: PCI-DSS compliance, explanation required for declined transactions, false positive rate < 3%, model retraining within 24h of new fraud patterns
- **Current state**: rule-based engine with 60% detection rate at 10% FPR, no ML

## Deliverables

### 1. System Architecture Design
- **End-to-end architecture**: transaction ingestion → feature computation → model scoring → decision → action
- **Streaming backbone**: Kafka / Pulsar for transaction ingestion, schema registry, partitioning strategy
- **Online feature store**: Redis / DynamoDB / Cassandra for real-time features (user velocity, merchant stats, device risk)
- **Model serving**: low-latency inference (ONNX Runtime, Nvidia Triton, custom C++), model versioning, canary deployment

### 2. Feature Pipeline Architecture
- **Real-time feature computation**: Flink / Kafka Streams for windowed aggregations (user tx count in 1h, 24h, 7d)
- **Batch features**: daily/weekly aggregates computed in Spark, materialized to online store
- **Feature consistency**: point-in-time correctness, exactly-once semantics, feature freshness SLAs
- **Feature catalog**: feature definitions, versioning, lineage, owner documentation

### 3. Model Training & Retraining Pipeline
- **Training data generation**: labeling pipeline (confirmed fraud + delay window), feature backfill, time-series cross-validation
- **Model selection**: XGBoost vs. TabNet vs. GNN for card network analysis
- **Automated retraining**: trigger-based (drift detection, new fraud pattern), scheduled daily
- **Champion/challenger**: A/B testing new models with shadow scoring, performance gate for promotion
- **Explainability**: SHAP values for declined transactions (regulatory requirement)

### 4. Model Serving & Decision Engine
- **Inference architecture**: stateless scoring service, horizontal autoscaling, pre-warmed model caches
- **Ensemble strategy**: multiple model outputs → decision fusion → final action (approve/decline/review)
- **Fallback**: degraded mode (rule-based) if ML service unavailable, circuit breaker patterns
- **Shadow scoring**: run new models in shadow for A/B evaluation without impacting decisions

### 5. Monitoring & Drift Detection
- **Data drift**: feature distribution monitoring (PSI, KL divergence), alert on drift thresholds
- **Model drift**: prediction drift, accuracy degradation (delayed labels), calibration monitoring
- **Concept drift**: fraud pattern shift detection, automated retraining triggers
- **Operational monitoring**: latency percentiles, throughput, error rates, GPU utilization
- **Dashboard**: Grafana dashboards for data drift, model performance, system health

### 6. Security & Compliance
- **PCI-DSS compliance**: encrypted data at rest and in transit, access control logging, segmentation
- **Explainability for regulators**: SHAP summaries for declined transactions, audit trail of decisions
- **Fraud data governance**: retention policies, anonymization for model training, data minimization
- **Adversarial robustness**: evasion attack detection (unusual feature values), adversarial training

### 7. Cost & Capacity Planning
- **Compute cost**: real-time inference (GPU/CPU per transaction), batch training costs
- **Storage cost**: feature store (online + offline), transaction logs, model artifacts
- **Autoscaling policies**: request-based scaling for inference, spot instances for batch training
- **Cost-per-decision**: tracked and optimized across model versions

### 8. Deployment & Rollout Strategy
- **Shadow mode**: run ML predictions alongside rule engine without affecting decisions
- **Partial rollout**: ML for low-value transactions first, expand to high-value
- **A/B testing**: controlled experiment measuring detection rate, FPR, customer impact
- **Rollback plan**: automated rollback if FPR exceeds threshold or latency degrades

## Evaluation Criteria

| Criteria | Weight | Description |
|----------|--------|-------------|
| Real-time architecture | 25% | Latency optimization, streaming design, autoscaling |
| Feature pipeline design | 20% | Real-time features, consistency, freshness SLAs |
| Model lifecycle | 20% | Retraining, champion/challenger, shadow scoring |
| Compliance & explainability | 15% | PCI-DSS, SHAP, audit trail |
| Resilience | 10% | Fallback patterns, circuit breakers, DR |
| Presentation | 10% | Clarity of architecture and tradeoff analysis |

## Format
- **Written**: PDF or Markdown (30–50 pages)
- **Live review**: 60-minute presentation + 30-minute Q&A
- **Tools**: any diagramming tool; throughput/latency calculations expected
