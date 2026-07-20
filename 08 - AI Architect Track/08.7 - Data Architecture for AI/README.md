# Data Architecture for AI

Design data pipelines, storage systems, and governance frameworks that power AI workloads at scale.

## Learning Objectives
- Architect data platforms that support both training and real-time inference
- Implement feature stores for consistent features across training and serving
- Design data quality monitoring and data contracts for ML pipelines

## Deep Topics

### 1. Data Mesh & Data Fabric for ML
- **Data mesh principles**: domain ownership, data as a product, self-serve data platform, federated governance
- **Data fabric**: metadata-driven integration across heterogeneous data sources
- **Applying data mesh to ML**: each domain owns its features, training datasets, and evaluation data
- **Cross-domain data discovery**: data catalog, data marketplace, data product APIs

### 2. Real-Time Feature Pipelines & Streaming Ingestion
- **Stream processing engines**: Apache Kafka, Apache Pulsar, Amazon Kinesis, Azure Event Hubs
- **Streaming feature computation**: Flink SQL, Kafka Streams, Spark Structured Streaming
- **Feature freshness SLAs**: milliseconds (clickstream), seconds (fraud detection), minutes (recommendations)
- **Exactly-once semantics**: ensuring feature consistency for training and serving
- **Backfill strategies**: reprocessing historical data with updated feature logic

### 3. Feature Store Implementation Patterns
- **Online store**: low-latency key-value serving (Redis, DynamoDB, Cassandra, Azure Cosmos DB)
- **Offline store**: large-scale historical feature storage (Parquet/Delta Lake on S3/ADLS/GCS)
- **Point-in-time joins**: correct feature retrieval for training datasets avoiding data leakage
- **Feature serving architecture**: dual-write / log-based / batch materialization patterns
- **Feature validation**: type checking, range checks, null detection at write time

### 4. Data Versioning & Data Contracts
- **Data versioning**: lakeFS, DVC, Delta Lake time travel for dataset reproducibility
- **Data contracts**: schema, semantics, SLAs, ownership between data producers and consumers
- **Schema registry**: Avro/Protobuf/JSON Schema with evolution rules (backward/forward compatibility)
- **Breaking change detection**: automated validation in CI pipelines before deployment

### 5. Synthetic Data Generation & Augmentation
- **Synthetic data techniques**: GANs, VAEs, diffusion models for tabular/image/text data
- **Data augmentation**: rotation, cropping, noise injection for vision; back-translation, mixup for NLP
- **Privacy-preserving synthetic data**: differential privacy guarantees, utility evaluation
- **Tools**: Mostly AI, Gretel, YData, SDV (Synthetic Data Vault)

### 6. Data Quality Testing & Monitoring
- **Quality dimensions**: completeness, consistency, timeliness, accuracy, uniqueness, validity
- **Testing frameworks**: Great Expectations, Soda, Deequ, DBT tests
- **Monitoring dashboards**: data quality scorecards, trend alerts for drift
- **Automated remediation**: quarantine bad data, alert owners, trigger reprocessing

### 7. Data Governance & Cataloging
- **Data catalog tools**: DataHub, Amundsen, Alation, Collibra, Apache Atlas
- **Metadata management**: technical, business, operational metadata
- **Data discovery**: search, browse, lineage graph, popularity metrics
- **Access control**: column-level, row-level, purpose-based access for training and inference

### 8. Integration with Data Lake / Warehouse / Lakehouse
- **Lakehouse architecture**: Delta Lake, Apache Iceberg, Apache Hudi for unified batch/streaming
- **Query engines**: Trino, Spark SQL, DuckDB for interactive exploration
- **Data sharing**: Delta Sharing, Apache Iceberg REST Catalog for cross-platform access
- **ETL/ELT patterns for ML**: incremental processing, feature materialization, training dataset generation

## Deliverables
- Design a feature store architecture supporting real-time personalization with <10ms p99 serving latency
- Implement data contracts and quality monitoring for a critical ML feature pipeline
- Architect a data mesh for an organization with 5+ domain teams producing features
