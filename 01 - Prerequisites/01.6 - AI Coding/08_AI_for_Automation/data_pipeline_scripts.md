# Data Pipeline Scripts

Generate ETL and data processing pipeline code.

## Prompt

```
"Write a daily ETL script that:
- Reads raw events from S3 (partitioned by date)
- Cleans and validates the data
- Computes daily aggregations per user
- Writes results to a PostgreSQL database
- Sends an alert if row count drops by more than 50%
- Logs each step with timing"
```

## Example

```python
def etl_pipeline(date: str, config: Config) -> None:
    logger = setup_logger()

    t0 = time.time()
    raw = read_s3(f"raw/{date}/*.parquet", config.s3_bucket)
    logger.info(f"Read {len(raw)} rows in {time.time()-t0:.1f}s")

    t0 = time.time()
    clean = validate_and_clean(raw)
    logger.info(f"Cleaned: {len(clean)} rows ({len(raw)-len(clean)} dropped)")

    t0 = time.time()
    agg = clean.groupby("user_id").agg({
        "event_count": "count",
        "revenue": "sum",
        "session_duration": "mean",
    }).reset_index()
    logger.info(f"Aggregated to {len(agg)} users")

    t0 = time.time()
    write_to_postgres(agg, config.db_url, table="daily_user_metrics")
    logger.info(f"Written in {time.time()-t0:.1f}s")

    if len(clean) < config.expected_min_rows * 0.5:
        send_alert(f"Row count anomaly: {len(clean)} vs expected {config.expected_min_rows}")
```

## Data Pipeline Prompts

- "Create a data validation step using Great Expectations"
- "Write a script to backfill historical data from a date range"
- "Generate a data quality report: nulls, duplicates, value ranges, distributions"
- "Create a script to synchronize data across S3 buckets in different regions"
- "Write a change data capture (CDC) script that syncs from Postgres to S3"
