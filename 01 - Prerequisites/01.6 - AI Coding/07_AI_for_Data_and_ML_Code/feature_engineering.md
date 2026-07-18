# Feature Engineering Code

Use AI to generate feature transformations based on your data schema.

## Prompt

```
"Here are the columns in my event log DataFrame:
user_id, event_name, timestamp, revenue, country, device_type.

Generate 10 useful features for predicting user churn.
Include: rolling aggregations, time-based features,
and user-profile features. Return the feature engineering
function as Python code."
```

## Example Output

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    # Time-based
    features["hour"] = features["timestamp"].dt.hour
    features["day_of_week"] = features["timestamp"].dt.dayofweek
    features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)

    # Rolling aggregations (per user)
    features = features.sort_values(["user_id", "timestamp"])
    features["revenue_7d"] = (
        features.groupby("user_id")["revenue"]
        .transform(lambda x: x.rolling(7, min_periods=1).sum())
    )
    features["event_count_30d"] = (
        features.groupby("user_id")
        .cumcount() + 1
    )

    # User profile features
    features["days_since_first_event"] = (
        features.groupby("user_id")["timestamp"]
        .transform(lambda x: (x.max() - x.min()).dt.days)
    )

    return features
```

## Feature Engineering Prompts

```
- "Generate interaction features between these columns: [columns]"
- "Create target encoding for high-cardinality categorical: [column]"
- "Create lag features for the target variable over 7, 14, 30 days"
- "Aggregate event counts per user per day for the last 90 days"
- "Generate embeddings for text columns using a pre-trained model"
```
