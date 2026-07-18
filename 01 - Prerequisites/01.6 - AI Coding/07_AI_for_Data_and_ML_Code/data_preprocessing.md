# Writing Data Preprocessing Code

AI excels at generating data cleaning and preprocessing pipelines.

## Prompt

```
"Write a function to preprocess this CSV. The file has columns:
timestamp, user_id, action, duration, ip_address, user_agent.
- Parse timestamp to datetime
- Drop rows with missing user_id
- Remove outliers in duration (>99th percentile)
- Extract day_of_week and hour from timestamp
- Return cleaned DataFrame and a dict of stats about dropped rows"
```

## Example Output

```python
def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"total_rows": len(df), "dropped": {}}

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    stats["dropped"]["missing_user_id"] = df["user_id"].isna().sum()
    df = df.dropna(subset=["user_id"])

    threshold = df["duration"].quantile(0.99)
    stats["dropped"]["duration_outliers"] = (df["duration"] > threshold).sum()
    df = df[df["duration"] <= threshold]

    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour

    return df, stats
```

## Prompt Patterns

```
- "Normalize all numeric columns to [0,1] range using MinMaxScaler"
- "Encode categorical columns with fewer than 20 unique values as one-hot"
- "Split the data into train/val/test by time (chronological split)"
- "Handle missing values: numeric → median, categorical → mode"
- "Standardize column names to snake_case"
```
