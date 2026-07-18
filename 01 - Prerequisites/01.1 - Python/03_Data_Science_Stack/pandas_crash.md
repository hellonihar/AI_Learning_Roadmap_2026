# Pandas Crash Course

## Reading & Writing
```python
import pandas as pd
df = pd.read_csv("data.csv")
df = pd.read_parquet("data.parquet")
df = pd.read_json("data.json")
df.to_csv("out.csv", index=False)
```

## Exploration
```python
df.head(); df.tail(); df.info(); df.describe()
df.shape; df.columns; df.dtypes
df["col"].value_counts()
df["col"].isna().sum()
```

## Selecting Data
```python
df["col"]              # column as Series
df[["a", "b"]]         # multiple columns
df.iloc[10:20]         # integer-location
df.loc[df["age"] > 30] # label-location
df.query("age > 30 & city == 'NYC'")
```

## GroupBy & Agg
```python
df.groupby("category")["value"].mean()
df.groupby("category").agg({
    "value": ["mean", "std", "count"],
    "price": "sum"
})
```

## Merge / Join
```python
pd.merge(df1, df2, on="key")
pd.merge(df1, df2, on="key", how="left")
df1.join(df2, lsuffix="_left", rsuffix="_right")
```

## Apply / Transform
```python
df["col"].map(lambda x: x * 2)
df["col"].apply(len)
df.groupby("cat")["val"].transform("zscore")
```

## Handling Missing
```python
df.dropna(); df.fillna(0)
df.fillna(df.median())
df.interpolate()
```
