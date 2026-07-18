# Prefix Sums

```python
arr = [3, 1, 4, 1, 5, 9]
prefix = [0]
for x in arr:
    prefix.append(prefix[-1] + x)
# prefix = [0, 3, 4, 8, 9, 14, 23]

# Sum of subarray arr[l:r] inclusive
def range_sum(l, r):
    return prefix[r+1] - prefix[l]

range_sum(1, 3)  # 1+4+1 = 6
```

## Use Cases

- **Quick range queries** — O(1) after O(n) preprocessing
- **Subarray sum problems** — "find subarray with sum = k"
- **2D prefix sum** — rectangle sum in matrix
- **Running statistics** — cumulative averages for time-series data

## AI Relevance

Feature engineering: cumulative sums, rolling statistics, normalized cumulative metrics.
