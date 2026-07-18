# Sliding Window Basics

```python
# Fixed window size k
def fixed_window(arr, k):
    window_sum = sum(arr[:k])
    result = [window_sum]
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        result.append(window_sum)
    return result

# Variable window (smallest window with sum >= target)
def min_window(arr, target):
    left = total = 0
    min_len = float("inf")
    for right in range(len(arr)):
        total += arr[right]
        while total >= target:
            min_len = min(min_len, right - left + 1)
            total -= arr[left]
            left += 1
    return min_len if min_len != float("inf") else 0
```

## Template

```python
left = 0
for right in range(n):
    window.add(arr[right])           # expand
    while window_needs_shrink:       # contract
        window.remove(arr[left])
        left += 1
    update_answer()                  # record result
```

## AI Relevance
- Time-series rolling windows (moving averages, volatility)
- Sequence model context windows
- Streaming data statistics
