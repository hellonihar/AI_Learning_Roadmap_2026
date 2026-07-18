# Linear Search

```python
def linear_search(arr, target):
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1
```

# Binary Search

```python
def binary_search(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == target: return mid
        if arr[mid] < target: l = mid + 1
        else: r = mid - 1
    return -1
```

# Binary Search on Answer

```python
# Find smallest value such that condition(x) is True
def binary_search_answer(lo, hi):
    while lo < hi:
        mid = (lo + hi) // 2
        if condition(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

# Example: square root of x
def my_sqrt(x):
    lo, hi = 0, x
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid * mid <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo
```

# Searching in Sorted Structures

```python
import bisect

arr = [1, 3, 5, 7, 9]
bisect.bisect_left(arr, 5)      # 2 (first index to insert)
bisect.bisect_right(arr, 5)     # 3 (last index + 1)

# Search in rotated sorted array
def search_rotated(arr, target):
    l, r = 0, len(arr) - 1
    while l <= r:
        mid = (l + r) // 2
        if arr[mid] == target: return mid
        if arr[l] <= arr[mid]:  # left half sorted
            if arr[l] <= target < arr[mid]: r = mid - 1
            else: l = mid + 1
        else:                   # right half sorted
            if arr[mid] < target <= arr[r]: l = mid + 1
            else: r = mid - 1
    return -1
```
