# Sliding Window

Fixed or variable window over a sequence. Common in time-series and substring problems.

```python
def sliding_window(arr, k):
    left = 0
    for right in range(len(arr)):
        # add arr[right] to window
        while invalid_condition():
            # remove arr[left] from window
            left += 1
        # record result
```

# Two Pointers

Used on sorted arrays, linked lists, and string palindrome problems.

```python
def two_pointers(arr):
    l, r = 0, len(arr) - 1
    while l < r:
        # process arr[l], arr[r]
        l += 1; r -= 1
```

# Prefix Sums

O(1) range sum queries after O(n) preprocessing.

```python
range_sum(l, r) = prefix[r+1] - prefix[l]
```

# Fast & Slow Pointer

Cycle detection and middle finding in linked lists.

```python
slow = slow.next
fast = fast.next.next
```

# Binary Search Pattern

Search in sorted space — arrays, matrices, or answer space.

```python
while lo < hi:
    mid = (lo + hi) // 2
    if condition(mid): hi = mid
    else: lo = mid + 1
```

# Greedy Basics

Make the locally optimal choice at each step.

```python
# Coin change (minimum coins, unlimited supply)
def min_coins(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for c in coins:
        count += amount // c
        amount %= c
    return count if amount == 0 else -1
```

## Pattern Recognition

| Pattern | When to Use | Example Problem |
|---|---|---|
| Sliding window | Contiguous subarray/substring | Max subarray sum |
| Two pointers | Sorted array, palindrome | Two sum, reverse |
| Prefix sum | Range queries, subarray sums | Subarray sum = k |
| Fast & slow | Linked list, cyclic problems | Cycle detection |
| Binary search | Sorted space, monotonic condition | Search in rotated array |
| Greedy | Locally optimal → globally optimal | Coin change, interval scheduling |
