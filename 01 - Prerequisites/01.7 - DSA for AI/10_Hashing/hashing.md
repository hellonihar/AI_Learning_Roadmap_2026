# Hash Maps & Hash Sets

```python
# HashMap / Dict
freq = {}
freq["a"] = freq.get("a", 0) + 1

from collections import defaultdict
freq = defaultdict(int)
freq["a"] += 1

# HashSet
seen = set()
if x in seen: ...
```

# Frequency Counting

```python
from collections import Counter

freq = Counter("abracadabra")
# Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})

# Most common
freq.most_common(2)  # [('a', 5), ('b', 2)]

# Compare counters
Counter("abc") == Counter("cba")  # True (anagrams)
```

# Subarray & Substring Problems

```python
# Two-sum (find pair with given sum)
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []

# Longest substring without repeating characters
def longest_unique(s):
    last_seen = {}
    left = max_len = 0
    for right, c in enumerate(s):
        if c in last_seen and last_seen[c] >= left:
            left = last_seen[c] + 1
        last_seen[c] = right
        max_len = max(max_len, right - left + 1)
    return max_len

# Subarray sum equals k
def subarray_sum(nums, k):
    prefix_sum = {0: 1}
    current = count = 0
    for x in nums:
        current += x
        count += prefix_sum.get(current - k, 0)
        prefix_sum[current] = prefix_sum.get(current, 0) + 1
    return count
```

# Optimized Lookups

- O(1) average for insert, delete, lookup
- O(n) worst case (hash collisions, rare)
- Trade-off: memory vs speed — hash maps use more memory than arrays but give fast lookups
