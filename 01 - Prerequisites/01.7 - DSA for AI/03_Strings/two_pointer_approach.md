# Two Pointer Approach

```python
# Palindrome check
def is_palindrome(s):
    l, r = 0, len(s) - 1
    while l < r:
        if s[l] != s[r]:
            return False
        l, r = l + 1, r - 1
    return True

# Reverse string in-place
def reverse(s: list[str]):
    l, r = 0, len(s) - 1
    while l < r:
        s[l], s[r] = s[r], s[l]
        l, r = l + 1, r - 1

# Two pointers on sorted array
def two_sum_sorted(arr, target):
    l, r = 0, len(arr) - 1
    while l < r:
        s = arr[l] + arr[r]
        if s == target: return [l, r]
        if s < target: l += 1
        else: r -= 1
    return []
```

## Use Cases

- Palindrome validation
- Reverse strings / arrays
- Pair sums in sorted arrays
- Remove duplicates in-place
- Merge two sorted sequences
