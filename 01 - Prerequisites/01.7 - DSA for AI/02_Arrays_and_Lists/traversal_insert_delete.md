# Traversal, Insertion, Deletion

```python
arr = [1, 2, 3, 4, 5]

# Traversal
for i, val in enumerate(arr): ...
for val in arr: ...

# Insertion (O(n))
arr.insert(2, 99)      # at index 2
arr.append(100)        # at end (O(1) amortized)

# Deletion (O(n))
arr.pop(2)             # by index
arr.remove(99)         # by value
arr.pop()              # last element (O(1))

# Slicing (creates new list)
sub = arr[1:4]         # indices 1,2,3
rev = arr[::-1]        # reversed copy
```

## Key Patterns

- **In-place modification** — avoid copies for large arrays
- **Two-pointer** — sorted arrays, partition, reverse
- **Running pointers** — merge, partition, deduplicate
