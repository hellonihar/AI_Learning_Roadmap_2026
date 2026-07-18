# Time & Space Complexity (Big-O)

## Common Complexities

| Notation | Name | Example |
|---|---|---|
| O(1) | Constant | Array access by index |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Single loop |
| O(n log n) | Linearithmic | Merge sort, Fast Fourier Transform |
| O(n²) | Quadratic | Nested loops |
| O(2ⁿ) | Exponential | Recursive Fibonacci (naive) |
| O(n!) | Factorial | Generating all permutations |

## Rules of Thumb

```
Ignore constants:      O(2n) → O(n)
Drop smaller terms:    O(n² + n) → O(n²)
Loops multiply:        nested loop → multiply complexities
Consecutive blocks:    add complexities → keep the largest
```

## Space Complexity

- **Auxiliary space**: extra space used by the algorithm (not counting input)
- **In-place algorithms**: O(1) extra space (e.g., bubble sort)
- **Recursive algorithms**: recursion depth → space complexity

## For AI Engineers

- Data preprocessing: O(n) is usually fine, O(n²) on 1M rows is not
- Model inference: O(1) or O(log n) lookup is ideal for real-time serving
- Always consider space: storing 1B embeddings in a hash map needs ~100GB+
