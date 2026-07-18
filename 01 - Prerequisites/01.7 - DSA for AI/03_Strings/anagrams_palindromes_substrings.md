# Anagrams, Palindromes, Substrings

```python
from collections import Counter

# Group anagrams
def group_anagrams(words):
    groups = {}
    for w in words:
        key = "".join(sorted(w))
        groups.setdefault(key, []).append(w)
    return list(groups.values())

# All substrings (O(n²))
def all_substrings(s):
    return [s[i:j] for i in range(len(s)) for j in range(i+1, len(s)+1)]

# Longest palindromic substring (expand around center)
def longest_palindrome(s):
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l, r = l - 1, r + 1
        return s[l+1:r]

    result = ""
    for i in range(len(s)):
        odd = expand(i, i)
        even = expand(i, i + 1)
        result = max(result, odd, even, key=len)
    return result
```

## Common Problems

- Valid anagram
- Find all anagrams in a string (sliding window)
- Longest substring without repeating characters
- Palindromic substrings count
- Longest palindromic substring
