# Character Frequency Techniques

```python
from collections import Counter

# Frequency map
freq = Counter("hello")  # {'h':1, 'e':1, 'l':2, 'o':1}

# Array for lowercase letters
freq = [0] * 26
for c in s:
    freq[ord(c) - ord('a')] += 1

# Check if two strings are anagrams
def is_anagram(s1, s2):
    return Counter(s1) == Counter(s2)
    # or: sorted(s1) == sorted(s2)
```

## Key Operations

- **Ord & chr**: `ord('a')` → 97, `chr(97)` → 'a'
- **Substring frequency**: `Counter(s[l:r])` or rolling update
- **Character grouping**: group anagrams using sorted string as key
