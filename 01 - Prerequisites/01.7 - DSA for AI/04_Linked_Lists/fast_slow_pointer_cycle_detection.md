# Fast & Slow Pointer Technique

```python
# Find middle
def middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

# Find kth from end
def kth_from_end(head, k):
    fast = head
    for _ in range(k):
        if not fast: return None
        fast = fast.next
    slow = head
    while fast:
        slow = slow.next
        fast = fast.next
    return slow
```

# Cycle Detection

```python
# Floyd's Cycle Detection (Tortoise & Hare)
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Find cycle start
def detect_cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:                     # cycle detected
            slow = head
            while slow != fast:              # find start
                slow = slow.next
                fast = fast.next
            return slow
    return None
```
