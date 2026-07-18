# Singly & Doubly Linked Lists

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Singly linked: 1 → 2 → 3 → None
# Doubly linked: None ← 1 ↔ 2 ↔ 3 → None

# Traversal
def traverse(head):
    while head:
        print(head.val)
        head = head.next

# Insert at head
def insert_head(head, val):
    new = ListNode(val, head)
    return new

# Reverse singly linked list
def reverse(head):
    prev = None
    while head:
        nxt = head.next
        head.next = prev
        prev = head
        head = nxt
    return prev

# Find middle (slow & fast)
def middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```
