# Queue & Deque

```python
from collections import deque

# Queue (FIFO)
q = deque()
q.append(1)         # enqueue
q.popleft()         # dequeue (O(1))
q[0]                # peek front

# Deque (double-ended)
d = deque()
d.append(1)         # add to right
d.appendleft(2)     # add to left
d.pop()             # remove from right
d.popleft()         # remove from left
```

# Circular Queue

```python
class CircularQueue:
    def __init__(self, k):
        self.arr = [0] * k
        self.front = self.rear = 0
        self.size = 0
        self.cap = k

    def enqueue(self, val):
        if self.size == self.cap: return False
        self.arr[self.rear] = val
        self.rear = (self.rear + 1) % self.cap
        self.size += 1
        return True

    def dequeue(self):
        if not self.size: return False
        self.front = (self.front + 1) % self.cap
        self.size -= 1
        return True
```

# Sliding Window Using Deque

```python
# Max in every window of size k
def max_sliding_window(arr, k):
    dq = deque()
    result = []
    for i in range(len(arr)):
        while dq and arr[dq[-1]] < arr[i]:
            dq.pop()
        dq.append(i)
        if dq[0] == i - k:
            dq.popleft()
        if i >= k - 1:
            result.append(arr[dq[0]])
    return result
```
