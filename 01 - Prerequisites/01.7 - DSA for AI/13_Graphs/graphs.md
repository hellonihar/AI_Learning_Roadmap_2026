# Adjacency List Representation

```python
graph = defaultdict(list)
graph[0].append((1, 4))  # (neighbor, weight)
graph[1].append((2, 3))
```

# Breadth First Search (BFS)

```python
from collections import deque

def bfs(graph, start):
    visited = {start}
    q = deque([start])
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return order
```

# Depth First Search (DFS)

```python
def dfs(graph, start):
    visited = set()
    order = []
    def explore(node):
        visited.add(node)
        order.append(node)
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                explore(neighbor)
    explore(start)
    return order
```

# Shortest Path (Dijkstra)

```python
import heapq

def dijkstra(graph, start):
    distances = {node: float("inf") for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        dist, node = heapq.heappop(pq)
        if dist > distances[node]: continue
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    return distances
```

# Directed Acyclic Graphs (DAG) & Topological Sort

```python
def topological_sort(graph):
    visited = set()
    result = []
    def dfs(node):
        visited.add(node)
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
        result.append(node)
    for node in graph:
        if node not in visited:
            dfs(node)
    return result[::-1]  # reverse for topological order
```

# Disjoint Set Union (Union Find)

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

# Minimum Spanning Tree (Kruskal)

```python
def kruskal(n, edges):
    edges.sort(key=lambda e: e[2])  # (u, v, weight)
    dsu = DSU(n)
    mst_weight = 0
    for u, v, w in edges:
        if dsu.union(u, v):
            mst_weight += w
    return mst_weight
```

# State Space Graphs

BFS/DFS over implicit graph where nodes are states and edges are valid transitions.

```python
# Example: shortest path in a maze
from collections import deque

def shortest_path(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    q = deque([(start[0], start[1], 0)])
    visited = {start}
    while q:
        r, c, dist = q.popleft()
        if (r, c) == end: return dist
        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                q.append((nr, nc, dist + 1))
    return -1
```
