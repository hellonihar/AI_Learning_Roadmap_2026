# Tree Node

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

# Inorder Traversal (Left → Root → Right)

```python
def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def inorder_iterative(root):
    stack, result = [], []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    return result
```

# Preorder Traversal (Root → Left → Right)

```python
def preorder(root):
    if not root: return []
    return [root.val] + preorder(root.left) + preorder(root.right)
```

# Postorder Traversal (Left → Right → Root)

```python
def postorder(root):
    if not root: return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

# Binary Search Tree

```python
def search_bst(root, val):
    if not root or root.val == val: return root
    if val < root.val: return search_bst(root.left, val)
    return search_bst(root.right, val)

def insert_bst(root, val):
    if not root: return TreeNode(val)
    if val < root.val: root.left = insert_bst(root.left, val)
    else: root.right = insert_bst(root.right, val)
    return root

def is_valid_bst(root, lo=float("-inf"), hi=float("inf")):
    if not root: return True
    if not (lo < root.val < hi): return False
    return is_valid_bst(root.left, lo, root.val) and is_valid_bst(root.right, root.val, hi)
```

# N-ary Tree

```python
class NaryNode:
    def __init__(self, val=0, children=None):
        self.val = val
        self.children = children or []

def nary_dfs(root):
    if not root: return []
    result = [root.val]
    for child in root.children:
        result.extend(nary_dfs(child))
    return result
```

# Tree Dynamic Programming

```python
# Max depth of binary tree
def max_depth(root):
    if not root: return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# Diameter of binary tree (longest path between any two nodes)
def diameter(root):
    result = 0
    def dfs(node):
        nonlocal result
        if not node: return 0
        left = dfs(node.left)
        right = dfs(node.right)
        result = max(result, left + right)
        return 1 + max(left, right)
    dfs(root)
    return result
```
