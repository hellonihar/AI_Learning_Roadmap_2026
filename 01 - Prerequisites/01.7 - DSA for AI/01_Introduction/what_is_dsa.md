# What is DSA & Why It Matters

## What is DSA?
- **Data Structures** — ways to organize data (arrays, trees, graphs, hash maps)
- **Algorithms** — steps to solve problems (searching, sorting, traversing)

## Why It Matters for AI Engineers

| Reason | Why |
|---|---|
| **Efficient code** | O(n²) vs O(n log n) matters when processing millions of data points |
| **Model optimization** | K-d trees for nearest neighbors, priority queues for beam search |
| **Interviews** | FAANG+ ML roles require DSA rounds |
| **Framework internals** | Understand how TensorFlow/pandas/NumPy work under the hood |
| **Custom operations** | Implement custom layers, loss functions, or data pipelines |

## When You Use DSA in AI

### Prefix Sums
1. **Time-series cumulative metrics** — Compute running accuracy, loss, or F1 at each training step: `prefix[i] = prefix[i-1] + metric[i]` then plot or early-stop based on running average.
2. **Feature normalization** — Precompute prefix sums of pixel values to implement integral-image-based filters (e.g., Viola-Jones face detection) or fast normalized cross-correlation.

### Sliding Window
1. **Rolling window statistics** — Moving average of model loss over the last 100 batches to smooth the training curve and detect divergence earlier than point-by-point.
2. **Sequence truncation** — When tokenizing long documents for LLMs (context window = 4096), slide a window with overlap to chunk text into manageable segments while preserving boundary context.

### Hashing
1. **Feature hashing** — Map high-cardinality categorical features (millions of unique words or user IDs) to a fixed-size feature vector using a hash function, avoiding massive one-hot matrices in linear models and embeddings.
2. **Vocabulary lookup** — Implement token-to-ID mapping in NLP pipelines using a hash map (`{token: id}`) for O(1) lookups instead of O(n) linear scans over 100K+ vocabularies.

### Trees
1. **Decision trees & random forests** — During training, the tree recursively partitions the feature space; each node is a binary decision (feature < threshold). Inference is O(depth) — much faster than distance-based methods at scale.
2. **Hierarchical clustering** — Build a dendrogram (tree) to visualize how data points group at different granularities. Used in customer segmentation, gene expression analysis, and document topic organization.

### Graphs
1. **Knowledge graphs** — Entities (nodes) connected by relations (edges): `(Einstein) —[born_in]→ (Germany)`. Used in RAG applications to retrieve structured relationships beyond vector similarity.
2. **Neural network computation graph** — Each operation (matrix multiply, ReLU, softmax) is a node in a DAG. Backpropagation is a topological-order traversal — reverse graph traversal computing gradients.

### Sorting
1. **Ranking predictions** — Sort model outputs by confidence score before returning top-k results. Used in recommendation systems, search ranking, and object detection confidence thresholding.
2. **Non-maximum suppression (NMS)** — In object detection, sort bounding boxes by confidence, then greedily remove boxes with high IoU overlap. Requires sorting + iterative greedy scan.

### Binary Search
1. **Threshold tuning** — Binary search over probability thresholds to maximize F1 score. Instead of a grid search over 100 thresholds, binary search finds the optimal one in O(log n) iterations.
2. **Learning rate search** — Implement a learning rate finder that exponentially increases LR and binary-searches the point where loss starts diverging, giving you the optimal max LR for cyclic schedules.
