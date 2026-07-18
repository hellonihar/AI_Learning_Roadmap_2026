# Vector Databases

## 1. Introduction to Vector Databases

### Why Keyword Search Is Insufficient for Semantic Tasks

Keyword search (BM25, TF-IDF, SQL `LIKE`) matches exact or near-exact word patterns. It has no understanding of meaning.

```
Query: "cheap hotels in Paris"

Keyword search matches:
  ✅ "The hotel was cheap" (has "cheap" + "hotel")
  ✅ "Paris hotels are expensive" (has "Paris" + "hotels")
  ❌ "Budget accommodations in the City of Light" (no keywords match, but it's the most relevant result)
  ❌ "Petit prix à Paris" (French translation — 0 keyword overlap)
  ✅ "Cheap hotels in Paris near Eiffel Tower" (exact match, good)
  ❌ "Expensive hotels in Paris" (has "hotels" + "Paris" but wrong meaning)
```

| Limitation | Why It Matters |
|---|---|
| **Synonym blindness** | "car" ≠ "vehicle", "cheap" ≠ "budget", "buy" ≠ "purchase" |
| **Polysemy confusion** | "Apple" (fruit) vs "Apple" (company) — same keyword, different meaning |
| **Language mismatch** | Query in English, documents in French — no overlap |
| **Concept matching** | "City of Light" → Paris — zero keyword overlap with "Paris" |
| **Spelling variation** | "color" vs "colour", "behaviour" vs "behavior" |
| **Order sensitivity** | "Paris hotels" and "hotels in Paris" are the same meaning but different words |

**Keyword search is fast and works for exact matches.** For semantic understanding — finding things by meaning, not by exact words — it fundamentally fails.

### From Symbolic Matching → Semantic Similarity

The evolution of search:

```
Stage 1: Exact match
  "hotel" → matches only "hotel"
  Query: "cheap hotels" → matches only documents with exactly "cheap" and "hotel"

Stage 2: Keyword search with stemming
  "hotel" → matches "hotel", "hotels", "hoteling"
  Better, but still symbolic — no understanding of meaning.

Stage 3: Vector search via embeddings
  "cheap hotels in Paris" → [0.23, -0.45, 0.12, ...] → finds semantically similar content
  Matches: "budget accommodation near the Eiffel Tower", "affordable places to stay in the City of Light"
```

Vector search doesn't look for words — it looks for **meaning**. Two pieces of text are "close" if they mean similar things, regardless of word overlap.

### Embeddings as Numerical Representations

An embedding is a list of numbers (a vector) that captures the meaning of a piece of text.

```
"cat" → [0.23, -0.45, 0.12, 0.89, -0.33, ...]  (1536 dimensions for OpenAI embeddings)
"dog" → [0.25, -0.42, 0.15, 0.85, -0.30, ...]  (similar — close in vector space)
"bicycle" → [0.01, 0.67, -0.89, -0.12, 0.55, ...]  (different — far from "cat")
```

**Key properties:**
- Similar text → similar vectors (high cosine similarity)
- Dissimilar text → different vectors (low cosine similarity)
- The vector space is learned, not hand-crafted — it captures semantic relationships discovered during training
- The number of dimensions (768, 1024, 1536, 4096) balances expressiveness vs storage cost

### Why Similarity Search Requires Specialized Indexing

Naive vector search — comparing a query vector against every stored vector — is O(n). For small datasets (< 10K), this is fine. For production scale (1M–1B vectors), it's impossible.

```
Dataset size: 10M vectors × 1536 dimensions
Naive search: Compare query to all 10M vectors → 10M distance calculations → ~10 seconds per query
                 → Unusable for real-time search

With indexing (HNSW, IVF): → ~10ms per query
                 → 1000× faster
```

**Specialized indexing structures** (HNSW, IVF, PQ) enable sub-linear search — finding the nearest neighbors without scanning every vector. This is what makes vector databases practical at scale.

### Vector DB vs Traditional Database

| Dimension | Traditional DB (PostgreSQL, MySQL) | Vector DB (Pinecone, Weaviate, Milvus) |
|---|---|---|
| **Index type** | B-tree, hash index | HNSW, IVF, PQ |
| **Query type** | Exact match, range, join | Similarity search (ANN) |
| **Search method** | `WHERE col = value` | `ORDER BY similarity(embedding, query) LIMIT K` |
| **Indexed data** | Strings, numbers, dates | Vectors (float arrays) |
| **Scales to** | Billions of rows | Billions of vectors |
| **Consistency** | ACID | Eventually consistent (most) |
| **Use case** | Transactions, reporting, analytics | Semantic search, RAG, recommendations |

**Hybrid approach:** Modern systems use both. PostgreSQL with pgvector extension or Weaviate/Milvus with built-in scalar filtering. Metadata (dates, categories) stored in traditional columns with B-tree indexes; vectors stored in ANN indexes.

---

## 2. Embeddings & Vector Space Fundamentals

### What Embeddings Represent

An embedding is a **compressed representation of meaning** — a dense vector where each dimension captures some learned aspect of the content.

```
"I love this product, it's amazing"
  → [0.32, -0.12, 0.87, 0.04, -0.55, ...]

Embeds:
  - Sentiment (positive: high activation on certain dimensions)
  - Topic (product review)
  - Intensity (amazing → strong positive)
  - Syntax (grammatical structure patterns)
  - Context (likely from an e-commerce platform)
```

**No individual dimension has a human-readable meaning.** You can't say "dimension 42 = positivity." The meaning is distributed across all dimensions — each one captures a tiny, non-interpretable signal. The overall geometry encodes the semantics.

### Cosine Similarity vs Dot Product vs Euclidean Distance

| Metric | Formula | Range | When to Use |
|---|---|---|---|
| **Cosine similarity** | cos(θ) = A·B / (‖A‖ × ‖B‖) | [-1, 1] | Default for text embeddings (normalized vectors) |
| **Dot product** | A·B = Σ(Ai × Bi) | [-∞, ∞] | When vectors are not normalized (OpenAI, Cohere) |
| **Euclidean distance (L2)** | ‖A - B‖ = sqrt(Σ(Ai − Bi)²) | [0, ∞] | When magnitude matters (recommendation systems) |

```
Two vectors in 2D:
  A = [1, 0], B = [0, 1]

  Cosine similarity: (1×0 + 0×1) / (1×1) = 0  → orthogonal, no similarity
  Dot product: 1×0 + 0×1 = 0  → no overlap
  Euclidean distance: sqrt((1-0)² + (0-1)²) = sqrt(2) ≈ 1.41  → far apart
```

**Practical rule:**
- Most text embeddings are normalized (length = 1). For normalized vectors, cosine similarity and dot product give the same ranking (they differ by a constant factor).
- OpenAI `text-embedding-3-*` models produce normalized vectors → use cosine similarity or dot product (they're equivalent).
- If you don't know which metric your embedding model uses, default to cosine similarity — it's the safest choice.

```
Normalized vector: ‖v‖ = 1  →  cos(A,B) = A·B  (identical ranking with dot product)
Unnormalized vector: cos(A,B) ≠ A·B  → choose based on whether magnitude matters
```

### High-Dimensional Intuition

Human intuition breaks down above 3 dimensions. High-dimensional spaces behave counterintuitively.

**The "curse of dimensionality":**

| Dimension | Volume of unit cube | Property |
|---|---|---|
| 1 | 1.0 | Easy — points spread along a line |
| 2 | 1.0 | Familiar — points on a plane |
| 3 | 1.0 | Visualizable — points in space |
| 10 | 1.0 | Max distance ≈ 2× min distance (all points are "far apart") |
| 100 | 1.0 | All pairwise distances are nearly equal — "nearest neighbor" is barely closer than "farthest neighbor" |
| 1536 | 1.0 | **Every pair of random vectors is approximately orthogonal and equidistant** |

**What this means for vector search:**
- In 1536-dimensional space, every random point is far from every other random point
- Meaningful similarity only works because embeddings aren't random — they're structured by training
- The difference between "similar" and "dissimilar" is real but small in absolute terms
- Indexing algorithms (HNSW, IVF) are essential because brute-force in high dimensions is expensive and the distance signal is weak

### Why Semantically Similar Text Clusters in Space

Embedding models are trained to pull similar text together and push dissimilar text apart.

```
Clusters formed by a sentence embedding model:

[Cluster: Positive reviews]
  "This product is amazing"
  "I love it, works perfectly"
  "Best purchase I've made this year"

[Cluster: Negative reviews]
  "Terrible quality, broke in a week"
  "Waste of money, don't buy"
  "Worst customer service experience"

[Distance between clusters]
  Positive centroid ↔ Negative centroid: 0.85 (far apart)
  Within positive cluster: average distance 0.15 (close together)
```

This clustering is learned, not programmed. The model discovers that positive words appear in similar contexts during training, so their vectors end up close together.

**Practical implication:** Vector search works because similar documents cluster. The retrieval quality depends on how well your embedding model separates clusters for your specific domain.

### Model-Dependent Embedding Spaces

Different embedding models produce different vector spaces. **You cannot mix embeddings from different models.**

```
OpenAI text-embedding-3-small: 1536 dimensions
  "dog" → [0.23, -0.45, ...]  ← Space A

Sentence-BERT (all-MiniLM-L6-v2): 384 dimensions
  "dog" → [0.56, -0.12, ...]  ← Space B (completely different)
```

**Rules:**
1. **Use the same model for indexing and querying.** The query must be embedded with the same model that embedded your documents.
2. **If you change models, re-embed everything.** There is no translation between embedding spaces.
3. **Different models have different strengths.** OpenAI's v3 models are good for general text. Cohere's embed-multilingual is better for multilingual. Sentence-BERT models are smaller and self-hostable.

| Model | Dimensions | Best For | Cost |
|---|---|---|---|
| OpenAI text-embedding-3-small | 1536 | General purpose, high quality | $0.02/1M tokens |
| OpenAI text-embedding-3-large | 3072 | Highest quality (marginally better) | $0.13/1M tokens |
| Cohere embed-english-v3 | 1024 | English, classification-aware | $0.10/1M tokens |
| Cohere embed-multilingual-v3 | 1024 | 100+ languages | $0.10/1M tokens |
| Sentence-BERT all-MiniLM-L6-v2 | 384 | Self-hosted, fast, cheap | Free (local) |
| Sentence-BERT all-mpnet-base-v2 | 768 | Self-hosted, best quality local | Free (local) |

---

## 3. Ecosystem & Tools

### FAISS (Local Indexing)

**FAISS** (Facebook AI Similarity Search) is a library, not a database. It provides vector indexing and search algorithms without a server, persistence layer, or management interface.

```python
import faiss
import numpy as np

# Create an index
dimension = 1536
index = faiss.IndexFlatL2(dimension)  # Brute-force L2 (fine for < 100K vectors)

# Or use HNSW for large-scale approximate search
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors per node

# Add vectors
vectors = np.random.rand(10000, dimension).astype('float32')
index.add(vectors)

# Search
query = np.random.rand(1, dimension).astype('float32')
distances, indices = index.search(query, k=5)
```

| Strength | Weakness |
|---|---|
| Fastest indexing and search | No built-in persistence (you manage saving/loading) |
| Full control over index parameters | No CRUD (updating individual vectors requires rebuilding) |
| Free, open source (MIT) | No query language — pure Python/C++ API |
| Minimal dependencies | No distributed/replicated setup |
| Supports GPU acceleration | No metadata filtering (must be handled separately) |

**When to use FAISS:**
- Batch processing (index once, query many times)
- Datasets that fit on a single machine (< 10M vectors)
- When you want full control and minimal infrastructure

### Pinecone (Managed SaaS)

**Pinecone** is a fully managed vector database. You upload vectors via API, and it handles indexing, scaling, and serving.

```python
import pinecone

pinecone.init(api_key="...")
index = pinecone.Index("my-index")

# Upsert vectors
index.upsert([
    ("id1", [0.1, 0.2, ...], {"category": "tech", "date": "2026-01-15"}),
    ("id2", [0.3, 0.4, ...], {"category": "finance", "date": "2026-02-20"}),
])

# Query
results = index.query(
    vector=[0.15, 0.25, ...],
    top_k=5,
    filter={"category": {"$eq": "tech"}}
)
```

| Strength | Weakness |
|---|---|
| Zero infrastructure management | Vendor lock-in (proprietary format) |
| Automatic scaling | Most expensive at scale |
| Built-in metadata filtering | Limited to Pinecone SDK (no SQL/gRPC) |
| Multi-tenancy support | No self-hosted option (unless using Pinecone for AWS) |
| Fast indexing (real-time updates) | Latency can vary under load |

**Pricing (approximate):** $0.10–0.30 per million vectors per month (varies by pod type and region).

**When to use Pinecone:**
- Teams with no DevOps/MLOps resources
- Need to get to production quickly
- Willing to pay a premium for managed infrastructure

### Weaviate (Open Source / Managed)

**Weaviate** is an open-source vector database with built-in vectorization, hybrid search, and GraphQL API.

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Define schema
client.schema.create_class({
    "class": "Document",
    "properties": [
        {"name": "title", "dataType": ["text"]},
        {"name": "content", "dataType": ["text"]},
        {"name": "category", "dataType": ["text"]},
    ],
    "vectorizer": "text2vec-openai"  # Automatic vectorization
})

# Import (automatically vectorized if vectorizer is configured)
client.data_object.create({
    "title": "Vector Databases Explained",
    "content": "Vector databases store embeddings...",
    "category": "tech"
}, "Document")

# Hybrid search (keyword + vector with weighted combination)
response = client.query.get(
    "Document", ["title", "content", "_additional {certainty}"]
).with_hybrid("vector database storage", alpha=0.75).do()
```

| Strength | Weakness |
|---|---|
| Open source (BSD) | Complex operational setup (Kubernetes recommended) |
| Built-in vectorization (connect to OpenAI, Cohere, etc.) | Schema management overhead |
| Hybrid search (BM25 + vector) out of the box | Slower than FAISS for pure vector search |
| GraphQL + REST APIs | Smaller ecosystem than Pinecone |
| CRUD with ACID-like transactions | Resource-intensive (Java-based) |

**When to use Weaviate:**
- Need hybrid search (keyword + semantic)
- Want built-in vectorization (save API calls)
- Have DevOps capacity to manage it
- Value open source with option for managed service

### Milvus (Open Source / Distributed)

**Milvus** is a cloud-native vector database designed for billion-scale similarity search, built on a distributed architecture with separate storage and compute layers.

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

connections.connect(host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
]
schema = CollectionSchema(fields, "Document collection")
collection = Collection("documents", schema)

# Create index
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 1024}
}
collection.create_index("embedding", index_params)

# Search
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=5,
    expr="category == 'tech'"
)
```

| Strength | Weakness |
|---|---|
| Billion-scale indexing | Overkill for small datasets (< 1M vectors) |
| Multiple index types (HNSW, IVF, PQ, DiskANN) | Complex deployment (Kubernetes + etcd + storage) |
| GPU acceleration for indexing | Steep learning curve |
| Hybrid search + scalar filtering | Parameter tuning needed for optimal performance |
| Cloud-native (separate compute/storage) | Expensive at small scale (high infra overhead) |

**When to use Milvus:**
- > 10M vectors
- Need distributed, scalable infrastructure
- Have ML infrastructure team to manage it

### Chroma (Embedded / Lightweight)

**Chroma** is a lightweight, open-source vector database designed for developer prototyping and small-scale applications.

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

# Add documents (automatically handles embeddings or accepts pre-computed)
collection.add(
    documents=["Vector databases store embeddings", "Embeddings capture semantics"],
    metadatas=[{"category": "tech"}, {"category": "tech"}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["What are vector databases?"],
    n_results=2
)
```

| Strength | Weakness |
|---|---|
| Simplest setup (pip install) | Not production-ready for scale |
| Automatic embedding (uses Sentence-BERT by default) | Limited filtering and query capabilities |
| In-process (no server needed) | No built-in sharding or replication |
| Great for prototyping and experimentation | Memory-bound for large datasets |
| Apache 2.0 license | Slower than FAISS for large index |

**When to use Chroma:**
- Prototyping and experimentation
- Small applications (< 100K vectors)
- Local development and testing
- Learning vector databases

### Production-Focused Comparison

| Criterion | FAISS | Pinecone | Weaviate | Milvus | Chroma |
|---|---|---|---|---|---|
| **Type** | Library | Managed SaaS | Open source DB | Open source DB | Embedded DB |
| **Scale (vectors)** | < 10M (single machine) | < 1B (auto-scaling) | < 10M (single), < 100M (cluster) | < 1B+ (distributed) | < 100K |
| **Query latency (p99)** | 1-5ms (in-memory) | 5-20ms | 10-50ms | 5-30ms | 5-20ms |
| **Metadata filtering** | Manual (external) | Built-in (indexed) | Built-in (indexed) | Built-in (indexed) | Basic |
| **Hybrid search** | No (vector only) | No (vector only) | Yes (BM25 + vector) | Yes (BM25 + vector) | No |
| **Persistence** | Manual (save/load) | Automatic | Automatic | Automatic | Automatic (persistent client) |
| **Replication/HA** | No | Yes (managed) | Yes (K8s) | Yes (built-in) | No |
| **GPU support** | Yes | No | No | Yes | No |
| **Setup complexity** | Low | Low | Medium | High | Very low |
| **Production readiness** | Medium (needs wrapper) | High | Medium-High | High | Low |
| **Cost (est. 1M vectors)** | Free (infra cost only) | $100-300/month | Infra cost + ops | Infra cost + ops | Free |
| **Best for** | High-performance local search | Quick production deployment | Hybrid search | Billion-scale | Prototyping |

### Selection Guide

```
How many vectors?
├── < 100K → Chroma (prototype) or FAISS (if production)
├── 100K – 10M → FAISS (if self-hosted) or Pinecone (if managed)
└── > 10M → Milvus (if self-hosted) or Pinecone (if managed)

Need hybrid search?
├── Yes → Weaviate or Milvus
└── No → FAISS or Pinecone

Have DevOps?
├── Yes → Milvus or Weaviate (more control, lower cost at scale)
└── No → Pinecone (fully managed)

Need real-time updates?
├── Yes → Pinecone or Weaviate (streaming CRUD)
└── No → FAISS (batch rebuild is fine)

Budget constrained?
├── Yes → FAISS or Milvus (self-hosted, open source)
└── No → Pinecone (pay for convenience)

Production readiness:
├── Quick launch → Pinecone
├── Full control → Milvus
├── Hybrid search → Weaviate
├── High-performance local → FAISS
└── Prototyping → Chroma
```

### Vector DB Integration with LLM Pipelines

```
Document ingestion:
  [Documents] → [Chunking] → [Embedding model] → [Vector DB index]

Query:
  [User query] → [Embedding model] → [Vector DB search] → [Top-K results]
                                                              ↓
                                              [LLM context assembly] → [LLM response]
```

**Common stack:**
```
Embedding: OpenAI text-embedding-3-small / Cohere embed-v3 / Sentence-BERT
Vector DB: Pinecone (managed) or Milvus (self-hosted)
LLM: GPT-4o / Claude / Llama 3
Orchestration: LangChain / LlamaIndex / custom
Chunking: LangChain text splitters / custom
```

**Latency budget for vector search in a RAG pipeline (target: < 500ms total):**
```
Embedding:        100-300ms  (API call to embedding model)
Vector search:     10-50ms   (HNSW-indexed, 1M vectors)
Metadata filter:    5-20ms   (depends on filter complexity)
Network:            10-50ms   (client → vector DB)
Total:            125-420ms  (fits within 500ms budget)
```
