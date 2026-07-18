# Retrieval-Augmented Generation (RAG)

## 1. Introduction to RAG

### Why LLMs Hallucinate

LLMs hallucinate because they're **next-token predictors, not fact databases**. The model generates the most probable continuation, not the truth.

```
"Einstein invented the ___"
→ "theory of relativity" (probability 0.87) — correct
→ "telephone" (probability 0.0001) — highly unlikely
→ "light bulb" (probability 0.02) — plausible but wrong
```

Without external grounding, the model relies entirely on patterns learned during training. If the pattern is incomplete, outdated, or ambiguous, the model guesses — and sometimes guesses wrong. Hallucination is not a bug; it's the default behavior of a probabilistic system operating without verified facts.

### Parametric Memory vs Non-Parametric Memory

| Memory Type | What It Is | Stored In | Characteristics |
|---|---|---|---|
| **Parametric** | Knowledge encoded in model weights during training | Model parameters (billions of floats) | Fixed at training time, expensive to update, always available (no lookup needed) |
| **Non-parametric** | External knowledge retrieved at inference time | Vector DB, document store, database | Updated independently, cheap to modify, requires retrieval step |

**Parametric memory problems:**
- Knowledge cutoff (model doesn't know recent events)
- Training data may be wrong or biased
- Updating requires full retraining or fine-tuning (expensive)
- Model "overwrites" low-frequency facts with high-frequency patterns

**Non-parametric memory advantages:**
- Always current (update the document, not the model)
- Factual grounding reduces hallucination
- Source attribution is possible ("according to document X")
- Domain adaptation without retraining

### The Limitation of Training-Time Knowledge

A model's knowledge is frozen at its training cutoff. Any fact that changes after that date is unknown to the model.

```
Model trained: January 2025
Knowledge cutoff: December 2024

User asks in July 2026:
  "Who is the CEO of company X?"
  "What is the latest product version?"
  "What is the current interest rate?"

→ Model knows nothing about events after December 2024
→ Without RAG, it must guess or admit ignorance
→ With RAG, it retrieves current documents and answers accurately
```

**Training cannot solve this problem.** You would need to retrain daily — which is impractical ($1M+ per training run for large models). Even fine-tuning weekly is expensive and risks catastrophic forgetting. RAG solves it by keeping knowledge in a separate, independently updated store.

### RAG as "Retrieve → Inject → Generate"

```
User query: "What is the company policy on remote work for Q3 2026?"

Step 1: Retrieve
  └── Search policy documents → Top 3 relevant chunks

Step 2: Inject
  └── Build context: [system prompt] + [retrieved chunks] + [user query]

Step 3: Generate
  └── LLM reads context → generates grounded answer
```

The LLM doesn't need to "know" the policy. It only needs to read, understand, and summarize what's in the retrieved documents. This shifts the burden from **storing facts in model weights** to **retrieving facts from a searchable index**.

### When to Use RAG vs Fine-Tuning

| Criterion | RAG | Fine-Tuning |
|---|---|---|
| **Knowledge freshness** | Real-time (update documents) | Stale until retrained |
| **Custom domain data** | Excellent (retrieve from your docs) | Good (learns patterns) |
| **Reducing hallucination** | Strong (grounds in retrieved text) | Weak (still parametric, no grounding) |
| **Teaching new formats/tone** | Weak (can't change model behavior easily) | Strong (trains on examples) |
| **Latency** | Slower (retrieval step adds 50-500ms) | Same as base model |
| **Cost** | Retrieval + larger context (more tokens) | Training + inference |
| **Infrastructure** | Retrieval pipeline + vector DB | GPU compute + training pipeline |
| **When to choose** | Facts change, need citations, need audit trail | Need consistent formatting/tone, task-specific behavior |

**Best use cases:**
- **RAG:** Customer support (knowledge base changes weekly), legal research (documents are the source of truth), product docs (multiple versions)
- **Fine-tuning:** Output format control (always output JSON), tone adaptation (always be formal), task specialization (always summarize in 3 bullet points)
- **Both:** RAG for facts + fine-tuning for behavior. Most production systems use both.

---

## 2. Core RAG Pipeline

```
User Query
    │
    ▼
┌─────────────────────┐
│ 2. Embed query       │ ← Query embedding model
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 3. Vector DB search  │ ← Similary search (ANN)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 4. Retrieve top-K    │ ← Top relevant document chunks
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 5. Inject into       │ ← Context assembly
│    context           │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 6. Generate          │ ← LLM reads context + query
│    response          │
└────────┬────────────┘
         │
         ▼
   Grounded Response
```

### Step-by-Step

**Step 1: User query**
```
"Does the company allow remote work for international employees?"
```

**Step 2: Embed query**
```python
query_embedding = embedding_model.encode("Does the company allow remote work for international employees?")
```

**Step 3: Vector DB search**
```python
results = vector_db.search(
    query_embedding,
    top_k=3,
    filter={"category": "hr_policy"}  # Metadata filter
)
```

**Step 4: Retrieve top-K chunks**
```
Result 1: "International remote work is permitted for roles that can be performed asynchronously..."
Result 2: "Employees working from outside the home country must comply with local tax laws..."
Result 3: "All remote work arrangements must be approved by the employee's manager..."
```

**Step 5: Inject into context**
```python
context = f"""
Document excerpts:
{result_1}
{result_2}
{result_3}

User question: {query}

Answer the question using ONLY the document excerpts above. If the documents don't contain the answer, say "I don't have that information."
"""
```

**Step 6: Generate response**
```python
response = llm.generate(context)
# → "International remote work is permitted for roles that can be performed asynchronously. Employees working from outside the home country must comply with local tax laws. All remote work arrangements require manager approval."
```

---

## 3. Data Preparation & Indexing

### Document Ingestion

The pipeline from raw documents to searchable index:

```
Raw documents (PDFs, web pages, wikis, databases)
    │
    ▼
Text extraction (PDF parser, HTML stripper, DB export)
    │
    ▼
Cleaning & preprocessing
    │
    ▼
Chunking (split into bite-sized pieces)
    │
    ▼
Metadata tagging (source, date, category, author)
    │
    ▼
Embedding generation (convert each chunk to vector)
    │
    ▼
Index construction (build ANN index for fast search)
```

### Cleaning & Preprocessing

| Artifact | Remedy |
|---|---|
| **HTML tags, scripts, styles** | Strip with BeautifulSoup, trafilatura |
| **Boilerplate (nav, footers, ads)** | Content extraction libraries |
| **Special characters, encoding issues** | Normalize to UTF-8, remove control chars |
| **Duplicate or near-duplicate content** | Deduplicate via hash or embedding similarity |
| **PII, sensitive data** | Redact or filter before indexing |
| **Empty or near-empty chunks** | Remove (adds noise, no signal) |

**Why cleaning matters:** Garbage in → garbage out. Noisy documents produce bad embeddings, retrieve poorly, and waste context budget. A clean index is the foundation of a good RAG system.

### Chunking Strategies

Chunking is the single most impactful parameter in RAG quality. Too small → missing context. Too large → diluted signal.

| Strategy | Chunk Size | Overlap | Best For |
|---|---|---|---|
| **Fixed-size** | 256–1024 tokens | 10–20% | Simple, works well for most cases |
| **Paragraph-based** | 1–3 paragraphs | 1 sentence | Articles, documentation |
| **Sentence-based** | 1–5 sentences | None | Facts, Q&A pairs |
| **Semantic** | Variable (split at topic boundaries) | None | Long documents, books |
| **Recursive** | Hierarchical (page → section → paragraph) | Per-level | Deep document understanding |

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,      # tokens per chunk
    chunk_overlap=50,    # overlap between chunks
    separators=["\n\n", "\n", ".", " ", ""],
    length_function=len,  # or token counter
)

chunks = splitter.split_documents(documents)
```

**Chunk size guidelines:**

| Chunk Size | Use Case | Pros | Cons |
|---|---|---|---|
| 128–256 tokens | Precise Q&A, fact lookup | High precision, low noise | May miss context |
| 512–1024 tokens | General RAG, documentation | Balanced | Good default |
| 1024–2048 tokens | Summarization, long-form | Captures full context | Noisy, expensive |

### Chunk Overlap

Overlap prevents information loss at chunk boundaries.

```
No overlap:
  Chunk 1: "The capital of France is Paris. It is known for"
  Chunk 2: "its cuisine and the Eiffel Tower."
  → Query: "What is Paris known for?" retrieves Chunk 2 — misses "Paris"

With overlap (50 tokens):
  Chunk 1: "The capital of France is Paris. It is known for its cuisine"
  Chunk 2: "known for its cuisine and the Eiffel Tower."
  → Query retrieves either chunk — both have the full context
```

Overlap of 10–20% of chunk size is recommended. Too much overlap creates redundant chunks (waste of storage + retrieval noise).

### Metadata Tagging

Every chunk should carry metadata that can be used for filtering, citation, and ranking.

```python
chunk = {
    "text": "International remote work is permitted for roles that can be performed asynchronously.",
    "metadata": {
        "source": "hr_policy_2026.pdf",
        "page": 12,
        "section": "Remote Work",
        "date_modified": "2026-01-15",
        "category": "hr_policy",
        "author": "HR Department",
        "version": "3.2"
    }
}
```

**Metadata use cases:**
- **Filtering:** `category == "hr_policy"` — only search specific document sets
- **Attribution:** "According to the HR policy (v3.2, page 12)..."
- **Recency weighting:** Prefer newer documents over older ones
- **Access control:** Only retrieve documents the user has permission to see

### Embedding Generation

Convert each chunk to a vector using an embedding model.

```python
from openai import OpenAI

client = OpenAI()
chunks = ["chunk text 1", "chunk text 2", ...]

# Batch for efficiency (most APIs support batch embedding)
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks
)

embeddings = [item.embedding for item in response.data]
```

**Batch size guidelines for embedding APIs:**
- OpenAI: batch up to 2048 inputs per request
- Cohere: batch up to 96 inputs per request
- Sentence-BERT: batch up to 256 (dependent on GPU memory)

### Index Construction

Build the search index from embeddings + metadata.

```python
import pinecone

pinecone.init(api_key="...")
index = pinecone.Index("rag-docs")

# Upsert vectors with metadata
vectors = []
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    vectors.append({
        "id": f"chunk_{i}",
        "values": embedding,
        "metadata": chunk["metadata"]
    })

# Batch upsert
index.upsert(vectors=vectors, batch_size=100)
```

---

## 4. Retrieval Strategies

### Similarity Search Basics

The simplest retrieval: embed query, search vector DB, return top-K.

```
Query embedding → [0.12, -0.34, 0.56, ...]

Vector DB:
  [0.11, -0.33, 0.55, ...]  cosine: 0.98  ← most similar
  [0.45,  0.12, -0.78, ...]  cosine: 0.23  ← unrelated
  [0.13, -0.31, 0.52, ...]  cosine: 0.97  ← second most similar

→ Return top 2 chunks
```

**Limitation:** Pure vector search captures semantic similarity but ignores keyword overlap. A query about "refund policy" might retrieve documents about "return policy" (semantically similar) but miss an exact match on "refund timeline" due to embedding noise.

### Hybrid Search (BM25 + Vectors)

Combine keyword search (BM25) with vector search to get the best of both.

```python
def hybrid_search(query, alpha=0.7):
    """
    alpha = 0.0 → pure keyword search
    alpha = 1.0 → pure vector search
    alpha = 0.7 → weighted combination (default)
    """
    # Vector search
    query_embedding = embed(query)
    vector_results = vector_db.search(query_embedding, top_k=10)

    # Keyword search (BM25)
    keyword_results = bm25_search(query, top_k=10)

    # Combine scores
    combined = {}
    for doc, score in vector_results:
        combined[doc.id] = alpha * score
    for doc, score in keyword_results:
        combined[doc.id] = combined.get(doc.id, 0) + (1 - alpha) * score

    # Sort by combined score
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return ranked[:5]
```

| Search Type | Finds | Misses |
|---|---|---|
| **Vector (semantic)** | "budget hotels" → "cheap accommodation" | "Hotel California" (specific name, not semantic) |
| **Keyword (BM25)** | Exact match, product names, codes | Synonyms, paraphrases, different languages |
| **Hybrid** | Both | Nothing (practically) |

**Best practice:** Use hybrid search as default. Start with alpha=0.7 (favor vector) and tune based on your domain. Code-heavy or name-heavy domains benefit from lower alpha (more keyword weight).

### Re-Ranking Retrieved Documents

A second-stage re-ranker improves retrieval precision after the initial search.

```
Initial retrieval (vector or hybrid): Top 20 documents
                                     │
                                     ▼
                  Re-ranker (cross-encoder): Score each query-document pair
                                     │
                                     ▼
                           Return top 3 (re-ordered)
```

```python
from sentence_transformers import CrossEncoder

re_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Initial retrieval
candidates = hybrid_search(query, top_k=20)

# Re-rank
pairs = [(query, doc.text) for doc in candidates]
scores = re_ranker.predict(pairs)

# Sort by re-ranker score
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
top_k = ranked[:3]
```

| Re-ranker | Quality Gain | Latency | Cost |
|---|---|---|---|
| None | Baseline | 0 | Free |
| Lightweight cross-encoder | +10-20% | 50-200ms | Low |
| LLM-as-judge (GPT-4o mini) | +15-30% | 200-500ms | Medium |
| LLM-as-judge (GPT-4o) | +20-35% | 500-2000ms | High |

**Best practice:** Use a lightweight cross-encoder as the default re-ranker. Only use LLM-based re-ranking when retrieval quality is critical and latency budget allows it.

### Query Expansion

Improve retrieval by expanding the query with related terms or generated variants.

```python
def expand_query(query):
    """Generate expanded query variants using an LLM."""
    prompt = f"""
    Generate 3 alternative versions of this search query to improve retrieval:
    "{query}"
    Return as a list of strings.
    """
    variants = llm_call(prompt)  # ["remote work international policy", "work from home abroad rules", ...]
    return [query] + variants

def search_with_expanded_query(query):
    variants = expand_query(query)
    all_results = []
    for v in variants:
        embedding = embed(v)
        results = vector_db.search(embedding, top_k=5)
        all_results.extend(results)
    # Deduplicate and re-rank
    return deduplicate(all_results)
```

| Type | What It Does | Effect |
|---|---|---|
| **Synonym expansion** | "car" → "car, automobile, vehicle" | Broadens recall |
| **LLM-generated variants** | "remote work policy" → "telecommuting rules, work from home guidelines, location flexibility" | Captures different phrasings |
| **Back-translation** | "Remote work policy" → French → English | Multilingual robustness |
| **Query decomposition** | "What are the policies for remote work and stock options?" → 2 sub-queries | Multi-faceted queries |

### Multi-Query Retrieval

For complex queries, generate multiple sub-queries and retrieve for each.

```
Original query: "What are the policies for remote work and stock options for international employees?"

Decomposed:
  Q1: "remote work policy for international employees"
  Q2: "stock option eligibility for international employees"

Retrieve top-3 for each → Merge → Deduplicate → Re-rank → Top-5
```

### Filtering with Metadata

Use metadata filters to narrow search scope before similarity search.

```python
results = vector_db.search(
    query_embedding,
    top_k=5,
    filter={
        "$and": [
            {"category": {"$eq": "hr_policy"}},
            {"date_modified": {"$gte": "2026-01-01"}},
            {"department": {"$in": ["engineering", "all"]}}
        ]
    }
)
```

**When to use metadata filtering:**
- User specifies a category ("Show me HR policies")
- Access control (user can only see certain documents)
- Recency constraints ("Only documents from the last year")
- Domain filtering (search within a specific product line)

---

## 5. Context Construction in RAG

### Ordering Retrieved Chunks

The order of chunks in the context affects what the model pays attention to.

| Order | Effect | Best For |
|---|---|---|
| **Most relevant first** | Model attends to best chunk first | General RAG |
| **Original document order** | Maintains narrative flow | Long documents with sequential reasoning |
| **Reversed (most relevant last)** | Recency effect — model attends to last seen | When you want the model to end with the best evidence |

**Best practice:** Most relevant chunk first (primacy effect) + include source metadata so the model can rank them.

```python
context = ""
for i, chunk in enumerate(retrieved_chunks):
    context += f"[Source {i+1}: {chunk.source}, page {chunk.page}]\n{chunk.text}\n\n"
```

### Removing Redundancy

When multiple retrieved chunks contain overlapping information, deduplicate before injecting.

```
Retrieved:
  Chunk 1: "The refund policy allows returns within 30 days of purchase."
  Chunk 2: "Customers may return items within 30 days of the original purchase date for a full refund."

→ Chunk 2 is redundant with Chunk 1 (same information, slightly different wording)

Deduplicated context:
  [Source 1] The refund policy allows returns within 30 days of purchase.
  [Source 2] Items must be unopened and in original packaging. (Unique info, not redundant)
```

**Deduplication methods:**
- **Cosine similarity:** If two chunks have similarity > 0.95, keep only one
- **Token overlap:** If > 80% shared tokens, keep the longer/more recent one
- **LLM-based:** Expensive but accurate — ask the LLM "does this chunk add new information?"

### Separating Instructions from Retrieved Text

Always delimit retrieved text so the model treats it as data, not instructions.

```
Good:
  <documents>
  <doc source="hr_policy_2026" page="12">
  International remote work is permitted for roles that can be performed asynchronously.
  </doc>
  </documents>

Bad:
  Documents:
  International remote work is permitted for roles that can be performed asynchronously.
  → If the document text contains instruction-like phrases ("Ignore previous instructions..."), the model may follow them.

User query: Does the company allow remote work?
```

### Limiting Context Overflow

**The problem:** Retrieved chunks fill the context window, leaving no room for the response.

```
Context window: 8,000 tokens

System prompt:    500
Chat history:   1,000
Retrieved chunks (5 × 500):  2,500
User query:       100
──────────────────────────
Total input:    4,100
Room for output: 3,900  ← Plenty

vs

Retrieved chunks (15 × 500):  7,500
Total input:    9,100  ← Exceeds 8K window!
▸ Truncation happens automatically
▸ Usually drops from the middle (losing important chunks)
▸ Or response is cut off
```

**Solutions:**
- Limit K based on chunk size: `max_chunks = (window - overhead) / chunk_size`
- Dynamic chunk pruning: keep top K chunks that fit within budget
- Compress verbose chunks (summarize before injecting)
- Use models with larger context windows (128K+) to avoid overflow

### Source Attribution

Always include source information so the LLM can cite its sources.

```python
context = ""
for chunk in retrieved_chunks:
    context += f"""
<document>
<source>{chunk.source}</source>
<page>{chunk.page}</page>
<text>{chunk.text}</text>
</document>
"""

system_prompt = "Answer using only the provided documents. For each claim, cite the source document."
```

**Why attribution matters:**
- User can verify claims
- Builds trust ("the model showed its work")
- Debugging (which document caused a wrong answer?)
- Compliance (audit trail for automated decisions)

---

## 6. Advanced RAG Patterns

### Types of RAG

| Type | How It Works | Best Use Case | Complexity |
|---|---|---|---|
| **Naive RAG** | Retrieve → inject → generate (single pass) | Simple Q&A, small knowledge base | Low |
| **Iterative RAG** | Retrieve → generate → identify gaps → retrieve again | Complex questions needing multi-hop reasoning | Medium |
| **Self-RAG** | Model reflects on its own retrieved docs and answers | Reducing hallucination, improving factuality | High |
| **Corrective RAG** | Re-rank and filter retrieved docs before generation | Noisy retrieval environments | Medium |
| **Adaptive RAG** | Dynamically choose retrieval strategy based on query | Heterogeneous queries (some need retrieval, some don't) | High |
| **Graph RAG** | Build knowledge graph from docs, traverse for retrieval | Multi-entity questions, relationship-heavy domains | Very high |
| **Agentic RAG** | Agent decides when and what to retrieve using tools | Complex workflows with multiple information needs | Very high |

### Iterative Retrieval

When a single retrieval pass doesn't provide enough information, retrieve again based on what's missing.

```
Query: "What is the impact of the new tax law on our Q3 earnings?"

Pass 1: Retrieve about "new tax law" → Only general description
  → Gaps identified: "How does this affect our specific industry?"

Pass 2: Retrieve about "tax law effect on [our industry]"
  → More specific results

Pass 3: Generate final answer using both passes
```

**When to use:** Complex questions where the first retrieval doesn't surface enough specific information.

### Self-RAG / Reflective Retrieval

The model generates intermediate reasoning, evaluates whether retrieved documents support it, and decides whether to retrieve more or generate the final answer.

```
Query: "What is the company policy on data retention?"

Step 1: Generate initial answer from parametric knowledge
  → "Data is retained for 7 years" (model's best guess)

Step 2: Retrieve documents about "data retention policy"
  → "Customer data: 5 years, Financial data: 7 years, Employee data: 3 years"

Step 3: Compare. Model detects discrepancy.
  → "My initial answer was incomplete. The correct policy varies by data type."

Step 4: Generate final fact-checked answer with citations
```

**When to use:** High-stakes applications where hallucination is unacceptable (legal, medical, compliance).

### Corrective RAG

Re-rank and filter retrieved documents before generation, discarding low-quality or irrelevant chunks.

```python
def corrective_rag(query, retrieved_docs):
    filtered = []
    for doc in retrieved_docs:
        relevance = classify_relevance(query, doc.text)  # LLM call or classifier
        if relevance >= 0.7:
            filtered.append(doc)
        else:
            log_low_relevance(query, doc)  # Track for debugging

    if len(filtered) == 0:
        return "I couldn't find relevant information to answer your question."

    return generate_with_docs(query, filtered)
```

**When to use:** Noisy retrieval environments where the vector DB returns irrelevant chunks (common with generic embedding models on specialized domains).

### Adaptive RAG

Dynamically choose the retrieval strategy based on the query type.

```python
def adaptive_rag(query):
    query_type = classify_query_type(query)
    # query_type ∈ {"factoid", "multi-hop", "summarization", "conversational", "unknown"}

    if query_type == "factoid":
        return single_pass_rag(query, top_k=1)  # Fast, cheap
    elif query_type == "multi-hop":
        return iterative_rag(query, max_hops=3)  # Slower, thorough
    elif query_type == "summarization":
        return map_reduce_rag(query, top_k=20)  # Broad retrieval
    elif query_type == "conversational":
        return rag_with_history(query, conversation_history)
    else:
        return hybrid_rag(query)  # Default fallback
```

**When to use:** Systems that serve a variety of query types with different retrieval needs. Saves cost by using cheap retrieval for simple queries and expensive retrieval only when needed.

### Graph RAG

Build a knowledge graph from documents, enabling multi-hop reasoning across entities.

```
Documents:
  "Alice is the CEO of Acme Corp."
  "Acme Corp acquired Beta Inc in 2025."
  "Beta Inc's main product is WidgetX."

Graph:
  Alice ──[CEO_of]──→ Acme Corp ──[acquired]──→ Beta Inc ──[product]──→ WidgetX

Query: "Who owns the company that makes WidgetX?"
  → Traverse: WidgetX ←[product]── Beta Inc ←[acquired]── Acme Corp
  → Answer: "WidgetX is made by Beta Inc, which was acquired by Acme Corp. The CEO of Acme Corp is Alice."
```

**When to use:** Questions that require connecting information across multiple documents (e.g., "How does our supply chain disruption affect product X delivery timeline?").

**Trade-off:** Graph construction is expensive and complex. Only use when queries genuinely require multi-entity, multi-hop reasoning.

### Agentic RAG

The LLM acts as an agent that decides when and what to retrieve, using tools.

```
User: "Compare the Q2 performance of our European and Asian divisions."

Agent decides:
  1. retrieve("European division Q2 performance")
  2. retrieve("Asian division Q2 performance")
  3. Compare results
  4. If financial data is missing: retrieve("European division Q2 financials")
  5. Generate comparison report
```

**When to use:** Complex workflows with conditional retrieval needs. The agent can decide to search multiple times, in different sources, with different queries.

**Risk:** Agent loops (retrieving indefinitely) — set a max retrieval limit (typically 3–5).

---

## 7. Evaluation & Metrics

### Retrieval Precision & Recall

| Metric | What It Measures | Formula |
|---|---|---|
| **Precision@K** | Of the top-K retrieved docs, how many are relevant? | relevant_in_top_K / K |
| **Recall@K** | Of all relevant docs, how many are in top-K? | relevant_in_top_K / total_relevant |
| **MRR (Mean Reciprocal Rank)** | How early is the first relevant doc? | 1 / rank_of_first_relevant |
| **NDCG@K** | Ranking quality (position-weighted) | Normalized discounted cumulative gain |

```python
def retrieval_precision_at_k(retrieved_ids, relevant_ids, k=5):
    retrieved_top_k = retrieved_ids[:k]
    relevant_in_top_k = len(set(retrieved_top_k) & set(relevant_ids))
    return relevant_in_top_k / k

def retrieval_recall_at_k(retrieved_ids, relevant_ids, k=5):
    retrieved_top_k = retrieved_ids[:k]
    relevant_in_top_k = len(set(retrieved_top_k) & set(relevant_ids))
    return relevant_in_top_k / len(relevant_ids) if relevant_ids else 0
```

**Targets:** Precision@5 > 0.8, Recall@5 > 0.7 for production RAG systems.

### Answer Faithfulness

Does the generated answer stay faithful to the retrieved documents?

```python
def check_faithfulness(answer, documents):
    """Use an LLM to check if each claim in the answer is supported by documents."""
    prompt = f"""
    Document: {documents}
    Answer: {answer}

    For each factual claim in the answer, determine if it is directly supported by the document.
    List any unsupported claims.
    """
    result = llm_call(prompt)
    return result  # "All claims supported" or list of violations
```

| Faithfulness Level | Meaning | Acceptable? |
|---|---|---|
| 100% | Every claim is supported by retrieved docs | Target |
| 90-99% | Minor extrapolation but no contradictions | Acceptable |
| 80-89% | Some unsupported claims | Needs improvement |
| < 80% | Significant hallucination or fabrication | Unacceptable |

### Groundedness Checks

Groundedness = is the answer actually based on the provided context, or is the model relying on its own knowledge?

```python
def check_groundedness(answer, context):
    prompt = f"""
    Context: {context}
    Answer: {answer}

    Could this answer be generated using ONLY the context provided, or does it require external knowledge?
    If the answer uses external knowledge not in the context, list those parts.
    """
    return llm_call(prompt)
```

**Best practice:** Run groundedness checks in production. If groundedness drops below threshold, fall back to "I don't have that information" response.

### Hallucination Rate

Percentage of generated claims that are not supported by retrieved documents.

| Evaluation Method | How It Works | Cost | Accuracy |
|---|---|---|---|
| **Human evaluation** | Annotators verify each claim | High | Highest |
| **LLM-as-judge** | GPT-4 evaluates faithfulness | Medium | High |
| **Structured extraction** | Extract claims, verify against docs | Low | Medium |
| **NLI model** | Natural language inference (entailment score) | Low | Medium |

**Target:** < 5% hallucination rate for production systems.

### Human Evaluation

Gold standard for RAG quality, but expensive.

| Criterion | What to Evaluate |
|---|---|
| **Usefulness** | Does the answer help the user? |
| **Accuracy** | Are all claims factually correct? |
| **Completeness** | Does the answer address all aspects of the question? |
| **Conciseness** | Is the answer appropriately detailed? |
| **Attribution quality** | Are sources cited correctly and usefully? |

**Sample size:** 100–500 query-response pairs per evaluation cycle.

### Synthetic Test Set Creation

Create evaluation datasets automatically using the LLM and your document set.

```python
def create_test_set(documents, n=100):
    """Generate synthetic Q&A pairs from documents."""
    test_cases = []
    for doc_chunk in sample_chunks(documents, n):
        prompt = f"""
        Generate a question that can be answered using this document chunk.
        Question should require reading the text, not prior knowledge.
        Provide the question and the expected answer.

        Document: {doc_chunk}
        """
        result = llm_call(prompt)
        test_cases.append({
            "question": result.question,
            "expected_answer": result.answer,
            "source_doc": doc_chunk
        })
    return test_cases
```

**Evaluate:**
```python
def evaluate_rag_pipeline(test_cases):
    for case in test_cases:
        # Run RAG
        response = rag_pipeline(case["question"])

        # Check if answer is faithful to expected answer
        faithfulness = check_faithfulness(response, case["source_doc"])

        # Check if correct answer was retrieved
        retrieval_ok = case["source_doc"] in retrieved_docs

        scores.append({"faithfulness": faithfulness, "retrieval_ok": retrieval_ok})

    return aggregate_metrics(scores)
```

---

## 8. Failure Modes of RAG

### Poor Chunking

| Problem | Symptom | Fix |
|---|---|---|
| **Chunks too small** | Missing context — model can't understand | Increase chunk size or add overlap |
| **Chunks too large** | Multiple topics in one chunk — retrieval is noisy | Decrease chunk size, use semantic splitting |
| **Splits at wrong boundaries** | Sentences cut in half — unreadable chunks | Use sentence-aware splitter |
| **No overlap** | Information lost at boundaries | Add 10-20% overlap |

**Real example:** A legal document chunked at 256 tokens without overlap. "The defendant is not liable for" in chunk 1, "damages caused by force majeure events" in chunk 2. A query about "force majeure" retrieves chunk 2, which says "damages caused by" without the "not liable" context → model thinks the defendant IS liable.

### Irrelevant Retrieval

Retrieved documents don't answer the user's question.

```
Query: "What is the refund policy?"
Retrieved:
  - "Our shipping policy ensures delivery within 5-7 business days." (relevance: 0.83)
  - "Terms and conditions section 4 applies to all purchases." (relevance: 0.79)
  - "Our products come with a one-year warranty." (relevance: 0.77)
  → None answer the question about refunds
```

| Cause | Fix |
|---|---|
| Embedding model doesn't distinguish well in this domain | Fine-tune embedding model on domain data |
| Chunks are too large, containing multiple topics | Improve chunking strategy |
| Query is ambiguous | Use query expansion or clarification |
| Vector DB has noise (irrelevant documents) | Improve document cleaning and filtering |
| Top-K is too high (including low-relevance docs) | Reduce K or add relevance threshold |

**Detection:** If the answer often starts with "Based on the provided documents, I don't have that information" (when you know the information exists), retrieval is failing.

### Context Overload

Too many retrieved documents overwhelm the context window.

```
Top-10 chunks retrieved: 7,000 tokens
System prompt + query + history: 2,000 tokens
Total: 9,000 tokens → exceeds 8K window
→ Model processes only the first 8K tokens (drops last 1K = some chunks + query tail)
→ Answer is based on incomplete context
```

| Fix | How |
|---|---|
| Reduce K | Retrieve fewer, higher-quality chunks |
| Smaller chunks | 256 tokens instead of 512 = 2× more chunks in same budget |
| Dynamic truncation | Prioritize most relevant chunks, drop less relevant ones |
| Larger context model | Use 128K context window model (GPT-4o, Claude, Gemini) |
| Compress chunks | Summarize verbose chunks before injecting |

### Model Ignoring Retrieved Data

The model relies on its parametric knowledge instead of the retrieved documents.

```
Retrieved: "Our refund policy allows returns within 30 days. Exceptions apply for electronics."

Model output: "Our refund policy allows returns within 90 days."
→ Model used its own (outdated) knowledge instead of the retrieved document
```

| Cause | Fix |
|---|---|
| Weak system prompt instruction | Strengthen: "Answer using ONLY the provided documents. Do not use your own knowledge." |
| Model disagrees with retrieved doc | Add: "If the documents contradict your knowledge, follow the documents." |
| Retrieved text is poorly formatted | Improve structure, use clear delimiters |
| Query is outside model's expertise | Lower temperature, emphasize grounding |

**Detection:** Compare generated answer against retrieved documents. If they diverge significantly, the model is ignoring the context.

### Contradictory Sources

Two retrieved documents contain conflicting information.

```
Doc A: "International remote work is permitted."
Doc B: "International remote work is not permitted for roles requiring on-site presence."
```

| Resolution Strategy | Implementation |
|---|---|
| **Recency wins** | Check metadata timestamps — prefer newer document |
| **Specificity wins** | More specific document overrides general one |
| **Surface both** | "Doc A says X, but Doc B says Y. These may apply to different scenarios." |
| **Fallback to uncertainty** | "The documents provide conflicting information on this topic." |

**Best practice:** When sources conflict, make the model aware of the conflict rather than silently picking one. "According to some policies, international remote work is permitted; according to others, it depends on the role."

### Stale Data

Retrieved documents are outdated but still in the index.

```
Document: "The CEO is John Smith." (from 2023)
Reality: "The CEO is Jane Doe." (changed in 2025)

Query: "Who is the CEO?"
Retrieved: "John Smith"
Answer: "The CEO is John Smith." — Wrong!
```

| Fix | How |
|---|---|
| Document expiry | Tag documents with valid-until date; filter expired docs |
| Version tracking | Keep only latest version in the active index |
| Regular index refresh | Re-index on a schedule based on document change frequency |
| Source authority | Tag with source authority score; prefer official vs unofficial sources |
| Staleness detection | Flag documents older than a threshold and deprecate them |

**Best practice:** Treat your vector index like a production database — maintain it, clean it, and remove stale entries. An unmaintained index becomes a liability.
