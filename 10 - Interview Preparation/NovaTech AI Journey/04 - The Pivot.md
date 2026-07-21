# Phase 4: The Pivot
## Months 9–15

---

### Chapter 11: RAG — The Architecture

Raj spent a month researching retrieval-augmented generation before writing a single line of code.

"RAG is simple in theory," he told the team. "User asks a question, you retrieve relevant documents from a knowledge base, you feed them to an LLM as context, it generates an answer grounded in those documents."

"The hard parts?"

"Three: retrieval quality, context management, and evaluation."

He drew the architecture:

```
User Query
    │
    ▼
Query Understanding
    ├─→ Query Rewriting (paraphrase for retrieval)
    ├─→ Intent Classification (billing, tech, account)
    └─→ Entity Extraction (customer ID, product, date)
    │
    ▼
Retrieval
    ├─→ Dense Retrieval (embedding similarity to vector DB)
    ├─→ Keyword Retrieval (BM25 for exact match)
    └─→ Hybrid Fusion + Re-ranking (cross-encoder)
    │
    ▼
Context Assembly
    ├─→ Dynamic chunking (fit relevant docs into context window)
    └─→ Metadata filtering (date, product, department)
    │
    ▼
Generation
    ├─→ System prompt + retrieved chunks + conversation history
    └─→ LLM generates response with citations
    │
    ▼
Guardrails
    ├─→ Factuality check (does response match retrieved docs?)
    ├─→ Safety check (toxicity, PII, off-topic)
    └─→ Confidence scoring
    │
    ▼
Response
```

"Dense retrieval uses embeddings. We fine-tune an embedding model on our ticket data. Queries and documents both get embedded, and we do nearest-neighbor search in a vector database."

"Which vector DB?" Elena asked.

"We'll start with Qdrant. Open source, fast, good filtering. If we need to scale horizontally, we can move to Milvus or Pinecone."

"Chunking strategy?" Anika asked.

Raj pulled up a table. "Support articles are short — most under 500 words. We'll keep them as whole chunks. For longer docs, recursive character split at 256 tokens with 32-token overlap. Metadata tagging: product category, article type, last updated date."

"And retrieval quality?"

"We need three metrics:
1. **Hit rate** — was the right document in the top-k?
2. **MRR** — how high was it ranked?
3. **NDCG** — how well are all relevant documents ranked?"

David nodded. "I'll work with Support to label a test set. 500 queries with known correct answers."

---

**Concepts introduced:** RAG architecture end-to-end, query understanding pipeline, dense vs. keyword retrieval, hybrid search, cross-encoder re-ranking, vector databases (Qdrant, Milvus, Pinecone), chunking strategies, metadata filtering, factuality checks, guardrails for LLM output, RAG evaluation metrics (Hit Rate, MRR, NDCG).

---

### Chapter 12: The Embedding Problem

Three weeks later, Raj had a problem.

"The embedding model we're using — `bge-large-en-v1.5` — is good at general semantic similarity. But it doesn't understand our domain language."

"Example?" David asked.

"'Escalate to tier 2' and 'This needs a senior agent' mean the same thing in support language. The generic embedding puts them far apart because the words are different. Domain-specific terms — 'chargeback,' 'provisioning,' 'SSO handshake' — the model doesn't distinguish between closely related concepts."

"What are the options?"

1. **Fine-tune the embedding model** on our ticket data. Best quality, but takes time and curated data.
2. **Use a domain-specific model** like BAAI/bge-finance for financial terms. Doesn't exist for our domain.
3. **Hybrid search** — combine dense embeddings with keyword (BM25). Handles exact terms better.

"We'll go with hybrid search for now," Raj decided. "Dense for semantic understanding, keyword for exact term matching. That covers most cases. We'll collect data for fine-tuning over the next quarter and revisit."

"And the re-ranking?"

"Cross-encoder. Cross-encoders are slower but more accurate — they compare each document against the query directly rather than through an embedding. We'll use a small model like `BAAI/bge-reranker-v2-m3` for the top 20 retrieved documents."

"What's the latency impact?"

Raj measured. "Dense retrieval: 30ms. Keyword: 10ms. Re-ranking 20 docs: 100ms. Total retrieval: ~150ms. LLM generation: ~500ms. Total: under 700ms. Acceptable."

---

**Concepts introduced:** Domain-specific embedding limitations, embedding model fine-tuning, hybrid search (dense + sparse), BM25 keyword retrieval, cross-encoder re-ranking, latency budget for RAG, the trade-off between retrieval depth and speed.

---

### Chapter 13: The Vector Database

Elena deployed Qdrant on their Kubernetes cluster. Three pods, three replicas, persistent volume claims for index storage.

Day one, it worked beautifully. 10,000 support articles indexed, search queries returning results in 20ms.

Day two, the index broke.

"Ingestion failed halfway through," Elena reported. "We tried to index 50,000 articles at once. The memory usage spiked and the pod got OOM-killed."

"Batch size?"

"Too large. And the default HNSW index parameters are tuned for 1M+ vectors. For our scale, we can relax the precision."

She adjusted:
- Batch size: 500 → 100
- HNSW M parameter: 16 → 8 (less memory, slightly lower recall)
- HNSW ef_construction: 200 → 100 (faster indexing)
- RAM limit: 2GB → 4GB

Second attempt: success.

"Learning a new database every week," Raj remarked.

"This is why I have a job," Elena said.

---

**Concepts introduced:** Vector database deployment on K8s, HNSW index tuning parameters (M, ef_construction), batch ingestion sizing, OOM errors in vector indexing, memory-latency-recall trade-offs in approximate nearest neighbor search, operational learning curve of new infrastructure.

---

### Chapter 14: The Evaluation Trap

The RAG system was ready for evaluation. Raj ran 500 test queries against it.

The numbers looked good: 87% hit rate@10, 92% faithful responses.

David was excited. Anika was cautious.

"Let's actually read through the responses," Anika said.

They sat down. 50 responses. By the 20th, they'd found a pattern.

"The model is hallucinating in subtle ways," Raj said. "It's not making things up from scratch. It's mixing information from multiple retrieved documents. A customer asks about pricing; the model pulls info from a 2024 pricing page and a 2026 version and blends them."

"That's called **contextual hallucination**," Anika said. "The response is grounded in retrieved docs, but it combines facts incorrectly."

"How do we fix it?"

"Three ways:
1. Better prompting — tell the model to use only the most recent document for time-sensitive facts
2. Chunk attribution — each chunk has a date, filter by relevance
3. Self-verification — have the model check its own output against sources"

Raj tried option 1 first. Added to the system prompt: *"If retrieved documents have conflicting dates, use the most recent information. Cite the date of the source document."*

Hallucination rate dropped from 8% to 3%.

"Not perfect," Raj said.

"Production isn't about perfect," Anika said. "It's about measurable improvement. 3% is good enough for Phase 1. We'll track it in monitoring and improve incrementally."

---

**Concepts introduced:** Contextual hallucination (grounded but wrong), temporal conflicts in retrieved data, prompt engineering for factuality, chunk-level metadata for filtering, self-verification patterns, incremental improvement over perfect-first, production monitoring of hallucination rate.

---

### Chapter 15: The First Win

Month 15. The RAG system went live in production for ticket triage.

The pipeline:

1. Customer submits a ticket
2. Intent classifier determines tier (billing, tech, account, general)
3. Embedding model encodes the ticket text
4. Hybrid search finds top-5 relevant support articles
5. Cross-encoder re-ranks for relevance
6. Llama-3-8B generates a suggested response
7. Guardrails check for safety, PII, and factuality
8. If confidence > 0.9: auto-respond
9. If confidence > 0.7: suggest to human agent
10. If confidence < 0.7: route to human with no suggestion

The first week:
- 15,000 tickets processed
- 32% auto-responded (no human involved)
- 45% had suggestions used by agents
- 23% routed to humans with no suggestion
- Agent satisfaction: 4.2/5
- Customer escalation rate: unchanged (not worse! — a win)

"This is real," David said, staring at the dashboard.

"We have a production AI system," Raj said.

"Working," Elena added. "It's actually working."

Marcus walked in. "The VP of Support just called me. She said her team's handling time dropped 40% this week. She asked when we can expand it to more ticket types."

The team exchanged glances. A year ago, they'd been in this same room, staring at failure.

"Tell her next month," Anika said. "Phase 2 starts now."

---

**Concepts introduced:** Confidence-based routing (auto-respond vs. suggest vs. escalate), tiered deployment strategy, production metrics dashboard, agent satisfaction measurement, handling time as a business KPI, incremental expansion strategy, the emotional milestone of first production success.

---

After everyone left, Raj stayed behind.

"A year ago, I had a notebook with 95% accuracy," he said. "Today I have a system with 32% auto-response rate that saves $2M a year."

"Which is more valuable?" Anika asked.

"The one that runs in production."

Anika smiled. "You've learned the most expensive lesson in AI engineering. And it only cost us $2M and one near-disaster."

"Cheap at twice the price," Raj said. "Most companies learn it four times before it sticks."
