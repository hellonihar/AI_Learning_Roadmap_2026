# LLMs & GenAI Architecture — Q&A

## Q1: Design a RAG system for a customer support chatbot from scratch.

**1. Ingestion pipeline:**
- Document chunking: recursive character split (500-1000 tokens with overlap)
- Embedding model: e.g., `text-embedding-3-large` or `BAAI/bge-large-en-v1.5`
- Vector DB: Milvus/Qdrant/Pinecone with HNSW index
- Optional: hybrid search (BM25 + dense) + re-ranker (Cohere rerank or BGE-reranker)

**2. Retrieval pipeline:**
- Query rewriting — rewrite user question for better retrieval
- Multi-query retrieval — generate 3-5 query variations, retrieve for all
- Retrieve top-k (k=5-10), re-rank, keep top-3
- Add metadata filtering (product category, date range)

**3. Generation pipeline:**
- Build prompt: system instruction + retrieved chunks + conversation history + user query
- Generate response with citation to source chunks
- Post-processing: check for hallucination (self-verify), format response

**4. Evaluation:**
- Retrieval: hit rate, MRR, NDCG
- Generation: faithfulness, relevance, helpfulness
- End-to-end: user satisfaction, escalation rate, resolution time

**Trade-offs:** More chunks → better recall but more tokens → higher cost/latency. Re-ranking adds 100ms but significantly improves quality.

---

## Q2: When would you fine-tune vs. use RAG? Walk through the trade-offs.

| Factor | RAG | Fine-Tuning |
|--------|-----|------------|
| Knowledge updates | Instant (just update the index) | Hours-days of training |
| New data | No retraining needed | Full retraining cycle |
| Reducing hallucination | Good (grounds in retrieved docs) | Indirect (memorization) |
| Custom behavior/tone | Limited (prompt engineering) | Strong (learns from examples) |
| Cost (inference) | Higher tokens (context + retrieved docs) | Lower tokens (no retrieval docs) |
| Latency | Higher (retrieval + generation) | Lower (generation only) |
| Data privacy | Docs stay in your DB | Need training data |

**Decision framework:**
- Need to update knowledge frequently → **RAG**
- Need specific output format/behavior → **Fine-tune**
- Both → **RAG + fine-tune** (fine-tune for behavior, RAG for knowledge)

---

## Q3: Describe the full RLHF pipeline. What are its alternatives?

**RLHF (Reinforcement Learning from Human Feedback):**

1. **SFT** — supervised fine-tuning on high-quality demonstrations
2. **Reward modeling** — train a reward model on human preference data (compare two responses, pick better one)
3. **RL optimization** — use PPO to maximize reward while staying close to the SFT model (KL penalty)

**Alternatives:**
- **DPO** (Direct Preference Optimization) — directly optimizes policy from preferences without a separate reward model. Simpler, more stable, now preferred by most labs.
- **KTO** — uses binary feedback (good/bad) instead of pair comparisons
- **Rejection sampling** — generate many responses, keep the best (works well for verifiable tasks)

**Why DPO won:** No reward model → less compute, no PPO instability, easier to scale preference data.

---

## Q4: How would you reduce latency in an LLM serving pipeline?

1. **Model optimization:**
   - Quantization: FP16 → INT8/FP8/INT4 (2-4x speed)
   - Speculative decoding: use a draft model (e.g., 1/10th size) for most tokens, verify with target model
   - KV cache optimization: GQA, KV cache quantization, prefix caching

2. **Infrastructure:**
   - Continuous batching (vLLM, TensorRT-LLM) — dynamically add/remove requests from a batch
   - Tensor parallelism — shard across GPUs; pipeline parallelism for batch throughput
   - Faster attention kernels: Flash Attention 2/3, PagedAttention

3. **System design:**
   - Semantic caching — cache responses for similar queries
   - Streaming — return tokens as generated (TTFT matters)
   - Request routing — route to nearest/smallest model that can handle the task
   - Prefill optimization — use chunked prefill for long contexts

4. **Targets:** TTFT < 200ms, TPOT < 50ms/token for interactive use.

---

## Q5: How do agents handle tool call failures? Design a robust agent architecture.

**Failure modes:** tool timeout, invalid JSON, API error, hallucinated tool name, wrong parameters.

**Robust agent loop:**

1. **Structured output** — force JSON schema using constrained decoding (outlines, guidance)
2. **Validation layer** — verify tool name exists, types match, required params present
3. **Retry with feedback** — on failure, feed the error message back to the LLM: "Call to tool X failed with error Y. Try again."
4. **Max retries** — configurable limit (usually 2-3), then escalate to fallback
5. **Fallback** — tell user "I couldn't complete that action" or route to human agent
6. **Observation summarization** — truncate long tool outputs to fit context window
7. **Timeout handling** — if tool takes > N seconds, return timeout error

**Key principle:** The agent loop is a state machine. Each step (tool call, observation, reasoning) is a state transition with error handling at every edge.

---

## Q6: Compare LoRA, QLoRA, and full fine-tuning. When would you use each?

| Method | Trainable Params | Memory | Speed | Quality | Use Case |
|--------|-----------------|--------|-------|---------|----------|
| Full FT | 100% | Very high | Slow | Best (if enough data) | Large data, domain shift |
| LoRA | 0.1-1% | Low | Fast | ~90-95% of full FT | Task adaptation, limited data |
| QLoRA | 0.1-1% | Very low | Fast | ~85-95% of full FT | Consumer GPUs, rapid iteration |

- **LoRA** — low-rank adapters inserted into attention layers. The base model stays frozen. Rank r=8-64 is typical.
- **QLoRA** — 4-bit NF4 quantization of base model + LoRA adapters. Fits 65B model on single 48GB GPU.

**Rule of thumb:** Start with LoRA. If quality isn't sufficient and you have data, try full FT. QLoRA when GPU-limited.

---

## Q7: What is speculative decoding and how does it improve inference speed?

**Idea:** Use a small, fast "draft" model to generate K candidate tokens, then verify them in parallel with the target model in a single forward pass. Accepted tokens are free — you paid for K verifications in one batch.

- **Speedup:** 2-3x for typical setups
- **No quality loss:** mathematically identical to target model distribution
- **Best when:** draft model is much smaller (1/10 params) and has high agreement with target

**Variants:** Medusa (multiple draft heads from the target model itself), self-speculative decoding (skip some layers as draft).

---

## Q8: Design a chunking strategy for RAG over documents of varying lengths.

1. **Small docs (<500 tokens):** Keep as single chunks (no splitting needed)
2. **Medium docs (500-2000 tokens):** Recursive character split with 200-chunk/20-overlap
3. **Large docs (2000+ tokens):** Semantic chunking — split at topic boundaries using embedding similarity or LLM-based section detection
4. **Code/structured:** Split by function/module boundaries
5. **PDFs:** Use hierarchical chunking — extract structure (headings), chunk per section, embed heading into chunk metadata

**Key decisions:**
- Chunk size trade-off: small chunks → better retrieval precision, worse context
- Overlap prevents information loss at boundaries (typically 10-20%)
- Metadata (title, heading, page number) improves retrieval
- Consider multi-representation: embed chunk summary but store full chunk text

---

## Q9: How do you evaluate a RAG system? What metrics matter?

| Dimension | Metric | What It Measures |
|-----------|--------|-----------------|
| Retrieval | Hit Rate @ k, MRR, NDCG | Is the right document retrieved? |
| Retrieval | Precision @ k | Are retrieved documents relevant? |
| Generation | Faithfulness | Does the answer contradict retrieved docs? |
| Generation | Answer Relevance | Does the answer address the question? |
| Generation | Context Precision | Does the answer use the right parts of context? |
| End-to-end | User satisfaction, Escalation rate | Business impact |

**Framework:** Use an LLM-as-judge (e.g., GPT-4, Llama-3.1-70B) to rate generated answers on faithfulness and relevance. Supplement with human evaluation for a subset.

**Tools:** RAGAS, TruLens, LangSmith, Arize AI.

---

## Q10: Explain the ReAct agent pattern with a concrete example.

**ReAct = Reasoning + Acting.** The LLM alternates between:
- **Thought** — reasoning about what to do next
- **Action** — calling a tool/API
- **Observation** — tool output
- → repeat until final answer

**Concrete example (weather bot):**
```
User: "Should I bring an umbrella in Tokyo tomorrow?"

Thought: I need to check the weather forecast for Tokyo tomorrow.
Action: get_weather(city="Tokyo", date="2026-07-22")
Observation: {"temp_high": 28, "rain_probability": 0.85}

Thought: 85% rain probability means it's very likely to rain.
Action: get_recommendation(rain_probability=0.85)
Observation: "High probability of rain"

Thought: I have enough information to answer.
Answer: Yes, there's an 85% chance of rain in Tokyo tomorrow. Bring an umbrella!
```

**Key:** The "Thought" step is critical — it externalizes the model's reasoning, making it interpretable and debuggable.

---

## Q11: What is prompt caching and when should you use it?

**Prompt caching** stores the KV cache from the system prompt, few-shot examples, or conversation history to avoid recomputing attention over repeated prefix tokens.

**When to use:**
- Long system prompts (common in agents)
- Few-shot examples that don't change between requests
- Multi-turn conversations (cache previous turns)
- RAG applications (cache common instructions)

**Benefits:** Reduces TTFT by 30-80% for cache hits. Cuts cost proportionally.

**Implementation:** Supported by Anthropic, OpenAI, vLLM (prefix caching). Cache is typically LRU with configurable TTL.

**Trade-off:** Cache consumes GPU memory. Longer cache lives → more memory pressure.

---

## Q12: Compare DPO vs PPO vs RLHF for alignment.

| Method | Reward Model? | Complexity | Stability | Data | Quality |
|--------|--------------|-----------|-----------|------|---------|
| SFT | No | Low | High | Demonstrations | Good behavior, limited |
| RLHF (PPO) | Yes | Very high | Low | Preference pairs | Best (if tuned well) |
| DPO | No | Low | High | Preference pairs | ~RLHF quality |
| KTO | No | Low | High | Binary feedback | Good, less data needed |

**Current state:** DPO has become the default alignment method in 2025-2026. It's simpler, more stable, and achieves comparable results to RLHF. PPO is still used in frontier labs with large preference datasets and specialized infra teams.

---

## Q13: How would you handle hallucination in a production LLM system?

**Multi-layered approach:**

1. **Retrieval** — ground the model with retrieved documents (RAG). Don't rely on parametric knowledge for facts.
2. **Prompt engineering** — instruct the model to say "I don't know" rather than guessing. Constrain the output format.
3. **Temperature** — lower temperature (0.0-0.3) for factual tasks reduces creativity/hallucination.
4. **Self-verification** — ask the model to verify its own answer against retrieved docs before responding.
5. **Contrastive decoding** — compare output with a smaller/weaker model, penalize tokens that the weak model prefers.
6. **Output guardrails** — check for factual consistency with sources (NeMo Guardrails, Guardrails AI).
7. **Human review loop** — for high-stakes domains (healthcare, legal), require human approval.

**No single solution works.** The best defense is layered: RAG + low temperature + verification + guardrails.

---

## Q14: Design a multi-agent orchestration system.

**Architecture:**

1. **Orchestrator agent** — receives user request, routes to specialized agents, aggregates results
2. **Specialist agents** — each has specific tools/knowledge (search, code, data analysis, creative writing)
3. **Shared memory** — the orchestrator maintains conversation context visible to all agents
4. **Handoff protocol** — standardized format for passing tasks between agents

**Communication patterns:**
- **Supervisor** — orchestrator decides which agent to call and when
- **Debate** — multiple agents solve the same problem, vote on best answer
- **Pipeline** — output of one agent feeds into another (e.g., search → summarize)
- **Hierarchical** — manager agent delegates to sub-agents

**Key challenges:** Shared context management, deadlock detection, agent hallucination cascading, cost management.

---

## Q15: What is MCP (Model Context Protocol) and why does it matter?

**MCP** = open protocol developed by Anthropic to standardize how LLMs connect to external tools and data sources. Think of it as "USB-C for AI" — a universal plug-and-play interface.

**Key components:**
- **MCP Server** — exposes tools, resources, and prompts via a standard interface
- **MCP Client** — connects the LLM to MCP servers (hosted by the application)
- **Discovery** — clients can query what tools/resources a server provides

**Why it matters:**
- Standardization replaces ad-hoc function calling implementations
- Plug-and-play: write one MCP server, use with any MCP-compatible client
- Security: sandboxed execution, permission scoping
- Ecosystem: growing library of MCP servers (databases, APIs, file systems, search)

---

## External Resources

- [Let's Data Science: 50 LLM & AI Engineer Questions (2026)](https://letsdatascience.com/blog/50-llm-and-ai-engineer-interview-questions-for-2026)
- [DataCamp: 36 LLM Interview Questions (2026)](https://www.datacamp.com/blog/llm-interview-questions)
- [Interview Guys: 10 LLM Engineer Questions (2026)](https://blog.theinterviewguys.com/llm-engineer-interview-questions-and-answers/)
- [CoPrep: Top AI Engineer Questions (2026)](https://www.coprep.ai/blog/top-ai-engineer-interview-questions-in-2026-llms-rag-agents-and-langchain)
- [InterviewBit: LLM Interview Questions](https://www.interviewbit.com/llm-interview-questions-answers/)
- [DataCamp: 30 RAG Interview Questions (2026)](https://www.datacamp.com/blog/rag-interview-questions)
- [DataCamp: 30 Agentic AI Interview Questions (2026)](https://www.datacamp.com/blog/agentic-ai-interview-questions)
- [GitHub: Alexey-Popov/awesome-ai-architect — interview-prep.md](https://github.com/Alexey-Popov/awesome-ai-architect/blob/main/interview-prep.md)
