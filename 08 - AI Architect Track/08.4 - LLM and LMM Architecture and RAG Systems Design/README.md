# LLM & LMM Architecture and RAG Systems Design

Design production-grade architectures for large language models, multimodal models, and retrieval-augmented generation systems.

## Learning Objectives
- Architect LLM serving infrastructure for low latency and high throughput
- Design RAG pipelines with hybrid search, re-ranking, and evaluation
- Build multimodal pipelines combining text, image, audio, and video
- Implement safety, security, and cost controls for LLM APIs

## Deep Topics

### 1. LLM Serving Patterns
- **Single-GPU inference**: quantization (GPTQ, AWQ, bitsandbytes), FlashAttention, PagedAttention
- **Multi-GPU serving**: tensor parallelism, pipeline parallelism, sequence parallelism
- **Inference engines**: vLLM, TensorRT-LLM, Text Generation Inference (TGI), Ollama
- **Continuous batching**: dynamic batching for variable-length sequences
- **Prefix caching**: KV-cache reuse for system prompts and few-shot examples
- **Speculative decoding**: draft model + target model for faster generation

### 2. Multimodal Model Pipelines
- Vision-Language Models (VLMs): LLaVA, GPT-4V, Gemini, Claude 3 Vision
- Audio-to-text / text-to-audio: Whisper, Parakeet, ElevenLabs, Suno
- Video understanding: frame extraction + VLM aggregation, temporal modeling
- Multimodal vector search: CLIP embeddings for cross-modal retrieval
- Pipeline orchestration: step functions with interleaved modality-specific models

### 3. Retrieval-Augmented Generation (RAG) Architectures
- **Naive RAG**: query → embedding → vector search → prompt + context → LLM → answer
- **Advanced RAG**: query rewriting, HyDE (Hypothetical Document Embeddings), multi-query expansion
- **Agentic RAG**: tool use, query decomposition, iterative retrieval, self-ask
- **Graph RAG**: knowledge graph traversal + semantic retrieval for complex reasoning
- **Routing**: intent-based routing to different retrieval strategies (web search, internal KB, SQL, code)

### 4. Knowledge Base Design
- **Vector databases**: Pinecone, Weaviate, Qdrant, Milvus, pgvector
- **Hybrid search**: dense (embedding) + sparse (BM25 / SPLADE) retrieval with reciprocal rank fusion
- **Graph stores**: Neo4j, Amazon Neptune for entity-relationship knowledge
- **Indexing strategies**: hierarchical navigable small world (HNSW), IVF, product quantization
- **Chunking strategies**: semantic chunking, recursive character splitting, document-aware splitting
- **Metadata filtering**: time-based, source-based, permission-aware retrieval

### 5. Prompt Engineering at Scale
- Prompt version control: prompt templates stored in Git or prompt management systems
- A/B testing prompts: offline eval with curated test sets, online eval with user feedback
- Prompt optimization: DSPy for programmatic prompt optimization, automatic prompt engineering
- Dynamic prompt assembly: context window management, sliding window for long conversations

### 6. Evaluation Frameworks for RAG Systems
- **Retrieval metrics**: recall@k, MRR, NDCG, precision@k
- **Generation metrics**: faithfulness/hallucination, answer relevance, context adherence
- **End-to-end evaluation**: RAGAS framework, TruLens, DeepEval, LangSmith
- **Ground truth datasets**: creating and maintaining eval sets for RAG
- **LLM-as-judge**: using GPT-4 or Claude as an automated evaluator

### 7. Autoscaling & Latency Budgeting
- Latency budgets: end-to-end SLA decomposition (routing, retrieval, inference, post-processing)
- Autoscaling policies: request-based, token-based, GPU utilization-based
- Response streaming: server-sent events (SSE) vs WebSocket for progressive token delivery
- Caching strategies: semantic caching for similar queries, response deduplication

### 8. Security Considerations for LLM APIs
- **Prompt injection**: input sanitization, system prompt hardening, instruction guardrails
- **Data leakage**: output filtering, PII redaction, differential privacy for training data
- **OWASP Top 10 for LLMs**: prompt injection, data leakage, supply chain, insecure output handling
- **Guardrails**: NVIDIA NeMo Guardrails, Guardrails AI, Microsoft Azure AI Content Safety
- **Rate limiting & abuse detection**: per-user rate limits, anomaly detection in query patterns

## Deliverables
- Design a RAG architecture for a customer support system with 10k+ queries/day under 2s p95 latency
- Compare serving strategies for an open-source LLM across vLLM, TGI, and TensorRT-LLM
- Build a RAG evaluation pipeline with faithfulness, relevance, and recall metrics
