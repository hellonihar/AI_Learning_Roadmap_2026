# Capstone: LLM-Powered Enterprise Knowledge System

Design a production-grade enterprise knowledge system that combines LLMs, RAG, multi-modal search, and agentic workflows to serve thousands of internal users.

## Context

A large professional services firm (legal, consulting, or financial)
wants to build an internal AI knowledge assistant. It must answer
questions across millions of documents (contracts, reports, emails,
presentations), support multi-modal queries (text, tables, images),
and execute multi-step research workflows via agents.

- **Scale**: 10k+ internal users, 10M+ documents (PDFs, emails, slides, spreadsheets)
- **Query load**: 5k–50k queries/day with p95 latency < 3s
- **Requirements**: grounded answers with citations, multi-turn conversation, multi-modal (charts, tables, diagrams in documents), agentic workflows for research tasks
- **Constraints**: strict data residency (EU), no training on customer data, audit trail for all model interactions, $50k–$200k/month infrastructure budget
- **Current state**: basic keyword search (Elasticsearch), no LLM capabilities

## Deliverables

### 1. Retrieval Architecture Design
- **Ingestion pipeline**: document parsing (PDF, DOCX, PPTX, images), chunking strategy (semantic, recursive, document-aware), metadata extraction, embedding generation
- **Index design**: vector DB selection (Weaviate/Pinecone/Qdrant/Milvus), hybrid search (dense + sparse), multi-tenancy, hierarchical indexes for different document types
- **Chunking strategy comparison**: tradeoffs of fixed-size vs. semantic vs. agentic chunking; chunk overlap, metadata injection
- **Multi-modal indexing**: CLIP embeddings for images/tables, table extraction and indexing (TableTransformer, Camelot), slide understanding

### 2. RAG Pipeline Architecture
- **Query processing**: query rewriting, HyDE, multi-query expansion, query decomposition
- **Retrieval**: hybrid search fusion (RRF), re-ranking (Cohere, BGE, cross-encoders), contextual retrieval
- **Generation**: prompt assembly, context window management, citation formatting
- **Fallback strategies**: no-results handling, low-confidence detection, escalation to human
- **Evaluation framework**: RAGAS (faithfulness, relevance, context recall), LLM-as-judge, human eval

### 3. Agentic Workflow Design
- **Research agent**: query decomposition → parallel sub-queries → retrieval → synthesis → report generation
- **Comparison agent**: cross-document analysis, diff identification, summarization
- **Workflow orchestration**: LangGraph / AutoGen / CrewAI state machines
- **Tool integration**: document search, web search, code execution, database queries, calendar/email
- **Human-in-the-loop**: approval gates for sensitive actions, escalation paths

### 4. Multi-Modal Pipeline
- **Image understanding**: table-to-text, chart interpretation, diagram analysis
- **Audio/video**: meeting transcription (Whisper) → summarization → search
- **Cross-modal retrieval**: query in one modality → retrieve in another
- **Streaming multi-modal**: handling user uploads during conversation

### 5. LLM Selection & Serving Strategy
- **Model selection**: open-source vs. proprietary, smaller specialist models vs. large general models
- **Serving architecture**: vLLM cluster, quantized models, speculative decoding for latency
- **Caching**: semantic caching (GPTCache), exact-match caching, prefix caching
- **Cost optimization**: model routing (simple queries → small model, complex → large model)

### 6. Security & Guardrails
- **PII detection & redaction**: in queries, retrieved documents, and generated responses
- **Prompt injection prevention**: input sanitization, system prompt hardening, instruction detection
- **Access control**: document-level permissions, need-to-know retrieval filtering
- **Audit trail**: full query/response logging, retrieval provenance, user feedback

### 7. Evaluation & Monitoring
- **Offline evaluation**: curated test sets with golden answers, retrieval recall@k
- **Online monitoring**: user satisfaction (thumbs up/down), response quality drift detection
- **A/B testing**: comparing retrieval strategies, prompt templates, model versions
- **Feedback loop**: user corrections → fine-tuning data → periodic model updates

### 8. Cost & Capacity Plan
- **Compute cost**: embedding generation, retrieval, inference (per-query breakdown)
- **Storage cost**: vector indexes, document cache, audit logs
- **Autoscaling**: request-based, token-based, user-activity-based scaling policies
- **Budget allocation**: infrastructure vs. API costs vs. evaluation and monitoring

## Evaluation Criteria

| Criteria | Weight | Description |
|----------|--------|-------------|
| RAG architecture depth | 25% | Retrieval, generation, evaluation design choices |
| Agentic workflow design | 20% | Orchestration, tool use, HITL patterns |
| Practical feasibility | 20% | Realistic cost, latency, and accuracy targets |
| Security & compliance | 15% | Data residency, PII, access control |
| Evaluation strategy | 10% | Offline and online metrics, feedback loops |
| Presentation | 10% | Clarity of diagrams and narrative |

## Format
- **Written**: PDF or Markdown (30–50 pages)
- **Live review**: 60-minute presentation + 30-minute Q&A
- **Tools**: any diagramming tool; demo of a working prototype (optional, bonus)
