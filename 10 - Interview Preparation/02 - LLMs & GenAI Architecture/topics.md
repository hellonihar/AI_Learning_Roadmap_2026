# LLMs & GenAI Architecture

## LLM Lifecycle
- Pre-training → Supervised Fine-Tuning (SFT) → RLHF / DPO
- Base model vs. instruct/chat model vs. aligned model
- Continual pre-training, domain adaptation

## Prompt Engineering
- Zero-shot, few-shot, chain-of-thought (CoT)
- System prompts, instruction tuning
- Prompt caching — when and why
- Structured outputs (JSON mode, grammar-constrained decoding)

## Retrieval-Augmented Generation (RAG)
- Chunking strategies, embedding models
- Vector databases (Pinecone, Weaviate, Milvus, Qdrant)
- Hybrid search (dense + sparse), re-ranking
- RAG evaluation: hit rate, MRR, NDCG, faithfulness
- Advanced: Agentic RAG, Self-RAG, Corrective RAG

## Fine-Tuning
- Full fine-tuning vs. LoRA / QLoRA / DoRA
- Adapter-based methods, soft prompts, prefix tuning
- Data preparation, quality filtering, decontamination
- When to fine-tune vs. prompt engineer vs. RAG

## Agents & Tools
- ReAct, Plan-and-Execute, LLM-compiler patterns
- Function calling / tool use
- Multi-agent orchestration
- MCP (Model Context Protocol), A2A (Agent-to-Agent)

## Reasoning & Inference
- Chain-of-Thought reasoning
- Speculative decoding, Medusa, parallel decoding
- Quantization: GPTQ, AWQ, GGUF — trade-offs
- vLLM, TensorRT-LLM, TGI for inference serving

## Common Interview Questions
- Design a RAG system for a customer support chatbot.
- When would you fine-tune vs. use RAG? What are the trade-offs?
- How would you reduce latency in an LLM serving pipeline?
- Describe the full RLHF pipeline and alternatives like DPO.
- How do agents handle tool call failures gracefully?
