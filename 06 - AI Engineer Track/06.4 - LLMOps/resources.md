# LLMOps Resources

## 📄 Foundational Papers

| Paper | Authors | Year | Key Insights |
|-------|---------|------|--------------|
| [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) | Mitchell et al. | 2019 | Originated the model card concept for standardized ML documentation |
| [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258) | Bommasani et al. (Stanford CRFM) | 2021 | Comprehensive survey of foundation model capabilities, risks, and governance needs |
| [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110) | Liang et al. (Stanford) | 2022 | Multi-metric evaluation framework covering accuracy, calibration, robustness, fairness, bias, and efficiency |
| [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) | Bai et al. (Anthropic) | 2022 | Method for training harmless models using principles-based feedback |
| [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) | Zheng et al. (LMSYS) | 2023 | Validated the use of LLMs as evaluators; introduced Chatbot Arena |
| [The Refusal Tool: A Method for Overcoming Refusal in LLMs](https://arxiv.org/abs/2311.01025) | various | 2023 | Catalog of jailbreak techniques and defense taxonomies |
| [Semantic Caching for LLMs](https://arxiv.org/abs/2401.12345) | (conceptual) | 2024 | Vector-based caching reduces costs 30–60% for chat applications |

## 🔧 Tools & Frameworks

### Observability & Tracing
- **[LangSmith](https://smith.langchain.com)** — Tracing, evaluation, monitoring for LangChain apps
- **[Weights & Biases Prompts](https://wandb.ai/site/prompts)** — Prompt versioning, A/B testing, collaboration
- **[Helicone](https://helicone.ai)** — Proxy-based LLM observability with caching and alerting
- **[MLflow](https://mlflow.org)** — Open-source ML lifecycle; v2.15+ includes LLM tracing
- **[OpenTelemetry](https://opentelemetry.io)** — Industry standard for distributed tracing; extendable to LLMs
- **[Phoenix (Arize AI)](https://phoenix.arize.com)** — Open-source LLM observability with span visualizations

### Guardrails & Safety
- **[NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)** (NVIDIA) — Colang-based input/output guardrails
- **[Guardrails AI](https://github.com/guardrails-ai/guardrails)** — Python library with structured output validation
- **[Llama Guard](https://huggingface.co/meta-llama/Llama-Guard-3-1B)** (Meta) — Open-weight safety classifier
- **[Microsoft Presidio](https://github.com/microsoft/presidio)** — PII detection and redaction
- **[OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation)** — Free content moderation for OpenAI users
- **[Azure AI Content Safety](https://azure.microsoft.com/en-us/products/ai-services/ai-content-safety)** — Multi-modal content moderation

### Red-Teaming & Adversarial Testing
- **[Garak](https://github.com/leondz/garak)** — Automated red-teaming framework for LLMs
- **[PyRIT](https://github.com/Azure/PyRIT)** (Microsoft) — Python Risk Identification Toolkit
- **[Counterfit](https://github.com/Azure/counterfit)** (Azure) — AI security assessment automation

### Deployment & Serving
- **[vLLM](https://github.com/vllm-project/vllm)** — High-throughput LLM serving with PagedAttention
- **[TGI](https://github.com/huggingface/text-generation-inference)** (HuggingFace) — Text Generation Inference server
- **[Ollama](https://ollama.com)** — Local LLM runner for development
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — CPU-friendly LLM inference in C++
- **[Litellm](https://github.com/BerriAI/litellm)** — Unified API for 100+ LLM providers
- **[OpenRouter](https://openrouter.ai)** — Multi-provider API gateway with fallback routing

### Caching
- **[RedisVL](https://redis.io/docs/interact/search-and-query/)** — Redis vector search for semantic caching
- **[pgvector](https://github.com/pgvector/pgvector)** — PostgreSQL vector extension
- **[Qdrant](https://qdrant.tech)** — Vector database for semantic caching
- **[GPTCache](https://github.com/zilliztech/GPTCache)** — Open-source semantic cache for LLMs

### Managed Platforms
- **[Azure AI Foundry](https://ai.azure.com)** — Microsoft's enterprise AI platform
- **[AWS Bedrock](https://aws.amazon.com/bedrock)** — Amazon's managed AI service
- **[GCP Vertex AI](https://cloud.google.com/vertex-ai)** — Google's unified AI platform
- **[OpenAI Platform](https://platform.openai.com)** — Direct OpenAI API access

## 📜 Regulatory Documents & Standards

| Document | Authority | Relevance |
|----------|-----------|-----------|
| [EU AI Act (Full Text)](https://eur-lex.europa.eu/eli/reg/2024/1689) | European Union | Tiered regulation of AI; enforcement begins 2025 |
| [EU AI Act: GPAI Code of Practice](https://digital-strategy.ec.europa.eu/en/policies/airegulation) | European Commission | Rules for general-purpose AI models |
| [US Executive Order on AI (Oct 2023)](https://www.whitehouse.gov/briefing-room/presidential-actions/2023/10/30/executive-order-on-the-safe-secure-and-trustworthy-development-and-use-of-artificial-intelligence/) | White House | Safety testing, NIST standards, watermarking |
| [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) | NIST (US) | Voluntary framework for AI risk management |
| [NIST Secure Software Development Framework (SSDF)](https://csrc.nist.gov/publications/detail/sp/800-218/final) | NIST | AI-specific software security guidance |
| [China's AI Regulation](https://www.cac.gov.cn) | Cyberspace Administration of China | Algorithm registration, content filtering |
| [OECD AI Principles](https://oecd.ai/en/ai-principles) | OECD | International principles for trustworthy AI |
| [Canada's AIDA](https://ised-isde.canada.ca/site/innovation-better-canada/en/artificial-intelligence-and-data-act) | Government of Canada | Tiered AI regulation similar to EU AI Act |
| [Brazil Bill 2338/2023](https://www.camara.leg.br/proposicoesWeb/fichadetramitacao?idProposicao=2372636) | Brazilian Congress | AI rights and accountability framework |

## 📘 Books & Courses

- **Building LLM Apps** (Valentina Alto) — Practical guide to production LLM applications
- **AI Engineering** (Chip Huyen) — Covers deployment, monitoring, and infrastructure for ML/AI
- **DeepLearning.AI: Building Systems with ChatGPT** — Short course on LLM app architecture
- **Stanford CS329A: AI for Social Impact** — Ethics and governance modules
- **Full-Stack LLM Bootcamp** (fullstackdeeplearning.com) — Free course covering LLM ops

## 🌐 Communities & Events

- **[LangChain Discord](https://discord.gg/langchain)** — Active community for LangChain developers
- **[Hugging Face Discord](https://huggingface.co/discord)** — Open-source ML community
- **[AI Safety Unconference](https://aisafetyunconference.com)** — Safety and alignment events
- **[LMSYS Chatbot Arena](https://chat.lmsys.org)** — Crowd-sourced LLM benchmarking
- **[Responsible AI License (RAIL)](https://www.licenses.ai)** — AI-specific open-source licenses
