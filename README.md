# AI Learning Roadmap 2026

I built this roadmap because I got tired of bouncing between tutorials. One week it's a YouTube video on transformers, the next it's a blog post about LoRA fine-tuning, and somewhere in between you're supposed to magically know how to ship a model to production. Nobody connects the dots for you. This repo is the connected dots.

**The journey starts with foundations that actually matter.** Before you touch a neural network, you need math that goes deeper than "derivative is slope" — we cover linear algebra through SVD, calculus through the chain rule in matrix form, probability through MLE and Bayes, optimization through Adam's update rule. You learn Python the way ML engineers use it (not the way CS 101 teaches it), SQL because real data lives in databases, DSA because thinking in algorithmic complexity saves your infrastructure bill, and cloud because training on a laptop is a rookie mistake. There's even a section on using AI coding tools effectively — because in 2026, not using them is like refusing to use autocomplete.

**Then you actually learn machine learning.** Not just calling `model.fit()`. You start with fundamentals — what is a model, how do loss functions work, what is bias-variance, why your validation split matters. Then 13 algorithm families from linear regression through gradient boosting and SVMs through clustering and anomaly detection. Feature engineering, hyperparameter tuning, handling imbalanced data, model interpretability — the stuff that separates a Kaggle winner from someone who watched a course. And then 37 end-to-end projects: recommendation systems, fraud detection, sales forecasting, resume screening, disease risk prediction. Each one has a problem statement, a dataset, an implementation, results, and deployment notes. You don't just learn — you do.

**From ML to deep learning, you build every intuition from scratch.** You implement backpropagation in NumPy before you touch PyTorch. You understand why vanishing gradients happen and why ReLU helps. You walk through CNNs from the convolution operation through ResNet's skip connections. You learn RNNs, LSTMs, GRUs — and then the transformer architecture that made them obsolete for most tasks. Attention mechanisms, positional encoding, BERT vs GPT, autoencoders, VAEs, GANs, diffusion models. Every topic has a lesson, a code walkthrough, exercises with answers, and a cheatsheet. Then you graduate to practical deep learning with PyTorch — building classifiers, segmentation models, object detectors, time series forecasters, and text generation systems. By the end, you're not just familiar with these concepts — you've written them.

**Now you learn what nobody teaches: how to ship.** ML System Design takes you through the full lifecycle: problem formulation, data system design, feature engineering strategy, modeling strategy, evaluation, serving, monitoring, and governance. One document maps every phase to real-world gates and team roles. Then MLOps: versioning data with DVC, building pipelines with Prefect, containerizing with Docker, tracking experiments with MLflow, serving models with FastAPI, CI/CD for ML, and automated retraining when drift is detected. For deep learning specifically: distributed training with FSDP and DeepSpeed, checkpointing strategies for multi-GPU training, large model storage with quantization, and optimized inference with TensorRT, ONNX Runtime, and vLLM. This is the section that saves your career when your model works in the notebook but falls apart in production. The AI Engineer track then covers LLM literacy, prompt and context engineering, RAG, vector databases, memory systems, fine-tuning (LoRA/QLoRA), agentic AI (ReAct, MCP, LangGraph), and LLMOps with guardrails, observability, and governance.

**The research track is where the frontier starts.** Reinforcement Learning fundamentals: MDPs, multi-armed bandits, dynamic programming, Monte Carlo methods, TD learning, and policy gradients all the way through PPO. LLM Foundations: the evolution from GPT-1 to GPT-4, scaling laws (Kaplan and Chinchilla), and alignment techniques (RLHF, DPO, Constitutional AI). LLM Engineering: efficient attention (FlashAttention, multi-query/grouped-query attention), KV-cache optimization and PagedAttention, quantization (GPTQ, AWQ, GGUF), and speculative decoding. Deployment at scale: model serving with continuous batching, inference optimization through kernel fusion and tensor parallelism, and cost optimization across GPU types from RTX 4090s to H100s. This section takes you from practitioner to someone who can read a paper and implement it.

668 files, one path, all the way through. Whether you want to be an ML engineer, an AI application builder, or a researcher — this roadmap takes you from your first line of Python to PagedAttention on an H100 cluster. And every step of the way, there's a lesson, a code walkthrough, exercises to test yourself, and a cheatsheet so you don't forget what you learned.

---

### [01 - Prerequisites](./01%20-%20Prerequisites)
Core foundations: Python, SQL, Mathematics, Software Essentials, Cloud Fundamentals, AI Coding, and DSA for AI.

### [02 - Machine Learning](./02%20-%20Machine%20Learning)
Fundamentals, feature engineering, ML algorithms, advanced concepts, and end-to-end projects.

### [03 - Deep learning](./03%20-%20Deep%20learning)
Neural network foundations, optimization, CNN / transfer learning, sequence models, Seq2Seq, and unsupervised DL.

### [04 - Practical Deep Learning](./04%20-%20Practical%20Deep%20Learning)
Hands-on with PyTorch, NLP, computer vision, and time series using deep learning.

### [05 - Machine Learning Track](./05%20-%20Machine%20Learning%20Track)
- **[05.1 - ML System Design](./05%20-%20Machine%20Learning%20Track/05.1%20-%20ML%20System%20Design)** &mdash; Problem formulation through feedback and monitoring
- **[05.2 - ML Ops](./05%20-%20Machine%20Learning%20Track/05.2%20-%20ML%20Ops)** &mdash; Data versioning, pipelines, Docker, monitoring, experiment tracking, CI/CD
- **[05.3 - MLOps for DL](./05%20-%20Machine%20Learning%20Track/05.3%20-%20MLOps%20for%20DL)** &mdash; Distributed training, checkpointing, large model storage, optimized inference

### [06 - AI Engineer Track](./06%20-%20AI%20Engineer%20Track)
- **[06.1 - LLM Literacy](./06%20-%20AI%20Engineer%20Track/06.1%20-%20LLM%20Literacy)** &mdash; LLM 101, prompt engineering, context engineering
- **[06.2 - AI Application Development](./06%20-%20AI%20Engineer%20Track/06.2%20-%20AI%20Application%20Development)** &mdash; Orchestration, vector DBs, RAG, memory, fine-tuning, automation
- **[06.3 - Agentic AI](./06%20-%20AI%20Engineer%20Track/06.3%20-%20Agentic%20AI)** &mdash; Planning, state/memory, tool use, MCP protocols, frameworks
- **[06.4 - LLMOps](./06%20-%20AI%20Engineer%20Track/06.4%20-%20LLMOps)** &mdash; UX, deployment, observability, guardrails, governance, managed platforms

### [07 - Research Track](./07%20-%20Research%20Track)
- **[07.1 - RL Fundamentals](./07%20-%20Research%20Track/07.1%20-%20RL%20Fundamentals)** &mdash; MDP, policy gradients, PPO
- **[07.2 - LLM Foundations](./07%20-%20Research%20Track/07.2%20-%20LLM%20Foundations)** &mdash; Transformers, language modeling, inference, decoding
- **[07.3 - LLM Engineering](./07%20-%20Research%20Track/07.3%20-%20LLM%20Engineering)** &mdash; Data, tokenization, pre-training, fine-tuning, eval, optimization
- **[07.4 - Deployment](./07%20-%20Research%20Track/07.4%20-%20Deployment)** &mdash; Model storage/serving, API development, deployment, post-deployment

### [08 - AI Architect Track](./08%20-%20AI%20Architect%20Track)
System architecture, governance, and strategy for senior engineers and AI architects.
- **[08.1 - AI System Architecture & Design Patterns](./08%20-%20AI%20Architect%20Track/08.1%20-%20AI%20System%20Architecture%20%26%20Design%20Patterns)** &mdash; Distributed ML systems, microservices, event-driven pipelines, model serving patterns
- **[08.2 - MLOps at Scale & Platform Engineering](./08%20-%20AI%20Architect%20Track/08.2%20-%20MLOps%20at%20Scale%20%26%20Platform%20Engineering)** &mdash; Self-serve ML platforms, CI/CD, monitoring, platform governance
- **[08.3 - Cloud AI Platform Architecture (Multi-cloud)](./08%20-%20AI%20Architect%20Track/08.3%20-%20Cloud%20AI%20Platform%20Architecture%20(Multi-cloud))** &mdash; Cloud-agnostic AI, multi-cloud DR, cost arbitrage, IaC
- **[08.4 - LLM and LMM Architecture and RAG Systems Design](./08%20-%20AI%20Architect%20Track/08.4%20-%20LLM%20and%20LMM%20Architecture%20and%20RAG%20Systems%20Design)** &mdash; LLM serving, RAG architectures, multimodal pipelines, guardrails
- **[08.5 - AI Governance, Security & Compliance](./08%20-%20AI%20Architect%20Track/08.5%20-%20AI%20Governance%2C%20Security%20%26%20Compliance)** &mdash; Model risk management, regulations, fairness, bias auditing
- **[08.6 - Performance Engineering & Cost Optimization](./08%20-%20AI%20Architect%20Track/08.6%20-%20Performance%20Engineering%20%26%20Cost%20Optimization)** &mdash; Model compression, inference acceleration, FinOps for AI
- **[08.7 - Data Architecture for AI](./08%20-%20AI%20Architect%20Track/08.7%20-%20Data%20Architecture%20for%20AI)** &mdash; Data mesh, streaming features, feature stores, data contracts
- **[08.8 - AI Strategy, Roadmap & ROI](./08%20-%20AI%20Architect%20Track/08.8%20-%20AI%20Strategy%2C%20Roadmap%20%26%20ROI)** &mdash; Business cases, build-vs-buy, OKRs, organizational design
- **[08.9 - Emerging Architectures & Research Translation](./08%20-%20AI%20Architect%20Track/08.9%20-%20Emerging%20Architectures%20%26%20Research%20Translation)** &mdash; Agentic systems, neuro-symbolic AI, world models, test-time compute scaling
- **Capstones:**
  - **[08.10 - Enterprise AI Platform](./08%20-%20AI%20Architect%20Track/08.10%20-%20Capstone%20Enterprise%20AI%20Platform)** &mdash; Full platform design with ADRs, cost model, migration plan, live review
  - **[08.11 - LLM Knowledge System](./08%20-%20AI%20Architect%20Track/08.11%20-%20Capstone%20LLM%20Knowledge%20System)** &mdash; RAG + agents + multi-modal search for enterprise knowledge
  - **[08.12 - Real-Time Fraud Detection](./08%20-%20AI%20Architect%20Track/08.12%20-%20Capstone%20Real-Time%20Fraud%20Detection)** &mdash; Sub-100ms streaming ML at 20k TPS with PCI-DSS compliance
  - **[08.13 - Multi-Cloud AI Infrastructure](./08%20-%20AI%20Architect%20Track/08.13%20-%20Capstone%20Multi-Cloud%20AI%20Infrastructure)** &mdash; Cloud-agnostic migration, cost arbitrage, DR across clouds
  - **[08.14 - AI Governance Program](./08%20-%20AI%20Architect%20Track/08.14%20-%20Capstone%20AI%20Governance%20Program)** &mdash; NIST AI RMF, EU AI Act, red teaming, fairness thresholds, 3LoD org

### [09 - Real World Business Cases](./09%20-%20Real%20World%20Business%20Cases)
End-to-end AI implementation case studies with full lifecycle documentation, resume content, and LinkedIn posts.
- **[01 - Retail Demand Forecasting](./09%20-%20Real%20World%20Business%20Cases/01-retail-demand-forecasting.md)** &mdash; AI-driven inventory optimization for omnichannel retail ($86M annual impact)
- **[02 - Insurance Claims Intelligence](./09%20-%20Real%20World%20Business%20Cases/02-insurance-claims-intelligence.md)** &mdash; AI-powered fraud detection & claims automation for P&C insurance ($213M annual impact)

---

## Speaker Script — From Notebook to Production: AI Delivery Across the SDLC

*This is a 60-minute talk that maps the knowledge in this repository to the full AI project delivery lifecycle — from discovery through retirement — with quality gates and compliance checkpoints at every stage.*

---

### [0:00–5:00] Opening — The "Model in a Notebook" Trap

Most AI projects fail not because the model is wrong, but because nobody connects model-building to the system that surrounds it. A Jupyter notebook with 98% accuracy might look like success — but that project is already failing. 85% of ML projects never reach production. The reason isn't model quality. It's everything else.

We're going to walk through 9 phases of AI value delivery. At each phase, I'll show you what you need to know, where this roadmap teaches it, and where you'll get burned if you skip it.

---

### [5:00–15:00] Act 1 — Before You Write a Single Line of Model Code

**Discovery — Is ML the right solution?**
The roadmap's `05.1 - ML System Design` teaches you to ask the hard question first: should we be doing ML at all? Pair it with `02.1 - ML Fundamentals` to distinguish problems that need ML from problems that need a simple heuristic. The quality gate here is an ROI analysis and data availability check. The compliance flag is your first EU AI Act classification — is this a high-risk use case? If you're building a hiring tool or a credit scoring system, that classification triggers a conformity assessment before you write any code.

**Problem Formulation — What exactly are we optimizing?**
This is where most teams go wrong. They pick a metric because it's easy to measure — not because it drives business value. `05.1.1` teaches you to map business objectives to ML tasks and define two metrics: the optimization metric the model trains on, and the success metric the business actually cares about. You document the asymmetry between false positives and false negatives — because that drives your decision threshold, and threshold choice is often more impactful than model architecture. If you're in a regulated industry, the compliance flag is now active: high-risk AI systems require a risk management system from the start.

**Data System Design — Can we trust the data?**
`05.1.2` covers data pipeline architecture, storage decisions, and validation. `01.2 - SQL` gives you the querying skills to extract and profile data. But the critical lesson is: you need drift detection from day one. If you don't know how your data distribution will change six months after launch, you're building a model that will silently fail. GDPR requires a data protection impact assessment at this stage. If you're using personal data, you need to document purpose limitation, data minimization, and retention policies.

**Feature System Design — Training-serving consistency**
The single most common production bug in ML systems is training-serving skew — where the feature at inference time differs from what the model learned on. `05.1.3` and `02.2 - Feature Engineering` teach you to build feature registries, write feature computation code once, and use it consistently across training and serving pipelines. The quality gate: training-serving skew must be zero. The compliance flag: if your features include protected attributes (age, gender, race), you've triggered a bias audit requirement.

*The most expensive mistake is building a great model on the wrong problem. These four phases catch that mistake before you've spent months training.*

---

### [15:00–30:00] Act 2 — Building That Actually Ships

**Modeling Strategy — Which algorithm, and how do we train it?**
This is what most people think AI is. The roadmap covers it thoroughly: `02.3` with 13 algorithm families from linear regression through gradient boosting to SVMs and clustering. Then `03.x` takes you deep into deep learning — from manual backpropagation in NumPy through CNNs, RNNs, transformers, GANs, and diffusion models. `04.x` gives you hands-on PyTorch skills for NLP, CV, and time series.

But modeling isn't just about picking an algorithm. `05.1.4` teaches you the discipline around it: establish a baseline first, maintain a modeling leaderboard, document every experiment. The quality gate at this phase is a completed model card — documenting intended use, limitations, training data, and evaluation results. This isn't optional paperwork; it's the artifact that tells your future self (or your regulator) what this model does and doesn't do.

**Evaluation Strategy — Does it actually work, and for whom?**
`05.1.5` covers offline evaluation (time-aware splits, leakage-resistant validation), online evaluation (A/B testing, shadow deployments, canary releases), and the critical one that most teams miss: slice-based evaluation. You don't just evaluate overall accuracy — you check performance across demographic groups, data segments, and edge cases.

This is where quality and compliance converge. NYC Local Law 144 requires third-party bias audits for AI hiring tools. The EU AI Act requires disaggregated evaluation by protected attributes for high-risk systems. `02.4 - Advanced ML Concepts` teaches you interpretability with SHAP and LIME, which gives you the tools to understand *why* your model makes different decisions for different groups. If you can't explain it, you can't deploy it in a regulated environment.

**Serving — How do we get predictions to users?**
`05.2.6` teaches you model serving patterns: batch inference for offline scoring, real-time REST endpoints with FastAPI, and serverless deployment for variable workloads. `05.3.4` takes it further with optimized inference — TensorRT, ONNX Runtime, vLLM, kernel fusion, continuous batching. If your model can't serve a prediction in under 200ms at p99, you don't have a model — you have a science project.

`06.2.1` covers orchestration patterns for LLM applications: chains, routers, parallel execution, and evaluator loops. For agentic systems, `06.3` teaches you ReAct, tool use, and frameworks like LangGraph. The quality gates are latency and throughput SLOs, a documented fallback chain, and cost per prediction within budget. The compliance requirement: audit logging of every prediction, because GDPR Article 22 gives users the right to an explanation for automated decisions.

*This is where most courses stop — "here's how to train a model." The roadmap goes further, through the deployment patterns that separate demos from products.*

---

### [30:00–45:00] Act 3 — Keeping It Alive

**Feedback & Monitoring — Is the model still working?**
Models degrade. Data distributions shift. User behavior changes. `05.1.7` teaches you the three types of drift: data drift (input distribution changes), concept drift (the relationship between inputs and targets changes), and prediction drift (the model's output distribution changes). `05.2.4` covers monitoring tools like Evidently, Prometheus, and Grafana.

The quality gate is automated drift detection with a defined retraining trigger. If you can't detect that your model has gone stale within a business-relevant time window, you shouldn't be in production.

For LLM applications, `06.4.3` adds observability layers: latency and throughput tracking, token usage and cost monitoring, response quality scoring, user feedback collection, and hallucination detection with tools like LangSmith and Helicone.

The compliance imperative: any model failure that affects users requires an incident response plan. The roadmap's governance section defines a 7-step process: detect → triage → mitigate → investigate → communicate → remediate → review. If a credit-scoring model drifts and starts rejecting qualified applicants, you need to know within hours, not weeks.

**Governance & Ethics — Is it fair, transparent, and legal?**
`05.1.8` covers bias audits, model cards, and responsible ML practices. `06.4.4` adds LLM-specific safety: defense-in-depth with input guardrails (prompt injection detection), context filters (PII in retrieved documents), output guardrails (harmful content filtering), and content moderation (hate, harassment, self-harm).

`06.4.5` maps the full regulatory landscape: the EU AI Act's four risk tiers, GDPR's right to explanation, NYC Local Law 144's bias audit requirement, the US Executive Order on AI safety testing, and emerging regulations in China, Canada, and Brazil.

The quality gate for governance is straightforward: a bias report accompanies every model release, with metrics like disparate impact ratio and equal opportunity difference tracked over time. The model card is published and versioned. The system card (for deployed applications) documents purpose, data sources, limitations, and oversight mechanisms.

*This is the section that most organizations skip until they get paged at 3 AM. By then, it's too late for prevention — you're in incident response.*

---

### [45:00–55:00] Act 4 — The Research Frontier

The first 45 minutes covered delivery of standard AI systems. This section covers what you need when standard isn't enough — when latency, cost, or capability requirements push you beyond what off-the-shelf solutions can deliver.

**Inference optimization** (`07.4.2`): Kernel fusion reduces GPU kernel launches from 15 per transformer layer to 3-4, cutting latency by 37% and HBM reads by 44%. Tensor parallelism shards models across GPUs for sub-second inference on 70B+ parameter models. These techniques change the economics of deployment — use cases that were previously uneconomical become viable.

**LLM alignment** (`07.2.3`): RLHF, DPO, and Constitutional AI reduce the guardrail burden. A well-aligned model needs less safety infrastructure because it refuses harmful requests by default. This is an active research area, and the roadmap covers the practical techniques.

**Quantization** (`07.3.3`): GPTQ, AWQ, and GGUF compress models to INT4 with minimal quality loss. This runs 70B models on consumer GPUs — a 24GB RTX 4090 instead of an 80GB A100. The cost impact on inference is transformative.

**Speculative decoding** (`07.3.4`): A small draft model generates tokens quickly, and the large model verifies them in parallel. This gives 2-3× latency improvement for LLM generation with zero quality loss.

**Scalable training** (`05.3.1`): FSDP, DeepSpeed ZeRO stages, and 3D parallelism (tensor + pipeline + data parallelism) enable training models that don't fit on a single GPU. This is the infrastructure behind every large model you've used.

*These are the techniques that separate teams that can deliver frontier AI from teams that are stuck calling GPT-4 APIs. They're not academic papers — they're deployment strategies that directly impact your latency, cost, and capability.*

---

### [55:00–60:00] Closing — The Roadmap as Your Delivery Guide

Let me connect the arc. The roadmap starts with prerequisites — the Python, math, and cloud skills that underpin everything. Then machine learning — 13 algorithms, feature engineering, 37 projects. Then deep learning — from manual backpropagation through diffusion models. Then production engineering — MLOps, ML system design, the full SDLC with gates and roles. Then the AI engineer track — LLMs, RAG, agents, guardrails, governance. And finally the research frontier — attention optimization, quantization, speculative decoding.

The metric that matters isn't model accuracy. It's time from business requirement to production prediction with compliance sign-off. The roadmap collapses this timeline because it teaches you every phase of delivery — not just the modeling step.

668 files, one path. Every phase of delivery, from "should we use AI?" to "how do we retire this model safely?" Every quality gate. Every compliance checkpoint. The roadmap doesn't teach you to build a model. It teaches you to deliver an AI system that survives contact with the real world.

**Call to action:** Fork the repo. Clone it. Pick one section you're weakest in — whether that's the math, the MLOps, or the governance — and complete it before you start your next project. The talk slides and SDLC reference document are already in the repo at `05.1 - ML System Design`. Use them as a template for your own projects.
