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
