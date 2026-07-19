# Resources — Research Track

## Foundational Papers

### Reinforcement Learning
| Paper | Year | Citation |
|-------|------|----------|
| **Playing Atari with Deep Reinforcement Learning** (DQN) | 2013 | Mnih et al. |
| **Human-level control through deep reinforcement learning** | 2015 | Mnih et al., Nature |
| **Continuous control with deep reinforcement learning** (DDPG) | 2016 | Lillicrap et al. |
| **Asynchronous Methods for Deep Reinforcement Learning** (A3C) | 2016 | Mnih et al. |
| **Proximal Policy Optimization Algorithms** (PPO) | 2017 | Schulman et al. |
| **Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL** (SAC) | 2018 | Haarnoja et al. |
| **Mastering the Game of Go without Human Knowledge** (AlphaZero) | 2017 | Silver et al., Nature |
| **Playing Atari with Sample Efficiency** (EfficientZero) | 2021 | Ye et al. |

### Transformers & LLMs
| Paper | Year | Citation |
|-------|------|----------|
| **Attention Is All You Need** | 2017 | Vaswani et al. |
| **BERT: Pre-training of Deep Bidirectional Transformers** | 2018 | Devlin et al. |
| **Language Models are Unsupervised Multitask Learners** (GPT-2) | 2019 | Radford et al. |
| **Language Models are Few-Shot Learners** (GPT-3) | 2020 | Brown et al. |
| **Training Language Models to Follow Instructions** (InstructGPT) | 2022 | Ouyang et al. |
| **GPT-4 Technical Report** | 2023 | OpenAI |
| **The Llama 3 Herd of Models** | 2024 | Grattafiori et al. |

### Scaling Laws
| Paper | Year | Citation |
|-------|------|----------|
| **Scaling Laws for Neural Language Models** | 2020 | Kaplan et al. |
| **Training Compute-Optimal Large Language Models** (Chinchilla) | 2022 | Hoffmann et al. |
| **Emergent Abilities of Large Language Models** | 2022 | Wei et al. |
| **Are Emergent Abilities a Mirage?** | 2023 | Schaeffer et al. |

### Alignment
| Paper | Year | Citation |
|-------|------|----------|
| **Deep Reinforcement Learning from Human Preferences** | 2017 | Christiano et al. |
| **Fine-Tuning Language Models from Human Preferences** (RLHF) | 2020 | Ziegler et al. |
| **Training a Helpful and Harmless Assistant from Human Feedback** | 2022 | Bai et al. (Anthropic) |
| **Constitutional AI: Harmlessness from AI Feedback** | 2022 | Bai et al. |
| **Direct Preference Optimization** (DPO) | 2023 | Rafailov et al. |
| **KTO: Model Alignment as Prospect Theoretic Optimization** | 2024 | Ethayarajh et al. |
| **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** | 2023 | Zheng et al. |

### Efficient Attention & Architecture
| Paper | Year | Citation |
|-------|------|----------|
| **Generating Long Sequences with Sparse Transformers** | 2019 | Child et al. |
| **Fast Transformer Decoding: One Write-Head is All You Need** (MQA) | 2019 | Shazeer |
| **FlashAttention: Fast and Memory-Efficient Exact Attention** | 2022 | Dao et al. |
| **FlashAttention-2: Faster Attention with Better Parallelism** | 2023 | Dao |
| **GQA: Training Generalized Multi-Query Transformer Models** | 2023 | Ainslie et al. |
| **Ring Attention with Blockwise Transformers** | 2023 | Liu et al. |

### Inference & Serving
| Paper | Year | Citation |
|-------|------|----------|
| **Efficient Memory Management for LLM Serving** (PagedAttention / vLLM) | 2023 | Kwon et al. |
| **ORCA: A Distributed Serving System for Transformer-Based Models** | 2022 | Yu et al. |
| **SpecInfer: Accelerating Generative LLM Serving** | 2023 | Miao et al. |
| **Medusa: Simple LLM Inference Acceleration** | 2023 | Cai et al. |
| **TensorRT-LLM: A TensorRT Toolset for LLM Inference** | 2023 | NVIDIA |

### Quantization
| Paper | Year | Citation |
|-------|------|----------|
| **GPTQ: Accurate Post-Training Quantization for GPTs** | 2022 | Frantar et al. |
| **AWQ: Activation-Aware Weight Quantization** | 2023 | Lin et al. |
| **QLoRA: Efficient Finetuning of Quantized LLMs** | 2023 | Dettmers et al. |
| **LLM.int8(): 8-bit Matrix Multiplication for LLMs** | 2022 | Dettmers et al. |
| **SmoothQuant: Accurate and Efficient PTQ for LLMs** | 2022 | Xiao et al. |
| **KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache** | 2024 | Liu et al. |

---

## Books & Courses

| Title | Author | Focus |
|-------|--------|-------|
| **Reinforcement Learning: An Introduction** (2nd ed.) | Sutton & Barto | Complete RL theory |
| **Algorithms for Reinforcement Learning** | Szepesvári | Concise mathematical treatment |
| **Speech and Language Processing** | Jurafsky & Martin | NLP fundamentals |
| **The Annotated Transformer** | Harvard NLP (online) | Transformer walkthrough |
| **CS224N: NLP with Deep Learning** | Stanford | LLM foundations |
| **CS285: Deep Reinforcement Learning** | UC Berkeley | Deep RL |
| **Hugging Face NLP Course** | Hugging Face | Practical transformers |

---

## Software & Frameworks

| Tool | Use |
|------|-----|
| **Gymnasium** | RL environments |
| **Stable-Baselines3** | RL algorithm implementations |
| **Ray RLlib** | Distributed RL |
| **Hugging Face Transformers** | LLM loading & inference |
| **vLLM** | LLM serving |
| **llama.cpp** | CPU inference with GGUF |
| **bitsandbytes** | Quantization |
| **TensorRT-LLM** | Optimised inference |
| **FlashAttention** | Efficient attention kernel |
| **Weights & Biases** | Experiment tracking |

---

## Leaderboards & Benchmarks

| Name | URL | Focus |
|------|-----|-------|
| **LMSys Chatbot Arena** | chat.lmsys.org | LLM ranking by humans |
| **Open LLM Leaderboard** | huggingface.co/spaces/open-llm-leaderboard | Open model benchmarks |
| **MMLU** | paperswithcode.com/dataset/mmlu | Multi-task understanding |
| **HumanEval** | github.com/openai/human-eval | Code generation |
| **RL Benchmark** | github.com/google-research/rl-benchmark | Reproducible RL results |
