# 07.2.1 — From GPT-1 to GPT-4

## Overview

The GPT (Generative Pre-trained Transformer) series by OpenAI traces the evolution from a proof-of-concept language model to the foundation of a new AI paradigm. Each generation brought architectural improvements, data scaling, and algorithmic advances that culminated in emergent abilities unseen in smaller models.

## GPT-1 (2018) — The Blueprint

**Paper**: *Improving Language Understanding by Generative Pre-Training*

GPT-1 introduced the **generative pre-training + discriminative fine-tuning** paradigm. A 12-layer, 117M-parameter decoder-only Transformer was pre-trained on BooksCorpus (~5GB of text) using the standard language modelling objective:

$$
\mathcal{L} = -\sum_i \log P(x_i \mid x_{<i})
$$

After pre-training, task-specific classification heads were added and fine-tuned on labelled data. This approach outperformed purely supervised models on most NLP benchmarks, establishing pre-training as the dominant method.

## GPT-2 (2019) — Scaling Works

**Paper**: *Language Models are Unsupervised Multitask Learners*

GPT-2 scaled to 1.5B parameters (48 layers) and was trained on WebText (40GB of curated Reddit links). The key insight: with sufficient scale, a language model naturally performs many tasks without explicit fine-tuning (zero-shot transfer). This was the first demonstration that **scaling alone** produces general-purpose linguistic competence.

GPT-2 also used **layer normalisation moved to the input** of each sub-block and increased the context window to 1024 tokens.

## GPT-3 (2020) — In-Context Learning

**Paper**: *Language Models are Few-Shot Learners*

GPT-3 scaled to 175B parameters (96 layers, 96 attention heads, 12288-dimensional embeddings), trained on 570GB of text (CommonCrawl, WebText2, Books, Wikipedia). The critical finding was **in-context learning**: by providing a few examples in the prompt (without gradient updates), GPT-3 could perform tasks it was never explicitly trained for.

GPT-3 introduced **sparse attention patterns** for efficiency and used alternating dense and locally banded sparse attention layers.

$$
P(\text{answer} \mid \text{examples} + \text{question})
$$

## GPT-3.5 / InstructGPT (2022) — Alignment

**Paper**: *Training Language Models to Follow Instructions*

GPT-3.5 added **instruction tuning** and **RLHF** (Reinforcement Learning from Human Feedback). Models were fine-tuned on human-written instruction-output pairs, then a reward model was trained from human preferences, and PPO optimised the policy:

$$
\text{Reward model: } R_\phi(x, y) \quad \text{Policy: } \pi_\theta(y \mid x)
$$

This alignment step dramatically improved helpfulness, honesty, and safety.

## GPT-4 (2023) — Multimodal and Reliable

**Paper**: *GPT-4 Technical Report*

GPT-4 is a **multimodal** model (text + image input) estimated at ~1.8T parameters using a mixture-of-experts (MoE) architecture with 8 experts per feedforward layer. It introduced:

- Improved factual accuracy and reliability
- Longer context (initially 8K, later 128K tokens)
- Vision capabilities
- Steerability via system prompts

GPT-4 demonstrated **emergent abilities** — capabilities not present in smaller models that appear suddenly at scale, including reasoning, coding, and tool use.

## Practical Context

The GPT evolution mirrors the broader LLM landscape: larger models, better alignment, multimodal inputs, and efficient inference. Understanding this lineage helps contextualise modern models like Llama 3, Gemini, and Claude.
