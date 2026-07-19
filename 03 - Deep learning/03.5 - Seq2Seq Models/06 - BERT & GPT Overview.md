# BERT & GPT Overview

## Introduction

BERT and GPT represent the two dominant paradigms that emerged from the Transformer architecture: **encoder-only** models optimized for understanding, and **decoder-only** models optimized for generation. Both demonstrated that large-scale pre-training on unlabeled text, followed by fine-tuning, produces powerful representations.

## BERT: Bidirectional Encoder Representations from Transformers

### Architecture

BERT is a **stack of Transformer encoder layers** (no decoder). The base version uses $L = 12$ layers, $d_{\text{model}} = 768$, $h = 12$ attention heads, and $d_{ff} = 3072$ (~110M params). BERT-Large uses $L = 24$, $d_{\text{model}} = 1024$, $h = 16$ (~340M params).

### Pre-Training Objectives

BERT is pre-trained on two unsupervised tasks:

**1. Masked Language Modeling (MLM)**
15% of input tokens are randomly masked, and BERT must predict the original token from context. Of the chosen tokens:
- 80% replaced with `[MASK]`
- 10% replaced with a random token
- 10% left unchanged

This forces BERT to learn bidirectional context — unlike previous left-to-right language models, BERT uses both left and right context to predict each token.

**2. Next Sentence Prediction (NSP)**
Given two segments A and B, the model predicts whether B follows A in the original document (50% positive, 50% negative). This teaches sentence-level relationships, useful for tasks like question answering (QA) and natural language inference (NLI).

### Key Properties

- **Bidirectional context**: Each token attends to all other tokens (no masking).
- **Encoder-only**: No generative capability; produces rich contextualized representations.
- **Fine-tuning**: A task-specific classification head is added on top of the `[CLS]` token or individual token representations.

### Use Cases

- Text classification (sentiment, topic labeling)
- Named entity recognition (NER)
- Question answering (SQuAD)
- Natural language inference (NLI)
- Sentence similarity / paraphrase detection
- Feature extraction for downstream models

## GPT: Generative Pre-Training

### Architecture

GPT is a **stack of Transformer decoder layers** (no encoder). GPT-1 (2018) used $L = 12$ layers with masked self-attention. GPT-2 scaled to 1.5B params, GPT-3 to 175B, GPT-4 to an undisclosed size.

### Pre-Training Objective

GPT is pre-trained using **standard autoregressive language modeling**:

$$P(\mathbf{x}) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

The model predicts the next token given all previous tokens, using a **causal (look-ahead) mask** to prevent attending to future positions.

### Key Properties

- **Unidirectional (causal) context**: Each token only attends to previous tokens.
- **Decoder-only**: Naturally generative; can produce arbitrary-length text.
- **Zero-shot / few-shot learning**: Large GPT variants can perform tasks from natural language instructions and in-context examples without fine-tuning.

### Use Cases

- Text generation (stories, articles, code)
- Dialogue / chatbots
- Summarization (via prompt engineering)
- Translation (via prompting)
- Code generation (Codex / GitHub Copilot)
- In-context learning from examples (few-shot)

## Architectural Differences

| Component | BERT | GPT |
|---|---|---|
| Transformer part | Encoder only | Decoder only |
| Attention mask | None (bidirectional) | Causal (left-to-right) |
| Pre-training | MLM + NSP | Autoregressive LM |
| Context | Bidirectional | Unidirectional (left) |
| Generation | No (needs head) | Yes (autoregressive) |
| Typical size | 110M–340M | 110M–175B+ |
| Fine-tuning | Task-specific heads | Prompt-based / fine-tune |
| Output | Fixed-size vectors | Probability over vocab |

## Why This Matters

The encoder vs. decoder choice reflects a fundamental trade-off:

- **BERT-like models** excel at tasks requiring full context understanding, where the entire input is available (e.g., classification, QA where the answer is contained in the passage).
- **GPT-like models** excel at generation tasks where the output is a continuation of the input, and at zero-shot generalization due to the language modeling objective's natural alignment with diverse tasks.

## Unified Models

T5 (Text-to-Text Transfer Transformer) uses a full encoder-decoder Transformer, treating every NLP task as text-to-text. This bridges the two paradigms — the encoder provides bidirectional understanding of the input, and the decoder generates the output autoregressively.

## Summary

BERT and GPT both stem from the Transformer but make opposite architectural choices: BERT uses the encoder for bidirectional understanding, while GPT uses the decoder for autoregressive generation. Understanding these design decisions is essential for choosing the right pre-trained model and for designing new architectures.
