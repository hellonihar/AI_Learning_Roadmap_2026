# ML & Deep Learning Fundamentals — Q&A

## Q1: Explain the Transformer architecture and its key components.

The Transformer (Vaswani et al., 2017) replaces recurrence with attention for parallel processing.

**Key components:**
- **Input Embeddings** — tokens → dense vectors
- **Positional Encoding** — sine/cosine (or RoPE, ALiBi) to encode sequence order
- **Self-Attention** — each token attends to all others; Q, K, V projections
- **Multi-Head Attention** — parallel attention heads capture different relationships
- **Feed-Forward Network** — two linear layers with a GELU/ReLU in between
- **Layer Norm + Residual** — stabilizes training, enables deep stacks
- **Encoder stack** (bidirectional) / **Decoder stack** (causal masking)

Interviewers want you to explain not just what, but why: parallelization enables scaling, attention solves long-range dependency issues, residual connections solve vanishing gradients in deep nets.

---

## Q2: What is the attention mechanism? How does self-attention differ from multi-head attention?

**Attention** computes a weighted sum over values, where weights are determined by the compatibility of a query with keys:

`Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V`

**Self-attention** = Q, K, V all come from the same sequence. Each token learns which other tokens to focus on.

**Multi-head attention** = run self-attention H times in parallel with different learned projections, then concatenate. Each head can specialize (e.g., syntax, semantics, positional relationships). The dimension per head is d_model / H, keeping compute roughly constant.

---

## Q3: Explain the bias-variance tradeoff with examples.

- **Bias** = error from simplifying assumptions (underfitting). High bias → model misses patterns.
- **Variance** = error from sensitivity to training data fluctuations (overfitting). High variance → model memorizes noise.
- **Tradeoff**: increasing model complexity reduces bias but increases variance.

**Example:** Linear regression (high bias, low variance) vs. deep decision tree (low bias, high variance). The optimal model balances both to minimize total error.

**In practice:** Use cross-validation to find the sweet spot. Regularization (L1/L2) increases bias slightly to reduce variance significantly.

---

## Q4: How does backpropagation work? Describe the process step by step.

1. **Forward pass** — input propagates through the network to produce a prediction
2. **Loss computation** — compare prediction to ground truth (e.g., cross-entropy)
3. **Gradient computation** — apply chain rule from output layer backward through each layer: dL/dW = dL/dy * dy/dz * dz/dW
4. **Parameter update** — W = W - lr * dL/dW (via SGD/Adam)

Key insight: gradients flow backward, reusing intermediate values from the forward pass (stored in memory). This is why activation functions need non-zero gradients — saturation kills learning.

---

## Q5: Compare Batch Normalization vs. Layer Normalization. When would you use each?

| Aspect | Batch Norm | Layer Norm |
|--------|-----------|------------|
| Normalizes across | batch dimension | feature dimension |
| Batch dependent? | Yes — behavior differs train vs. inference | No — same at train and inference |
| Good for | CNNs, fixed-size inputs | RNNs, Transformers, variable-length sequences |
| Issues | Small batch sizes break statistics; sequence models can't use it | Slightly more compute |

**Transformers use LayerNorm** because sequences have variable length and batch norm's reliance on batch statistics is problematic for NLP.

---

## Q6: What are scaling laws in LLMs? Explain Chinchilla optimal compute.

**Scaling laws** (Kaplan et al., 2020) show model performance follows a power-law relationship with:
- Model size (parameters)
- Dataset size (tokens)
- Compute (FLOPs)

**Chinchilla scaling** (Hoffmann et al., 2022) found that most LLMs were undertrained — for a given compute budget, model size and training tokens should be scaled equally. For every 2x in model size, training tokens should also increase 2x.

**Practical impact:** Chinchilla optimal models (e.g., Llama 3 — 15T tokens for 8B params) outperform larger models trained on fewer tokens at the same compute cost.

---

## Q7: What is the difference between supervised, unsupervised, and reinforcement learning?

| Paradigm | Data | Feedback | Example |
|----------|------|----------|---------|
| Supervised | Labeled (X, y) | Direct error signal | Classification, regression |
| Unsupervised | Unlabeled (X) | Intrinsic structure | Clustering, dimensionality reduction |
| Self-supervised | Unlabeled (X) | Pretext task (predict part of X) | Masked LM, contrastive learning |
| Reinforcement | States, actions, rewards | Delayed reward signal | Game playing, robotics |

Modern LLMs use all three: self-supervised pre-training (next token prediction), supervised fine-tuning, and RLHF (RL).

---

## Q8: Explain vanishing/exploding gradients and how to mitigate them.

**Vanishing gradients** — gradients become extremely small in deep networks (especially with sigmoid/tanh), causing early layers to stop learning. **Exploding gradients** — gradients grow exponentially, causing numerical instability.

**Mitigations:**
- ReLU/GELU activations (non-saturating)
- Residual connections (gradient highway)
- Layer/Batch Normalization
- Gradient clipping (cap gradients at a threshold)
- Proper weight initialization (Xavier, Kaiming)
- Lower learning rates + learning rate warmup

Transformers are notably stable due to residual connections + LayerNorm, enabling 100+ layer stacks.

---

## Q9: Compare SGD, Adam, and AdamW optimizers.

| Optimizer | Key Idea | Pros | Cons |
|-----------|---------|------|------|
| SGD | Basic gradient step + momentum | Simple, good generalization | Slow convergence, sensitive to LR |
| Adam | Adaptive LR per param + momentum | Fast convergence, robust to LR | Can generalize worse than SGD; requires weight decay fix |
| AdamW | Adam with decoupled weight decay | Best of both — fast convergence + good generalization | Slightly more hyperparameters |

**Current consensus:** AdamW is the default for transformer training. SGD+momentum still works well for CNNs and when you want better generalization with careful tuning.

---

## Q10: What evaluation metrics would you use for a classification vs. regression problem?

**Classification:**
- **Accuracy** — overall correctness (bad for imbalanced classes)
- **Precision** — TP / (TP + FP) — "how many predicted positives are real?"
- **Recall** — TP / (TP + FN) — "how many real positives did we catch?"
- **F1-Score** — harmonic mean of precision and recall
- **AUC-ROC** — threshold-independent measure of separability
- **LogLoss** — probabilistic calibration

**Regression:**
- **MSE/RMSE** — penalizes large errors heavily
- **MAE** — robust to outliers
- **R²** — proportion of variance explained
- **MAPE** — percentage error (useful for business context)

For LLMs specifically: perplexity (intrinsic), BLEU/ROUGE (not recommended), LLM-as-judge, human evaluation, task-specific metrics.

---

## Q11: Explain Grouped Query Attention (GQA) and why it's used.

Standard multi-head attention uses separate K and V heads per query head (expensive). Grouped Query Attention shares K and V heads across groups of query heads.

- **MHA** (Multi-Head Attention): N Q heads, N K/V heads — most expressive, most memory
- **GQA**: N Q heads, G K/V heads (G < N) — intermediate
- **MQA** (Multi-Query Attention): N Q heads, 1 K/V head — least memory, slight quality loss

GQA is used in Llama 2/3, Mistral, and most modern LLMs. It reduces KV cache memory by ~2-4x during inference with minimal quality degradation.

---

## Q12: How does mixed precision training work (FP16, BF16)?

Mixed precision uses lower-precision (16-bit) arithmetic for most operations while keeping critical parts in FP32:
- **Forward/backward pass** — FP16/BF16 (faster, less memory)
- **Loss scale** — multiply loss to prevent underflow in FP16 gradients
- **Master weights** — kept in FP32 for accurate gradient updates
- **BF16** (bfloat16) has same exponent range as FP32 — no loss scaling needed, preferred for training

**Benefits:** ~2x speedup on modern GPUs (Tensor Cores), ~50% memory reduction. Nearly all LLM training uses mixed precision.

---

## Q13: What is the KV cache and how does it impact inference?

The **KV cache** stores the Key and Value tensors from previous tokens during autoregressive decoding. Instead of recomputing attention over the entire context at each step, we cache K and V:

- **Without cache:** O(n²) per token — compute attention over all n tokens each step
- **With cache:** O(n) per token — only compute attention with the new token's Q

**Memory impact:** KV cache size = batch_size * seq_len * num_layers * d_k * 2 * precision_bytes. For a 70B model at 4K context with batch 32, this can be ~40GB+.

**Optimizations:** GQA/MQA, KV cache quantization, sliding window attention, prompt caching.

---

## Q14: Compare encoder-only, decoder-only, and encoder-decoder architectures.

| Type | Architecture | Example | Use Case |
|------|-------------|---------|----------|
| Encoder-only | Bidirectional self-attention | BERT | Classification, NER, embeddings |
| Decoder-only | Causal (masked) self-attention | GPT, Llama | Generation, chatbots, code |
| Encoder-Decoder | Encoder (bidirectional) + Decoder (cross-attention) | T5, BART | Translation, summarization |

**Trend:** Decoder-only models (GPT family) have become dominant due to their simplicity, scalability, and in-context learning abilities. Encoder-only models are still best for representation learning tasks. Encoder-decoder is used when the input and output distributions differ significantly.

---

## Q15: How would you debug a model that isn't converging?

1. **Check data** — are labels correct? Is data shuffled? Any NaN/inf values?
2. **Start small** — overfit a single batch (if it can't, there's a bug in the model)
3. **Gradient checks** — are gradients reasonable? (compare to numerical gradients)
4. **Learning rate** — too high (oscillation/NaN) or too low (stagnation)? Use LR range test
5. **Loss landscape** — monitor loss curves. Flat? Diverging? Spiky?
6. **Architecture** — wrong output activation? Missing residual connection? Wrong loss function?
7. **Normalization** — are activations exploding/vanishing? Check LayerNorm placement
8. **Weight initialization** — wrong init can kill gradients (e.g., sigmoid without proper init)
9. **Debugging tools** — TensorBoard/W&B for histogram of gradients, weights, activations
10. **Simplify** — remove augmentation, reduce depth, use a known-good baseline, then reintroduce complexity

---

## External Resources

- [DataCamp: Top 35 AI Interview Questions (2026)](https://www.datacamp.com/blog/ai-interview-questions)
- [InterviewBit: LLM Interview Questions](https://www.interviewbit.com/llm-interview-questions-answers/)
- [Precision AI Academy: 60+ AI & ML Questions](https://precisionaiacademy.com/ai-interview-questions)
- [GitHub: aakriti1318/interview_questions/AI_Architect](https://github.com/aakriti1318/interview_questions/tree/main/AI_Architect)
- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Paper: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Paper: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
