# ML & Deep Learning Fundamentals

## Key Concepts
- Supervised, Unsupervised, Semi-supervised, Reinforcement Learning
- Bias-Variance Tradeoff
- Overfitting / Underfitting — regularization (L1, L2, Dropout, Early Stopping)
- Loss functions: Cross-Entropy, MSE, MAE, Huber
- Optimization: SGD, Adam, AdamW, learning rate scheduling
- Evaluation metrics: Accuracy, Precision, Recall, F1, AUC-ROC, LogLoss

## Neural Networks
- Perceptron, MLP, activation functions (ReLU, GELU, SwiGLU)
- Backpropagation & gradient descent variants
- Batch Normalization, Layer Normalization
- Vanishing/Exploding gradients

## Deep Learning Architectures
- CNNs: convolutions, pooling, ResNets, EfficientNet
- RNNs / LSTMs / GRUs — when to use vs. transformers
- Autoencoders, VAEs, GANs
- Transformers: self-attention, multi-head attention, positional encoding

## Transformers (Deep Dive)
- Encoder-only (BERT), Decoder-only (GPT), Encoder-Decoder (T5)
- Attention variants: dot-product, cross-attention, causal masking
- KV Cache, GQA (Grouped Query Attention), MQA (Multi-Query Attention)
- RoPE, ALiBi positional embeddings
- Flash Attention, memory-efficient attention

## Training & Scaling
- Pre-training objectives (MLM, CLM, denoising)
- Scaling laws, Chinchilla optimal compute
- Distributed training: Data Parallel, Model Parallel, FSDP, DeepSpeed
- Mixed precision training (FP16, BF16, FP8)

## Common Interview Questions
- Explain how attention works. Why is it O(n²)?
- What is the difference between LayerNorm and BatchNorm?
- How would you debug a model that isn't converging?
- What scaling law governs LLM training?
- Compare encoder-only vs. decoder-only architectures and their use cases.
