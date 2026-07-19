# Vanilla RNN Architecture

## Overview

A Recurrent Neural Network (RNN) is a class of neural networks designed for sequential data. Unlike feedforward networks, RNNs maintain a **hidden state** that acts as a memory, capturing information about previous inputs in the sequence. This makes them suitable for time series, text, speech, and any data where order matters.

## The Hidden State Update

At each time step $t$, the RNN combines the current input $\mathbf{x}_t$ with the previous hidden state $\mathbf{h}_{t-1}$ to produce a new hidden state $\mathbf{h}_t$. The core update equation is:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
$$

Where:
- $\mathbf{h}_t \in \mathbb{R}^{d_h}$ — hidden state at time $t$
- $\mathbf{x}_t \in \mathbb{R}^{d_x}$ — input vector at time $t$
- $\mathbf{W}_{hh} \in \mathbb{R}^{d_h \times d_h}$ — recurrent weight matrix (hidden-to-hidden)
- $\mathbf{W}_{xh} \in \mathbb{R}^{d_h \times d_x}$ — input-to-hidden weight matrix
- $\mathbf{b}_h \in \mathbb{R}^{d_h}$ — bias vector
- $\tanh$ — hyperbolic tangent activation, squashing values to $(-1, 1)$

The same weight matrices $\mathbf{W}_{hh}$ and $\mathbf{W}_{xh}$ are **shared across all time steps**. This parameter sharing is what enables the RNN to generalize across sequence lengths.

## Forward Pass Through Time

Given an input sequence $(\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T)$, the forward pass proceeds as:

1. **Initialize** $\mathbf{h}_0 = \mathbf{0}$ (zero vector)
2. For $t = 1$ to $T$:
   $$
   \mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)
   $$
3. At each time step (or only at the final step), produce an output:
   $$
   \hat{\mathbf{y}}_t = \text{softmax}(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)
   $$

The hidden state $\mathbf{h}_t$ encodes information about the entire sequence up to time $t$. Information flows forward through time, with each state depending on all previous states.

## Loss Computation

For a sequence with ground truth labels $\mathbf{y}_1, \dots, \mathbf{y}_T$, the total loss is the sum of losses at each time step:

$$
\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t(\mathbf{y}_t, \hat{\mathbf{y}}_t)
$$

For classification tasks, $\mathcal{L}_t$ is typically cross-entropy loss:

$$
\mathcal{L}_t = -\sum_{c=1}^{C} y_{t,c} \log(\hat{y}_{t,c})
$$

For regression tasks (e.g., time series forecasting), mean squared error is used:

$$
\mathcal{L}_t = \|\mathbf{y}_t - \hat{\mathbf{y}}_t\|^2
$$

## Backpropagation Through Time (BPTT)

BPTT is the standard algorithm for training RNNs. It unrolls the computational graph through all time steps and applies backpropagation.

### Steps of BPTT

1. **Forward pass**: Compute and store all hidden states $\mathbf{h}_1, \dots, \mathbf{h}_T$ and outputs $\hat{\mathbf{y}}_1, \dots, \hat{\mathbf{y}}_T$.

2. **Loss computation**: Calculate $\mathcal{L} = \sum_t \mathcal{L}_t$.

3. **Backward pass**: For $t = T$ down to $1$, compute gradients:
   $$
   \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} + \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{t+1}} \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}
   $$
   
   The gradient at time $t$ depends on both the direct loss at $t$ and the gradient from the future.

4. **Parameter gradients**: Accumulate gradients across all time steps:
   $$
   \frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}}
   $$

### The Chain of Gradients

The gradient of $\mathcal{L}$ with respect to $\mathbf{h}_t$ involves a product of Jacobians across all future time steps:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \sum_{k=t}^{T} \frac{\partial \mathcal{L}_k}{\partial \mathbf{h}_k} \prod_{j=t+1}^{k} \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}
$$

Each Jacobian $\frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}$ involves the recurrent weight matrix $\mathbf{W}_{hh}$ multiplied by the derivative of $\tanh$:

$$
\frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \text{diag}(1 - \tanh^2(\mathbf{a}_j)) \, \mathbf{W}_{hh}
$$

## Vanishing Gradient Problem

The vanishing gradient problem is the fundamental limitation of vanilla RNNs. Examining the gradient product:

$$
\prod_{j=t+1}^{k} \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \prod_{j=t+1}^{k} \text{diag}(1 - \tanh^2(\mathbf{a}_j)) \, \mathbf{W}_{hh}
$$

### Causes

1. **$\tanh$ saturation**: The derivative $\tanh'(x) = 1 - \tanh^2(x)$ is at most $1.0$ and approaches $0$ for large $|x|$, shrinking gradients.

2. **Repeated multiplication**: The matrix $\mathbf{W}_{hh}$ is multiplied $k-t$ times. If its eigenvalues are less than $1$, the product vanishes exponentially with sequence length.

3. **Exploding gradients**: Conversely, if eigenvalues exceed $1$, gradients can explode (typically handled by gradient clipping).

### Consequences

- The network cannot learn long-range dependencies (e.g., a subject at the start of a sentence affecting a verb at the end).
- Hidden states at early time steps receive negligible gradient updates.
- Practically, vanilla RNNs struggle with sequences longer than 10–20 steps.

### Mitigation Strategies

- **Gradient clipping** (for exploding gradients): rescale gradients if their norm exceeds a threshold.
- **Weight initialization**: careful initialization (e.g., identity matrix for $\mathbf{W}_{hh}$) can help.
- **ReLU activation**: replacing $\tanh$ with ReLU helps but introduces unbounded activations.
- **LSTM / GRU**: gated architectures designed specifically to address the vanishing gradient problem.

## Summary

Vanilla RNNs provide the foundational framework for sequence modeling. The hidden state update and BPTT algorithm are the core concepts. However, the vanishing gradient problem severely limits their practical utility for long sequences, motivating the development of gated architectures like LSTMs and GRUs.
