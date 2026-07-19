# 01 — Perceptron & Multi-Layer Perceptron (MLP)

## 1. Biological Neuron vs. Artificial Neuron

The artificial neuron is a crude mathematical abstraction of its biological counterpart.

| Biological Neuron | Artificial Neuron |
|---|---|
| Dendrites receive signals from other neurons | Input features $x_1, x_2, \dots, x_n$ |
| Cell body sums the incoming signals | Weighted sum $z = \sum w_i x_i + b$ |
| Axon transmits the output if threshold is exceeded | Activation function $a = g(z)$ |
| Synapse adjusts connection strength | Weight $w_i$ learned during training |

A biological neuron fires (sends an electrical impulse) when its membrane potential exceeds a threshold. An artificial neuron computes a weighted sum of its inputs, adds a bias, and passes the result through a non-linear activation function:

$$y = g\left( \sum_{i=1}^{n} w_i x_i + b \right)$$

## 2. The Perceptron Algorithm

The Perceptron, introduced by Frank Rosenblatt in 1958, is the simplest type of artificial neuron. It uses a **step function** as its activation:

$$g(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

The algorithm proceeds as follows:

1. Initialize weights $w_i$ and bias $b$ to small random values.
2. For each training sample $(\mathbf{x}^{(i)}, y^{(i)})$:
   - Compute the output $\hat{y}^{(i)} = g(\mathbf{w}^\top \mathbf{x}^{(i)} + b)$
   - Update weights: $w_j \leftarrow w_j + \eta \cdot (y^{(i)} - \hat{y}^{(i)}) \cdot x_j$
   - Update bias: $b \leftarrow b + \eta \cdot (y^{(i)} - \hat{y}^{(i)})$

where $\eta$ is the learning rate. The Perceptron **converges** in a finite number of steps if the data is **linearly separable** (Perceptron Convergence Theorem).

## 3. Limitations of Single-Layer Perceptrons

The single-layer Perceptron can only learn **linearly separable** decision boundaries. This is a severe limitation:

- It **cannot** learn the XOR function (as famously shown by Minsky & Papert, 1969).
- It fails on any problem where classes cannot be separated by a hyperplane.
- It can only represent **linear** functions of the input.

For a single-layer Perceptron with $n$ inputs, the decision boundary is the hyperplane defined by $\mathbf{w}^\top \mathbf{x} + b = 0$, which is always a linear surface.

## 4. Multi-Layer Perceptron (MLP)

An MLP stacks multiple layers of neurons, with **non-linear activation functions** between them:

- **Input layer**: receives the raw features.
- **Hidden layers**: one or more layers that learn intermediate representations.
- **Output layer**: produces the final prediction.

Each layer $l$ computes:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$

where $\mathbf{a}^{[0]} = \mathbf{x}$ is the input. With non-linear activations (e.g., ReLU, tanh) and at least one hidden layer, an MLP can approximate **any** continuous function — this is the Universal Approximation Theorem.

### 4.1 Why Depth Matters

- **Shallow** networks (1 hidden layer) are universal approximators but may require exponentially many neurons.
- **Deep** networks (many hidden layers) can represent the same function with far fewer neurons by learning hierarchical features.
- Each layer builds progressively more abstract representations: edges → shapes → objects in vision, or characters → words → sentences in NLP.

## 5. Universal Approximation Theorem

> **Theorem (Cybenko, 1989; Hornik, 1991):** A feedforward network with a single hidden layer containing a finite number of neurons and a non-linear activation function (e.g., sigmoid) can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to any desired degree of accuracy, provided the hidden layer has enough neurons.

**Key implications:**
- The theorem guarantees **existence** of a network that can approximate any function — but it does **not** guarantee we can find it (optimization may get stuck).
- It does **not** tell us how many neurons are needed — this grows exponentially for some functions.
- Modern deep learning succeeds not just because of the theorem, but because deep architectures generalize better and are easier to optimize in practice.

**Limitations of the theorem:**
- It applies to **continuous** functions only.
- It requires **enough** hidden units — which may be impractically large.
- It says nothing about **learnability** from finite data.

## 6. Training an MLP: Key Components

Training an MLP requires three ingredients:

1. **Forward propagation**: Compute predictions by passing data through the network.
2. **Loss function**: Measure how wrong the predictions are ($\mathcal{L}(y, \hat{y})$).
3. **Backpropagation**: Compute gradients of the loss w.r.t. every weight using the chain rule.
4. **Gradient descent**: Update weights to minimize the loss.

These topics are covered in detail in the following sections (Activation Functions, Forward & Backward Propagation, Loss Functions).

## Summary

| Concept | Key Takeaway |
|---|---|
| Perceptron | Single linear decision boundary; converges for linearly separable data |
| MLP | Multiple layers with non-linear activations; can learn non-linear functions |
| Universal Approximation | One hidden layer can approximate any continuous function given enough neurons |
| Depth | Deep networks learn hierarchical features more efficiently than wide shallow networks |

---

**Next:** [02 - Activation Functions.md](./02%20-%20Activation%20Functions.md)
