# Resources — Foundations of Neural Networks

---

## Books

| Title | Author(s) | Notes |
|---|---|---|
| **Deep Learning** | Goodfellow, Bengio, Courville | Chapters 6 (Feedforward Networks), 7 (Regularization), 8 (Optimization) are essential. The definitive textbook. |
| **Neural Networks and Deep Learning** | Michael Nielsen | Free online book. Excellent intuitive explanations of backpropagation. |
| **Pattern Recognition and Machine Learning** | Christopher Bishop | Chapter 5 (Neural Networks) — rigorous mathematical treatment. |
| **The Elements of Statistical Learning** | Hastie, Tibshirani, Friedman | Chapters 10–11. Less focus on deep learning but excellent statistical foundations. |

---

## Online Courses

| Course | Platform | Instructor | Notes |
|---|---|---|---|
| **CS231n: CNNs for Visual Recognition** | Stanford / YouTube | Andrej Karpathy, Fei-Fei Li, Justin Johnson | Lectures 3–5 cover backprop, activation functions, and neural network basics. The best public lecture series on neural nets. |
| **CS229: Machine Learning** | Stanford / YouTube | Andrew Ng | Lecture 9 (Neural Networks) — classic introduction. |
| **fast.ai Practical Deep Learning** | fast.ai | Jeremy Howard, Rachel Thomas | Top-down approach: code first, theory as needed. Lesson 1–4 cover building neural nets from scratch. |
| **Deep Learning Specialization** | Coursera | Andrew Ng | Course 1 (Neural Networks & Deep Learning) covers foundations thoroughly. |

---

## Key Papers

| Paper | Year | Authors | Contribution |
|---|---|---|---|
| **A logical calculus of the ideas immanent in nervous activity** | 1943 | McCulloch & Pitts | First mathematical model of a neuron |
| **The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain** | 1958 | Frank Rosenblatt | The Perceptron algorithm |
| **Perceptrons** | 1969 | Minsky & Papert | Proved limitations of single-layer Perceptrons (XOR problem) |
| **Learning representations by back-propagating errors** | 1986 | Rumelhart, Hinton, Williams | Popularized backpropagation |
| **Approximation capabilities of multilayer feedforward networks** | 1991 | Hornik | Universal Approximation Theorem |
| **Rectified Linear Units Improve Restricted Boltzmann Machines** | 2010 | Nair & Hinton | Introduced ReLU for deep learning |
| **ImageNet Classification with Deep Convolutional Neural Networks** | 2012 | Krizhevsky, Sutskever, Hinton | AlexNet — breakthrough result with ReLU |
| **Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet** | 2015 | He et al. | He initialization, PReLU |
| **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** | 2015 | Ioffe & Szegedy | Batch normalization |
| **Deep Residual Learning for Image Recognition** | 2016 | He et al. | Residual connections (ResNet) |
| **Swish: A Self-Gated Activation Function** | 2017 | Ramachandran, Zoph, Le | Swish activation |
| **Gaussian Error Linear Units (GELUs)** | 2016 | Hendrycks & Gimpel | GELU activation |

---

## Interactive Visualizations

| Resource | Link | What It Shows |
|---|---|---|
| **TensorFlow Playground** | [playground.tensorflow.org](https://playground.tensorflow.org) | Interactive neural network training with different activations, architectures, and datasets |
| **3Blue1Brown — Neural Networks** | YouTube | Stunning visual explanations of backpropagation and gradient descent |
| **Neural Network Zoo** | asimovinstitute.org | Visual guide to neural network architectures |

---

## Recommended Study Path

1. **Start with CS231n lectures 3–5** for the best visual/intuitive understanding of backpropagation.
2. **Read Goodfellow Ch. 6** (Deep Feedforward Networks) for the rigorous mathematical treatment.
3. **Implement from scratch** — complete the `code_walkthrough.md` exercises and build your own NumPy network.
4. **Experiment on TensorFlow Playground** — build intuition for hyperparameters.
5. **Read the classic papers** in chronological order (1969 → 1986 → 2012 → 2015) to understand the historical progression.
