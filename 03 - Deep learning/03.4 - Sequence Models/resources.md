# Resources: Sequence Models

## Foundational Papers

| Paper | Authors | Year | Notes |
|-------|---------|------|-------|
| [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) | Hochreiter & Schmidhuber | 1997 | Original LSTM paper — introduces the cell state and gating mechanism |
| [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) | Cho et al. | 2014 | Introduces the GRU |
| [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555) | Chung et al. | 2014 | Systematic comparison of LSTM vs GRU |
| [Bidirectional Recurrent Neural Networks](https://www.cs.toronto.edu/~hinton/absps/brnn.pdf) | Schuster & Paliwal | 1997 | Introduces bidirectional RNNs |
| [The Vanishing Gradient Problem During Learning Recurrent Neural Nets and Problem Solutions](https://www.researchgate.net/publication/221618312_The_Vanishing_Gradient_Problem_During_Learning_Recurrent_Neural_Nets_and_Problem_Solutions) | Hochreiter et al. | 1998 | Foundational analysis of the vanishing gradient problem |
| [A Simple Way to Initialize Recurrent Networks of Rectified Linear Units](https://arxiv.org/abs/1504.00941) | Le et al. | 2015 | Identity initialization for RNNs with ReLU |
| [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329) | Zaremba et al. | 2014 | Dropout for LSTMs |
| [Zoneout: Regularizing RNNs by Randomly Preserving Hidden Activations](https://arxiv.org/abs/1606.01305) | Krueger et al. | 2016 | Alternative regularization for RNNs |

## Courses and Lectures

| Resource | Provider | Notes |
|----------|----------|-------|
| [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/) | Stanford | Lecture 9 covers RNNs, LSTMs, and visual recognition applications |
| [CS224n: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) | Stanford | Deep coverage of RNNs/LSTMs for NLP; highly recommended |
| [fast.ai Practical Deep Learning](https://course.fast.ai/) | fast.ai | Practical RNN/LSTM implementation lessons |
| [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) | Coursera (Andrew Ng) | Course 5 covers sequence models in depth |
| [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/) | MIT | Lecture on sequence models with modern applications |

## Blog Posts and Tutorials

| Resource | Author | Notes |
|----------|--------|-------|
| [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | Christopher Olah | The definitive illustrated guide to LSTMs |
| [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) | Andrej Karpathy | Character-level RNNs; great intuition and examples |
| [WildML — RNN Tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) | Denny Britz | Multi-part RNN tutorial with code |
| [Understanding GRU Networks](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be) | Simeon Kostadinov | Visual GRU explanation |
| [RNN from Scratch in NumPy](https://towardsdatascience.com/rnn-from-scratch-in-numpy-91e9a99a6a1a) | Towards Data Science | NumPy-only RNN implementation |
| [LSTM Forward and Backward Pass](https://arunmallya.github.io/writeups/nn/lstm/index.html) | Arun Mallya | Detailed LSTM gradient derivations |

## Video Resources

| Video | Speaker | Notes |
|-------|---------|-------|
| [LSTMs Explained Simply](https://www.youtube.com/watch?v=8HyCNIVRbSU) | StatQuest | Beginner-friendly walkthrough |
| [Illustrated Guide to LSTMs and GRUs](https://www.youtube.com/watch?v=ICKg7TqO-vA) | Machine Learning Mastery | Visual step-by-step explanation |

## Software Implementations

| Library | Notes |
|---------|-------|
| [PyTorch RNN Module](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) | Production-grade RNN/LSTM/GRU implementations |
| [TensorFlow RNN API](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN) | Similar coverage with Keras interface |
| [JAX RNN](https://jax.readthedocs.io/) | Functional approach; good for research |

## Related Topics

- **Attention and Transformers**: The successor to RNNs for most NLP tasks. Key paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017).
- **CTC (Connectionist Temporal Classification)**: Used with RNNs for sequence-to-sequence alignment (speech recognition).
- **Seq2Seq with Attention**: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al., 2014).
