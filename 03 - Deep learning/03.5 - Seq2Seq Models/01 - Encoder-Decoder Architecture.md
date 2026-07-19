# Encoder-Decoder Architecture

## Overview

The encoder-decoder architecture, also known as sequence-to-sequence (Seq2Seq), is a neural network design that maps a variable-length input sequence to a variable-length output sequence. Proposed by Sutskever et al. (2014) and Cho et al. (2014), it became the foundation of modern neural machine translation, text summarization, speech recognition, and conversational AI.

## Architecture

The architecture consists of two recurrent neural networks (RNNs) connected by a fixed-dimensional context vector.

### Encoder

The encoder reads the input sequence token by token and compresses its information into a hidden state. Given an input sequence $\mathbf{x} = (x_1, x_2, \dots, x_T)$, the encoder computes:

$$h_t = f(x_t, h_{t-1})$$

where $f$ is an RNN cell (typically an LSTM or GRU), $h_t$ is the hidden state at time step $t$, and $h_0$ is usually initialized to zero. After processing the entire sequence, the final hidden state $h_T$ (or a combination of all hidden states) serves as the **context vector** $c$ — a fixed-length summary of the entire input.

In practice, the context vector is often taken as the final encoder hidden state: $c = h_T$.

### Decoder

The decoder is another RNN that generates the output sequence $\mathbf{y} = (y_1, y_2, \dots, y_{T'})$ one element at a time, conditioned on the context vector $c$ and all previously generated tokens. The decoder hidden state at time $t$ is:

$$s_t = g(y_{t-1}, s_{t-1}, c)$$

where $s_0$ is typically initialized with the context vector (e.g., $s_0 = \tanh(W_s c)$). The output probability distribution over the vocabulary at step $t$ is:

$$P(y_t \mid y_{<t}, \mathbf{x}) = \text{softmax}(W_o s_t + b_o)$$

The decoder continues generating tokens until it produces an end-of-sequence (`<EOS>`) token or reaches a maximum length.

## Teacher Forcing

During training, the decoder is fed the **ground truth** previous token rather than its own prediction. This technique, called **teacher forcing**, stabilizes and accelerates training by preventing the model from drifting into states it has not been trained on.

Without teacher forcing, an early mistake in the decoder would cascade, producing unrealistic sequences that the model has never seen during training. With teacher forcing, the decoder always conditions on correct history, allowing gradients to flow more effectively.

However, teacher forcing creates a mismatch between training and inference (exposure bias). At inference time, the decoder must rely on its own predictions. Scheduled sampling — gradually mixing ground truth and predicted tokens during training — is a common mitigation.

## Bidirectional Encoder

A powerful extension is the **bidirectional encoder** (BiRNN). Two separate RNNs process the input in forward and backward directions:

$$\overrightarrow{h}_t = \text{RNN}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1})$$
$$\overleftarrow{h}_t = \text{RNN}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$$

The hidden state at each step is the concatenation: $h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$. This gives the encoder access to both past and future context for each input token, significantly improving representation quality.

## Applications

### Machine Translation
The canonical application. The encoder reads the source sentence (e.g., French: "Je suis étudiant"), and the decoder generates the target sentence (e.g., English: "I am a student"). Seq2Seq models dominated translation from 2014–2017 before being superseded by Transformers.

### Text Summarization
The input is a long document, and the output is a concise summary. Because the output is typically much shorter than the input, the information bottleneck imposed by the fixed context vector is particularly problematic, motivating the attention mechanism.

### Chatbots / Dialogue Systems
The encoder processes the dialogue history or user utterance, and the decoder generates a response. Conversational Seq2Seq models often suffer from generic responses ("I don't know") due to the bottleneck and lack of diversity in the decoder.

### Image Captioning
A CNN serves as the encoder (producing a single vector or spatial feature map), and an RNN decoder generates a textual description of the image.

## Limitations

1. **Information Bottleneck**: The fixed-length context vector must encode the entire input sequence, regardless of its length. Performance degrades sharply for long sequences.
2. **Vanishing Gradients**: Despite LSTM/GRU mitigating this somewhat, very long sequences still pose challenges for gradient propagation.
3. **Sequential Computation**: Both encoder and decoder process tokens one at a time, preventing parallelization during training and inference.
4. **Exposure Bias**: The mismatch between teacher-forced training and autoregressive inference degrades generation quality.

These limitations motivated the development of the **attention mechanism** and, later, the **Transformer architecture**, which address the bottleneck and parallelization issues respectively.

## Summary

The encoder-decoder architecture established the paradigm of mapping variable-length sequences through a neural bottleneck. While largely replaced by Transformers in production systems, it remains essential for understanding how neural sequence modeling evolved, and the principles of conditioning generation on encoded input carry directly into modern architectures.
