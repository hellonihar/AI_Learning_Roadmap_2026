# RNN Applications

## Text Classification

RNNs classify text by processing a sequence of word or character embeddings and using the final hidden state (or a pooled representation of all states) for classification:

$$
\mathbf{h}_T = \text{RNN}(\mathbf{x}_1, \dots, \mathbf{x}_T)
$$
$$
\hat{y} = \text{softmax}(\mathbf{W}_{hy} \mathbf{h}_T + \mathbf{b}_y)
$$

- **Architecture**: Typically a bidirectional LSTM or GRU, often stacked 2–3 layers deep.
- **Pooling**: Max-pooling or mean-pooling across all time steps often works better than the final state alone.
- **Tasks**: Topic classification, spam detection, news categorization.

## Sentiment Analysis

A specialized text classification task where the model predicts sentiment polarity (positive, negative, neutral) from a review or social media post.

- **Preprocessing**: Tokenization, lowercasing, handling negation ("not good" vs "good").
- **Word embeddings**: Pretrained embeddings (GloVe, Word2Vec) or learned character-level embeddings.
- **Aspect-based sentiment**: Identify sentiment toward specific entities (e.g., "battery life is great but the screen is dim").

## Language Modeling

Language modeling predicts the next token given previous tokens: $P(\mathbf{x}_t \mid \mathbf{x}_1, \dots, \mathbf{x}_{t-1})$.

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(\mathbf{x}_t \mid \mathbf{x}_1, \dots, \mathbf{x}_{t-1})
$$

- **Word-level LM**: Models probability of word sequences. Vocabulary size typically 10k–100k.
- **Character-level LM**: Models character sequences. Smaller vocabulary but longer sequences.
- **Evaluation**: Perplexity ($\text{PPL} = \exp(\mathcal{L} / T)$), cross-entropy loss.
- **Usage**: Foundation for text generation, machine translation, speech recognition.

## Character-Level RNNs

Character-level RNNs operate on individual characters rather than words. Each character is one-hot encoded or embedded, and the RNN predicts the next character.

### Advantages

- **No tokenization needed**: Works with any language or script.
- **Handles OOV (Out of Vocabulary)**: No unknown word problem.
- **Learns morphology**: Captures spelling patterns, prefixes, suffixes.

### Disadvantages

- **Longer sequences**: A sentence of 10 words becomes ~50 characters.
- **Harder to capture long-range semantics**: More time steps to propagate through.
- **Slower inference**: More steps to generate text.

### Notable Implementation

Karpathy's "The Unreasonable Effectiveness of Recurrent Neural Networks" (2015) demonstrated character-level RNNs generating Shakespeare, Linux source code, and Wikipedia articles.

## Time Series Forecasting

RNNs predict future values of a time series from historical observations.

$$
\hat{y}_{T+1} = \text{RNN}(\mathbf{x}_1, \dots, \mathbf{x}_T)
$$

### Common Architectures

- **RNN-based**: LSTM or GRU encoders for multi-step forecasting.
- **Seq2Seq**: Encoder-decoder with RNNs for multi-horizon prediction.
- **Attention**: Adding attention over the input sequence improves long-range forecasting.

### Applications

- **Finance**: Stock price prediction, volatility forecasting.
- **Energy**: Electricity load forecasting, wind power prediction.
- **Weather**: Temperature, precipitation, air quality prediction.
- **IoT**: Sensor data anomaly detection and prediction.

### Practical Considerations

- **Normalization**: Scale inputs to $[-1, 1]$ or $[0, 1]$; many series need differencing for stationarity.
- **Lookback window**: The number of past time steps used as input is a critical hyperparameter.
- **Multi-step strategies**: Direct (predict each step with separate output), iterative (feed prediction as next input), or seq2seq.

## Music Generation

RNNs model music as a sequence of notes, chords, or raw audio frames.

### Representations

- **Midi events**: Note-on, note-off, velocity, time delta.
- **Piano roll**: Binary matrix of time steps × pitches.
- **Audio waveforms**: Raw samples (needs very deep models).

### Architectures

- **LSTM**: Processing sequences of musical events; state-of-the-art before Transformers.
- **Character-level analog**: Treat notes like "characters" in a musical language.
- **Hierarchical RNNs**: One RNN for bar-level structure, another for note-level details.

### Notable Projects

- **Magenta (Google Brain)**: Open-source music generation with LSTM and Transformer models.
- **DeepBach**: Bach chorale generation using an LSTM-based model.

## Summary

RNNs are versatile across domains. Text classification and sentiment analysis use the full hidden state sequence for prediction. Language modeling and character-level RNNs are fundamental to natural language generation. Time series forecasting applies RNNs to numerical sequences with practical considerations around normalization and lookback. Music generation extends similar ideas to creative domains. In all cases, LSTMs and GRUs are the architectures of choice due to their ability to handle longer dependencies.
