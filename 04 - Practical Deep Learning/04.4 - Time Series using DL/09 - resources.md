# Resources — Time Series using Deep Learning

## Key Research Papers

### Foundational

| Paper | Year | Contribution |
|---|---|---|
| **Long Short-Term Memory** (Hochreiter & Schmidhuber) | 1997 | LSTM cell — solved vanishing gradient for sequential data |
| **Learning Phrase Representations using RNN Encoder-Decoder** (Cho et al.) | 2014 | Seq2Seq framework with gated recurrent units |
| **Sequence to Sequence Learning with Neural Networks** (Sutskever et al.) | 2014 | LSTM-based seq2seq, teacher forcing |

### Convolutional & WaveNet

| Paper | Year | Contribution |
|---|---|---|
| **WaveNet: A Generative Model for Raw Audio** (Oord et al., DeepMind) | 2016 | Dilated causal convolutions + gated activations + skip connections |
| **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling** (Bai et al.) | 2018 | TCN — showed CNNs can match/beat RNNs on many sequence tasks |
| **Temporal Convolutional Networks for Sequence Modeling** | — | Practical guide to causal convolutions, dilations, residual blocks |

### Attention & Transformers

| Paper | Year | Contribution |
|---|---|---|
| **Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting** (Lim et al., Google) | 2019 | TFT — VSN + LSTM + attention + quantile outputs |
| **Attention Is All You Need** (Vaswani et al.) | 2017 | Transformer architecture — self-attention for sequences |
| **Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting** (Zhou et al.) | 2021 | ProbSparse attention for efficient long-sequence forecasting |
| **Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting** (Wu et al.) | 2021 | Replaces self-attention with auto-correlation for seasonal data |

### Anomaly Detection

| Paper | Year | Contribution |
|---|---|---|
| **LSTM-Based Encoder-Decoder for Multi-Sensor Anomaly Detection** (Malhotra et al.) | 2016 | LSTM-autoencoder for anomaly detection in time series |
| **Deep Learning for Anomaly Detection: A Survey** (Chalapathy & Chawla) | 2019 | Comprehensive survey of deep anomaly detection methods |

## Books

| Title | Author(s) | Focus |
|---|---|---|
| **Deep Learning** (Chapter 10) | Goodfellow, Bengio, Courville | RNNs, LSTMs, sequence modeling foundations |
| **Forecasting: Principles and Practice** | Hyndman & Athanasopoulos | Classical + practical forecasting (R, but concepts transfer) |
| **Hands-On Time Series Analysis with Python** | B. Vishwas & A. Patel | Practical Python implementations (ARIMA to DL) |
| **Python for Time Series** | Megan Risdal | Applied time series with pandas & scikit-learn |

## Libraries

| Library | Description | Installation |
|---|---|---|
| **PyTorch** | Core DL framework for all architectures | `pip install torch` |
| **PyTorch Forecasting** | High-level time series models (TFT, NBeats, DeepAR) built on PyTorch Lightning | `pip install pytorch-forecasting` |
| **Darts** | Time series library by Unit8 — unified API for ARIMA to Transformer | `pip install darts` |
| **GluonTS** | AWS's time series toolkit — many probabilistic models | `pip install gluonts` |
| **sktime** | Scikit-learn compatible time series toolkit | `pip install sktime` |
| **tsai** | State-of-the-art deep learning for time series (TCN, InceptionTime, etc.) | `pip install tsai` |

### Library Quick Comparison

| Library | Best For | Model Coverage |
|---|---|---|
| **PyTorch Forecasting** | TFT, NBeats, DeepAR (production) | 5–10 advanced DL models |
| **Darts** | Comparing classical + DL models quickly | 15+ models (ARIMA to Transformer) |
| **GluonTS** | Probabilistic forecasting research | 20+ probabilistic models |
| **tsai** | State-of-the-art TS classification & regression | 30+ architectures |

## Blogs & Tutorials

| Title | Link / Author | Content |
|---|---|---|
| **Time Series Forecasting with LSTMs in PyTorch** | PyTorch.org tutorials | Multi-step forecasting walkthrough |
| **The WaveNet Trick** | Kevin Alex Zhang | Explained causal dilated convolutions |
| **Temporal Fusion Transformer Explained** | Towards Data Science | Deep dive into TFT architecture |
| **Anomaly Detection with LSTM Autoencoders** | Paperspace Blog | Step-by-step implementation |

## Online Courses

| Course | Provider | Coverage |
|---|---|---|
| **Sequence Models** | deeplearning.ai (Coursera) | RNNs, LSTMs, attention, seq2seq |
| **Machine Learning for Time Series** | Kaggle Learn | Feature engineering, forecasting with trees/DL |
| **Time Series Analysis** | Stanford (STATS 202) | Classical foundations with some DL |

## Conferences & Venues

- **ICLR** — Advances in time series Transformers (Autoformer, FEDformer)
- **NeurIPS** — Time series competitions, anomaly detection benchmarks
- **KDD** — Applied time series papers, TFT original publication
- **IEEE ICASSP** — Signal processing, WaveNet variants

## Datasets for Practice

| Dataset | Description | Task |
|---|---|---|
| **Air Passengers** | Monthly airline passengers (1949–1960) | Univariate forecasting |
| **Electricity Transformer Temperature (ETT)** | 2 years of transformer data | Long-sequence forecasting |
| **M4 Competition** | 100,000 time series from various domains | Benchmark for forecasting methods |
| **Numenta Anomaly Benchmark (NAB)** | Labeled anomaly time series (server metrics, social media) | Anomaly detection |
| **Web Traffic** | Wikipedia page views | Multivariate forecasting |
| **Monash Time Series Archive** | Large repository of diverse time series | Benchmarking DL models |
