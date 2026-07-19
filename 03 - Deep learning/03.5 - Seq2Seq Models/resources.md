# Resources: Seq2Seq Models & Attention

## Foundational Papers

### Seq2Seq & Encoder-Decoder

1. **Sequence to Sequence Learning with Neural Networks** — Sutskever, Vinyals, Le (2014)
   [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
   *The original Seq2Seq paper using LSTM encoders and decoders on machine translation.*

2. **Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation** — Cho et al. (2014)
   [arXiv:1406.1078](https://arxiv.org/abs/1406.1078)
   *Introduced the GRU and the encoder-decoder framework for translation.*

### Attention Mechanisms

3. **Neural Machine Translation by Jointly Learning to Align and Translate** — Bahdanau, Cho, Bengio (2015)
   [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
   *The first attention mechanism for Seq2Seq. Introduced additive attention.*

4. **Effective Approaches to Attention-based Neural Machine Translation** — Luong, Pham, Manning (2015)
   [arXiv:1508.04025](https://arxiv.org/abs/1508.04025)
   *Introduced multiplicative (dot-product/general) attention and local/global variants.*

### Transformer

5. **Attention Is All You Need** — Vaswani et al. (2017)
   [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
   *The Transformer paper. Scaled dot-product attention, multi-head attention, positional encoding. The single most important paper in modern NLP.*

### Positional Encoding

6. **Self-Attention with Relative Position Representations** — Shaw, Uszkoreit, Vaswani (2018)
   [arXiv:1803.02155](https://arxiv.org/abs/1803.02155)
   *Extends Transformer with relative position representations instead of absolute.*

### Pre-trained Models

7. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** — Devlin et al. (2019)
   [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
   *Encoder-only Transformer pre-trained with MLM and NSP.*

8. **Improving Language Understanding by Generative Pre-Training** — Radford et al. (2018) (GPT-1)
   [OpenAI](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
   *Decoder-only Transformer pre-trained with autoregressive LM.*

9. **Language Models are Few-Shot Learners** — Brown et al. (2020) (GPT-3)
   [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
   *Scaling decoder-only Transformers to 175B parameters and demonstrating in-context learning.*

10. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** — Raffel et al. (2020) (T5)
    [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
    *Full encoder-decoder Transformer treating all NLP tasks as text-to-text.*

## Online Courses & Lectures

| Resource | Link | Coverage |
|---|---|---|
| Stanford CS224n (NLP with Deep Learning) | [web.stanford.edu/class/cs224n](https://web.stanford.edu/class/cs224n/) | Seq2Seq, attention, Transformers |
| Stanford CS231n (CNNs for CV) — attention section | [cs231n.stanford.edu](https://cs231n.stanford.edu/) | Attention in vision contexts |
| DeepLearning.AI — Sequence Models | [coursera.org](https://www.coursera.org/specializations/deep-learning) | Seq2Seq, attention, LSTMs |
| MIT 6.S191 — Transformers | [youtube.com](https://youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) | Modern deep learning including Transformers |
| Hugging Face Course | [huggingface.co/course](https://huggingface.co/course) | Practical Transformers with HF library |

## Blog Posts & Visual Explanations

| Resource | Link |
|---|---|
| Jay Alammar — The Illustrated Transformer | [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/) |
| Jay Alammar — The Illustrated BERT, ELMo, etc. | [jalammar.github.io/illustrated-bert](https://jalammar.github.io/illustrated-bert/) |
| Jay Alammar — Visualizing A Neural Machine Translation Model | [jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) |
| Lilian Weng — Attention? Attention! | [lilianweng.github.io](https://lilianweng.github.io/posts/2018-06-24-attention/) |
| Peter Bloem — Transformers from Scratch | [peterbloem.nl/blog/transformers](https://peterbloem.nl/blog/transformers) |
| The Annotated Transformer (Harvard NLP) | [nlp.seas.harvard.edu/2018/04/03/attention.html](https://nlp.seas.harvard.edu/2018/04/03/attention.html) |
| Andrej Karpathy — Let's build GPT from scratch | [youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| Transformers from Scratch (blog series) | [e2eml.school/transformers.html](https://e2eml.school/transformers.html) |

## Implementation Repositories

| Repository | Description |
|---|---|
| [tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor) | Original Transformer implementation from Google |
| [huggingface/transformers](https://github.com/huggingface/transformers) | State-of-the-art pre-trained models (BERT, GPT, T5, etc.) |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | Minimal, clean GPT implementation |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) | Line-by-line PyTorch implementation with explanations |
| [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) | Implementations of Transformer, GPT, BERT with explanations |

## Reference Books

- **Speech and Language Processing (3rd ed.)** — Jurafsky & Martin
  [web.stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3/)
  *Chapters on NMT, attention, and Transformers.*

- **Dive into Deep Learning** — Zhang et al.
  [d2l.ai](https://d2l.ai/)
  *Chapters on Seq2Seq, attention, and Transformers with code.*

## Videos

| Title | Link |
|---|---|
| "Attention Is All You Need" (Yannic Kilcher) | [youtube.com/watch?v=iDulhoQ2pro](https://www.youtube.com/watch?v=iDulhoQ2pro) |
| Transformer explainer (3Blue1Brown) | [youtube.com/watch?v=wjZofJX0v4M](https://www.youtube.com/watch?v=wjZofJX0v4M) |
| Seq2Seq + Attention (Aladdin Persson) | [youtube.com/watch?v=EoGUlvhRYpk](https://www.youtube.com/watch?v=EoGUlvhRYpk) |
