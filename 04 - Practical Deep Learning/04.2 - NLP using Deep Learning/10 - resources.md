# Resources

## Foundational Papers

| Paper | Year | Key Idea |
|-------|------|----------|
| [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (Word2Vec) | 2013 | CBOW and Skip-gram embeddings |
| [GloVe: Global Vectors for Word Representation](https://arxiv.org/abs/1503.01132) | 2014 | Matrix factorization of co-occurrence counts |
| [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) (FastText) | 2016 | Subword n-gram embeddings |
| [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | 2014 | Attention mechanism for seq2seq |
| [Attention is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Transformer architecture |
| [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) | 2019 | Masked language model pre-training |

## Sequence Labeling

| Paper | Year | Key Idea |
|-------|------|----------|
| [Bidirectional LSTM-CRF for Sequence Tagging](https://arxiv.org/abs/1508.01991) | 2015 | BiLSTM-CRF architecture |
| [End-to-end Sequence Labeling via Bi-LSTM-CRF-CNN](https://arxiv.org/abs/1603.01354) | 2016 | CNN for character-level features |
| [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) | 2016 | Comprehensive survey of NER models |

## Text Generation

| Paper | Year | Key Idea |
|-------|------|----------|
| [Generating Text with Recurrent Neural Networks](https://arxiv.org/abs/1308.0850) | 2013 | Character-level LSTM LM |
| [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | 2019 | Top-p (nucleus) sampling |
| [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833) | 2018 | Story generation with transformers |

## Parameter-Efficient Fine-Tuning

| Paper | Year | Key Idea |
|-------|------|----------|
| [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) | 2021 | Low-rank weight updates |
| [Prefix-Tuning](https://arxiv.org/abs/2101.00190) | 2021 | Learnable prefix tokens |

## Online Courses & Tutorials

- **[Stanford CS224n: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)** — The definitive NLP course. Lecture videos, assignments, and readings cover everything from word vectors to transformers. The assignments include building a Neural Machine Translation system and a QA system.

- **[d2l.ai — Dive into Deep Learning (NLP Chapter)](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html)** — Interactive book with PyTorch code. Covers word embeddings, BERT pre-training, fine-tuning, and machine translation. Every section has executable code.

- **[HuggingFace NLP Course](https://huggingface.co/learn/nlp-course)** — Hands-on course teaching transformers pipeline, tokenizers, fine-tuning, and sharing models. Covers all major tasks: classification, NER, QA, summarization, translation.

- **[HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/index)** — Comprehensive API docs with task-specific tutorials, model cards, and examples.

- **[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)** — Line-by-line implementation of the Transformer in PyTorch with Harvard's annotated explanation.

- **[PyTorch NLP Tutorials](https://pytorch.org/tutorials/beginner/nlp/)** — Official PyTorch NLP examples: sequence models, word embeddings, text classification.

- **[TensorFlow Neural Machine Translation Tutorial](https://www.tensorflow.org/text/tutorials/transformer)** — Transformer implementation for machine translation.

## Tools & Libraries

- **HuggingFace Transformers** — Primary library for pre-trained models.
- **HuggingFace Tokenizers** — Fast BPE/WordPiece/SentencePiece tokenization in Rust.
- **torchtext** — Data processing utilities (tokenizers, vocab, datasets).
- **spaCy** — Industrial-strength NLP with pre-trained pipelines for tokenization, NER, POS.
- **SentencePiece** — Language-independent subword tokenizer.
- **Gensim** — Topic modeling and word2vec/fastText training.
- **seqeval** — Sequence labeling evaluation (span-based F1).

## Model Hubs

- **[HuggingFace Model Hub](https://huggingface.co/models)** — 500k+ models sorted by task, framework, language.
- **[TensorFlow Hub](https://tfhub.dev/)** — Pre-trained TF models, including BERT and Electra.

## Books

- **Speech and Language Processing (Jurafsky & Martin)** — Chapters 6-10 cover neural networks, embeddings, and sequence labeling. Free online: https://web.stanford.edu/~jurafsky/slp3/
- **NLP with Transformers (Tunstall et al.)** — Practical guide to using HuggingFace transformers for all major tasks. O'Reilly 2022.
