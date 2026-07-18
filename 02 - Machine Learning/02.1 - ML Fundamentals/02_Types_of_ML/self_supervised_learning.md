# Self-Supervised Learning (Conceptual)

The model creates its own labels from the data itself. No human labels needed.

## Core Idea

Design a "pretext task" where part of the input is predicted from the rest. The model learns useful representations as a byproduct.

## Common Pretext Tasks

### Masking
- **NLP**: Mask 15% of words in a sentence, predict the missing ones (BERT)
- **Vision**: Mask patches of an image, reconstruct the missing patches (MAE)

### Contrastive Learning
- Pull similar (augmented) views together, push different views apart
- Example: Take one cat image → create two augmented versions (crop + rotate each). The model learns: "these are the same cat." On a different image (dog), it learns: "this is different."

## Why It Matters

Self-supervised pre-training has become the standard for large models:

1. **LLMs (GPT, Llama, BERT)**: Pretrained on raw internet text (trillions of words) with the "predict next word" pretext task. This builds rich language understanding before any fine-tuning.
2. **Computer vision (DINO, MAE, SimCLR)**: Pretrained on millions of unlabeled images. A doctor can then fine-tune on just 100 labeled X-rays instead of 10,000.
3. **Speech (wav2vec 2.0)**: Pretrained on raw audio (predict masked time segments), then fine-tuned on 1 hour of transcribed speech to achieve state-of-the-art results.

## Relationship to Other Learning Types

```
Supervised:    input → label (human provides labels)
Self-Supervised: input → pseudo-label (model creates labels from input structure)
Semi-Supervised: input → mix of human labels + pseudo-labels
Unsupervised:  input → patterns (no explicit prediction task)
```
