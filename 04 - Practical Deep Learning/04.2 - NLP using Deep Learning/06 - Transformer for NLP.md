# Transformers for NLP

## Why Transformers?

RNNs process tokens sequentially: O(T) steps for a T-length sequence, cannot parallelize, and struggle with very long-range dependencies (despite LSTMs). The **Transformer** (Vaswani et al., 2017) replaces recurrence with **self-attention**:

- All tokens attend to all tokens in O(1) parallel steps.
- Effective sequence length is not limited by gradient propagation.
- The quadratic O(T²) cost in sequence length is manageable with optimized implementations.

## Architecture at a Glance

A Transformer is an encoder-decoder stack of **self-attention + feed-forward** layers, each with residual connections and layer normalization.

```
Output → [Add & Norm] → [FFN] → [Add & Norm] → [Multi-Head Attention] → Input
```

For NLP, two main flavors emerged:

- **BERT** (encoder-only): Bidirectional context for understanding tasks (classification, NER, QA).
- **GPT** (decoder-only): Autoregressive left-to-right context for generation tasks.

## Using Pre-trained Transformers via Hugging Face

The `transformers` library provides a unified API for hundreds of pre-trained models.

### Tokenizers

Every model has its own tokenizer (usually BPE or WordPiece). The `AutoTokenizer` automatically selects the correct one:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoding = tokenizer("Transformers are powerful!", padding=True, truncation=True, return_tensors="pt")
# {'input_ids': tensor([[ 101,  ...,  102,    0,    0]]),
#  'attention_mask': tensor([[1, ..., 1, 0, 0]])}
```

Key parameters:
- `padding=True`: Pad to the longest sequence in the batch.
- `truncation=True`: Truncate to `max_length` (default: model's max, usually 512).
- `return_tensors="pt"`: Return PyTorch tensors.
- `attention_mask`: Tells the model which tokens are real (1) vs padding (0).

### BERT for Classification

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits                     # (1, 2)
    probs = torch.softmax(logits, dim=-1)       # (1, 2)
```

The model outputs `logits` — the same shape as a linear classifier on top of the `[CLS]` token's hidden state. BERT's `[CLS]` token is designed to aggregate sentence-level information.

### GPT for Generation

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    max_length=50,
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
# "Once upon a time, in a faraway land, there lived a brave knight..."
```

`model.generate()` supports all sampling strategies: `temperature`, `top_k`, `top_p`, `num_beams` for beam search, `repetition_penalty`.

## Model Hub

The [Hugging Face Model Hub](https://huggingface.co/models) hosts over 500k models. You can browse by:

- **Task**: text-classification, token-classification, text-generation, etc.
- **Framework**: PyTorch, TensorFlow, JAX.
- **Language**: multilingual models like `bert-base-multilingual-cased`.

Popular models for NLP:

| Model | Size | Notes |
|-------|------|-------|
| `bert-base-uncased` | 110M | General-purpose encoder |
| `roberta-base` | 125M | Improved BERT training |
| `distilbert-base` | 66M | 40% smaller, 95% performance |
| `gpt2` | 124M | General-purpose decoder |
| `microsoft/deberta-v3-base` | 86M | Strong on NLU benchmarks |
| `google/flan-t5-base` | 250M | Encoder-decoder, instruction-tuned |

## Minimal Inference Pipelines

The `pipeline` API wraps tokenizer + model + preprocessing:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I loved this movie!")  # [{'label': 'POSITIVE', 'score': 0.998}]

generator = pipeline("text-generation", model="gpt2")
generator("Deep learning is", max_length=30, num_return_sequences=3)
```

## Fine-Tuning vs Feature Extraction

- **Fine-tuning**: Update all model parameters on the downstream task. Best performance but expensive.
- **Feature extraction**: Freeze the pre-trained model and train only a small classifier on top. Cheaper, works well with limited data.
- **Adapter/LoRA**: Insert small trainable modules while keeping the base model frozen. State-of-the-art for parameter-efficient fine-tuning.

```python
# Feature extraction example
model = AutoModel.from_pretrained("bert-base-uncased")
model.eval()
with torch.no_grad():
    outputs = model(**encoding)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
```

## Key Takeaways

- Transformers replace recurrence with self-attention, enabling parallelization and long-range dependencies.
- Hugging Face provides a unified API for tokenizers, models, and pipelines.
- BERT (encoder) is for understanding tasks; GPT (decoder) is for generation.
- The Model Hub contains 500k+ models ready for download.
- `pipeline()` is the fastest way to use a model for inference.
- Fine-tuning > feature extraction > zero-shot in terms of task performance (at increasing cost).
