# Hugging Face Transformers

## Pipelines (Quick Start)
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I love this product!")

summarizer = pipeline("summarization")
summarizer(long_text)

generator = pipeline("text-generation", model="gpt2")
generator("Once upon a time")
```

## Using Models & Tokenizers
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

## Batched Inference
```python
batch = ["text one", "text two", "text three"]
inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
preds = outputs.logits.argmax(dim=-1)
```

## Datasets
```python
from datasets import load_dataset

dataset = load_dataset("imdb")
dataset["train"][0]  # {"text": ..., "label": 0/1}
```
