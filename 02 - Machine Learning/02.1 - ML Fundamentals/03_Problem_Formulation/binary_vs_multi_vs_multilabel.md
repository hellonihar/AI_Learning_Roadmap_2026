# Binary vs Multi-Class vs Multi-Label Classification

## Binary Classification

Exactly **2** possible classes.

**Examples:**
1. **Email spam**: "spam" or "not spam" — every email falls into exactly one of two buckets
2. **Credit card fraud**: "fraudulent" or "legitimate" — a transaction is one or the other
3. **Medical test**: "positive" or "negative" for a specific condition (e.g., COVID test)

## Multi-Class Classification

**3+** possible classes. Each sample belongs to **exactly one** class.

**Examples:**
1. **Image classification**: Photo → "cat", "dog", "bird", or "fish" — each image has exactly one animal
2. **Language identification**: Text → "English", "Spanish", "French", or "Mandarin" — a sentence is in exactly one language
3. **Handwritten digit recognition**: Digit image → "0", "1", ..., "9" — each digit is exactly one number

## Multi-Label Classification

Each sample can belong to **multiple classes simultaneously**.

**Examples:**
1. **Movie genre tagging**: A film can be both "action" and "comedy" → model must output all applicable genres, not just one
2. **Medical diagnosis**: A patient can have "diabetes" AND "hypertension" AND "obesity" — all three labels are true
3. **Image tagging**: A photo contains "sky", "tree", "person", "beach" — model must predict every object present

## Comparison Table

| Aspect | Binary | Multi-Class | Multi-Label |
|---|---|---|---|
| Number of classes | 2 | 3+ | 2+ |
| Labels per sample | 1 | 1 | 0 to many |
| Output layer | 1 neuron (sigmoid) | N neurons (softmax) | N neurons (sigmoid each) |
| Loss | Binary cross-entropy | Categorical cross-entropy | Binary cross-entropy (per label) |
