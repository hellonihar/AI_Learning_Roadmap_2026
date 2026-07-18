# Hypothesis-Driven Debugging

Instead of asking "why is this broken?", propose a hypothesis and ask AI to verify.

## The Hypothesis Loop

```
1. Observe symptom → "The model predicts nan for all inputs"
2. Propose hypothesis → "Maybe learning rate is too high"
3. Ask AI → "Could a high learning rate cause nan predictions?
   What else could cause this?"
4. AI evaluates → "Yes, LR > 1e-3 can cause gradient explosion.
   Also check: input normalization, division by zero in loss"
5. Test → Check LR, inputs, loss function
6. Repeat → "I checked LR (1e-4) and inputs (normalized).
   What else?" → "Check for log(0) in your loss"
```

## Example

```
You: "My binary classifier always predicts class 0, never class 1."
AI: "Several possibilities: 1) Class imbalance → no weights
     2) Learning rate too low → stuck in local minimum
     3) Last layer activation is sigmoid but loss expects logits
     4) All labels are 0 in your training data"
You: "Let me check the loss function..."
AI: "If you're using nn.BCELoss, ensure you pass through
sigmoid first. If using nn.BCEWithLogitsLoss, don't apply
sigmoid before the loss. What are you using?"
```

## Good Hypothesis Prompts

```
- "What could cause [symptom]? Give me 3 most likely causes ordered by probability"
- "I checked [X] and [Y], they're fine. What else should I check?"
- "Given the evidence [evidence], is [hypothesis] the most likely cause?"
```
