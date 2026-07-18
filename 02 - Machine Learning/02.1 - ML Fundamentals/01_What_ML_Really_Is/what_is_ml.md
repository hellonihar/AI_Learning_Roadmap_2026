# What Machine Learning Really Is

## ML vs Traditional Programming

| Traditional Programming | Machine Learning |
|---|---|
| You write rules explicitly | The model discovers rules from data |
| Fixed logic, deterministic | Statistical, probabilistic |
| Brittle to unseen edge cases | Generalizes to new examples |
| Example: `if temp > 30: return "hot"` | Example: learn from 10K labeled weather examples |

**Traditional programming**: input `+` rules → output
**Machine learning**: input `+` output → rules

## Why ML Works: Pattern Discovery from Data

- ML finds statistical patterns too complex for humans to hand-code
- More data → more reliable patterns
- It generalizes: patterns learned from training set apply to unseen data (if the data is representative)

### Example 1: Spam Detection
- Traditional: write regex rules (`contains "viagra"`, `from unknown sender`, etc.) — brittle, spammers easily bypass
- ML: train on 100K labeled emails → model learns subtle patterns (misspellings, unusual sender behavior, embedding structure) → catches never-before-seen spam

### Example 2: Credit Card Fraud
- Traditional: rule like `if transaction > $10K in foreign country → flag`
- ML: learns your spending profile — amount, merchant, location, time-of-day patterns — flags a \$3 coffee purchase at 3 AM in a city you're not in, even though \$3 is normally fine

### Example 3: Image Recognition
- Traditional: impossible to hand-code rules for "is this a cat?"
- ML: learns hierarchical features — edges → shapes → textures → object parts → cats — from millions of labeled images
