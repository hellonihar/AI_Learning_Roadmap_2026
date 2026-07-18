# Step-by-Step Reasoning

When debugging complex issues, ask the AI to reason step by step.

## Chain of Thought Prompting

```
"Walk through the execution of this function with input X=5.
At each step, state what the values of all variables are.
Then tell me where it goes wrong."
```

## Example

```python
def calculate_discount(price, tier):
    if tier == "gold":
        discount = 0.2
    elif tier == "silver":
        discount = 0.1
    else:
        discount = 0.0
    return price * (1 - discount)

calculate_discount(100, "Gold")  # returns 100, not 80
```

**Prompt:** "Step through this with price=100, tier='Gold'. Why doesn't it apply the gold discount?"

**AI response:**
```
Step 1: price=100, tier="Gold"
Step 2: Check "Gold" == "gold" → False (case mismatch!)
Step 3: Check "Gold" == "silver" → False
Step 4: else branch → discount = 0.0
Step 5: return 100 * (1 - 0.0) = 100

Issue: Case-sensitive comparison. "Gold" != "gold".
Fix: Normalize case first: tier = tier.lower()
```

## When to Use

- Bugs that only happen with specific inputs
- Recursive or loop-heavy logic where you lose track
- Complex data transformations
- Race conditions or async ordering issues
