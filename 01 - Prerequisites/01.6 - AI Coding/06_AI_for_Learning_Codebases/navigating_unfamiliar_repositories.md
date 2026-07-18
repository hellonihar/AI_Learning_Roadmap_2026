# Navigating Unfamiliar Repositories

Use AI as a GPS for codebases you've never seen before.

## Navigation Prompts

```
"Where is the training loop defined?"
"How does the project load environment variables?"
"Where are the database models/schemas defined?"
"What's the entry point for the API server?"
"Where is the main configuration class?"
"How does authentication work?"
```

## Example

```
Prompt: "I need to add a new metric to the evaluation pipeline.
Find all files involved in the current eval pipeline and show the
data flow from predictions → scores."
```

**AI response:**
```
The evaluation pipeline flows through these files:

1. src/eval/runner.py          → orchestrates evaluation (line 45)
   Calls: loader.load_predictions() → metric functions → writer.save_results()

2. src/eval/metrics.py         → all metric functions defined here (line 12-89)
   Each function: (y_true, y_pred) → float

3. src/eval/loader.py          → loads predictions from CSV/DB

4. src/eval/writer.py          → saves results to JSON and DB

To add a new metric:
1. Add the function in metrics.py
2. Register it in METRICS_REGISTRY dict (line 90)
3. It will automatically appear in the output
```
