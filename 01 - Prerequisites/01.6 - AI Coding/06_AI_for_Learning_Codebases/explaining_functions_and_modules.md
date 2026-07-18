# Explaining Functions and Modules

Drill into specific parts of a codebase using targeted prompts.

## Prompt Templates

```
"Explain the function {name} in {file}. What does it do?
What are the inputs and outputs? What side effects does it have?"
```

```
"Explain the {class_name} class. What's its responsibility?
What are its key methods? What pattern does it follow?"
```

```
"Explain the {module_name} module. What's its public API?
Which other modules depend on it?"
```

## Example

```
"Explain the 'preprocess' function in src/data/loader.py"
```

**AI response:**
```
Preprocess function at loader.py:142-189:

Signature: preprocess(df: DataFrame, config: Config) -> DataFrame

Purpose: Applies transformations to raw DataFrame before training.

Steps:
1. Drops columns not in config.feature_columns
2. Fills numeric NaN with column median
3. One-hot encodes categorical columns from config.cat_columns
4. Normalizes numeric columns to [0,1] range
5. Logs shape before/after and any dropped rows

Side effects: None (pure function, no I/O outside logging)
Dependencies: sklearn.preprocessing.LabelEncoder, pandas

Key detail: If config.normalize is False, step 4 is skipped.
Use this to skip normalization during inference.
```
