# Dependency Management

## Pip / UV

```bash
# Install from requirements
uv pip install -r requirements.txt

# Freeze exact versions (for reproducibility)
uv pip freeze > requirements.lock

# Install in editable mode (for your own package)
uv pip install -e .
```

## Poetry
```bash
poetry new my-ai-lib
poetry add openai torch
poetry install
poetry lock     # generate poetry.lock
```

## Conda (for non-Python deps like CUDA)
```bash
conda create -n ai-env python=3.12
conda activate ai-env
conda install cudatoolkit=11.8 pytorch=2.0 -c pytorch
```

## Dependency Pinning Strategy
```
# Loose (for libraries):
openai>=1.0

# Strict (for applications / experiments):
openai==1.14.3
torch==2.1.2
transformers==4.36.2
```

## Lock Files
Always commit lock files (`requirements.lock`, `poetry.lock`, `uv.lock`) to ensure reproducible installs across machines and CI.
