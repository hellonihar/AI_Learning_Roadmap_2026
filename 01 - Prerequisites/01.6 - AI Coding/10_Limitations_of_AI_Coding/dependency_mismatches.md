# Dependency Mismatches

AI often suggests libraries, versions, or combinations that conflict.

## Common Issues

### 1. Wrong Version
```python
# AI generated:
import torch
model = torch.nn.TransformerEncoderLayer(...)
# But project uses torch 1.9 which doesn't have TransformerEncoderLayer
```

### 2. Incompatible Library Combinations
```python
# AI suggests:
pip install transformers datasets accelerate

# But these might require different torch versions:
# transformers 4.38 → torch >= 1.11
# datasets 2.18 → torch >= 1.9
# accelerate 0.28 → torch >= 1.10
# All usually fine, but mismatches happen with newer/older releases
```

### 3. Platform-Specific Packages
```python
# AI suggests:
pip install dlib  # ❌ requires C++ compiler, often fails on Windows
```

### 4. Conflicting Pinned Dependencies
```
# requirements.txt has:
pandas==1.5.3
# AI generated code uses:
df.map(lambda x: x*2)  # .map() was added in pandas 2.0
```

## How to Mitigate

### 1. Tell AI your versions
```python
"Write code compatible with pandas 1.5.3 and Python 3.10"
```

### 2. Use dependency resolvers
```bash
pip install -r requirements.txt  # pip resolves conflicts
uv pip install -r requirements.txt  # faster resolver
```

### 3. Lock your dependencies
```bash
pip freeze > requirements.lock
# Or use:
uv lock
```

### 4. Ask AI to check compatibility
```
"Will this code work with [library]==[version]? List any potential
version conflicts with the existing dependencies: [list]"
```
