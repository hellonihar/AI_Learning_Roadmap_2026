# Error Handling & Logging

## Try / Except / Finally
```python
try:
    result = risky_operation()
except ValueError as e:
    print(f"Bad value: {e}")
except Exception as e:
    print(f"Unexpected: {e}")
    raise  # re-raise
else:
    print("No error occurred")
finally:
    cleanup()
```

## Custom Exceptions
```python
class ModelNotTrainedError(Exception):
    pass

def predict(model, X):
    if not model.fitted:
        raise ModelNotTrainedError("Call .fit() first")
```

## Logging
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Training started")
logger.warning("Low accuracy: %.3f", acc)
logger.error("Failed to load model")
```
