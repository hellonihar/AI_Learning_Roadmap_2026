# Modularization

Break large functions into smaller, focused, testable modules.

## Monolithic → Modular

```python
# Before (one giant function):
def process_orders(orders):
    # 50 lines: read, validate, transform, save, email
    ...
```

**Prompt:** "Split this function into focused modules: reader, validator, transformer, notifier"

```python
# After:
# reader.py
def read_orders(path: str) -> list[dict]:
    ...

# validator.py
def validate_order(order: dict) -> bool:
    ...

# transformer.py
def transform_order(order: dict) -> dict:
    ...

# notifier.py
def send_confirmation(email: str, order_id: str) -> None:
    ...

# main.py
def process_orders(path: str) -> None:
    orders = read_orders(path)
    valid = [o for o in orders if validate_order(o)]
    transformed = [transform_order(o) for o in valid]
    for t in transformed:
        send_confirmation(t["email"], t["id"])
```

## Modularization Prompts

```
- "Split this file into separate modules by concern"
- "Extract the database layer into a repository class"
- "Separate the config loading into its own module"
- "Move the data models into models.py"
- "Create a services layer between routes and data access"
```
