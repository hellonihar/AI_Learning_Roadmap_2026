# Testing with pytest

## Basic Tests
```python
# test_model.py
def test_predict_shape():
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)

def test_predict_range():
    assert all(0 <= p <= 1 for p in preds)
```

## Fixtures
```python
import pytest

@pytest.fixture
def trained_model():
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

def test_accuracy(trained_model):
    acc = trained_model.score(X_test, y_test)
    assert acc > 0.8
```

## Parametrize
```python
@pytest.mark.parametrize("lr,expected", [
    (1e-3, True),
    (1e-1, True),
    (10.0, False),
])
def test_learning_rate(lr, expected):
    assert (0 < lr < 1) == expected
```

## Mocking
```python
from unittest.mock import patch

@patch("model.predict")
def test_api(mock_predict):
    mock_predict.return_value = [0.95]
    response = client.post("/predict", json={"features": [1,2,3]})
    assert response.json()["confidence"] == 0.95
```

## Running
```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term
```
