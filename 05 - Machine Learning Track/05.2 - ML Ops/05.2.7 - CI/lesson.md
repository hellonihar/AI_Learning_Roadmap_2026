# 05.2.7 CI/CD for Machine Learning

## Why CI/CD for ML?

Traditional CI/CD tests code and deploys to production. ML CI/CD extends this to test **data**, **models**, and **experiments**, because an ML system's behavior is determined by all three. A code change that passes unit tests might still break the ML system if it changes a preprocessing step that shifts the feature distribution.

### The ML CI/CD Pipeline

```
Lint ─► Unit Tests ─► Data Tests ─► Train ─► Evaluate ─► (Conditional) Promote ─► Deploy
   │                         │            │           │
   ▼                         ▼            ▼           ▼
flake8 / black          pytest         great_expectations   mlflow
```

## Testing in ML CI/CD

### 1. Testing Data (Schema Validation)

Before training, validate that incoming data meets expectations. **Great Expectations** is the most popular tool:

```python
import great_expectations as gx

suite = gx.ExpectationSuite("data_quality")
suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="age")
)
suite.add_expectation(
    gx.expectations.ExpectColumnMinToBeBetween(column="age", min_value=0, max_value=120)
)
suite.add_expectation(
    gx.expectations.ExpectColumnDistinctValuesToBeInSet(
        column="gender",
        value_set=["M", "F", "Other"],
    )
)
```

Other data tests:
- **Schema enforcement**: column names, dtypes, nullability.
- **Range checks**: no negative values for positive-only features.
- **Distribution checks**: KS/PSI test against training data.
- **Anomaly detection**: rows with all NaNs, duplicated rows, extreme outliers.

### 2. Testing Models

Model tests verify that the training process produces a valid model.

**Invariance tests**: Model predictions should not change when irrelevant features change.

```python
def test_prediction_invariant_to_irrelevant_feature():
    original = model.predict(sample)
    sample["id_column"] = 99999
    changed = model.predict(sample)
    assert original == changed
```

**Performance tests**: Minimum acceptable performance on a holdout set.

```python
def test_model_performance():
    metrics = evaluate(model, test_data)
    assert metrics["accuracy"] > 0.85
    assert metrics["f1_score"] > 0.80
```

**Fairness tests**: Performance parity across demographic groups.

```python
def test_fairness():
    group_a_accuracy = evaluate(model, test_data[test_data.group == "A"])
    group_b_accuracy = evaluate(model, test_data[test_data.group == "B"])
    assert abs(group_a_accuracy - group_b_accuracy) < 0.05
```

**Robustness tests**: Small perturbations to inputs should not flip predictions.

**Model size / type checks**: Ensure the artifact is the expected type and size.

### 3. Testing Code

Standard software testing for data processing, feature engineering, and deployment code.

```python
# test_features.py
def test_one_hot_encoding_creates_correct_columns():
    df = pd.DataFrame({"color": ["red", "blue", "green"]})
    encoded = one_hot_encode(df, "color")
    assert list(encoded.columns) == ["color_blue", "color_green", "color_red"]

def test_normalize_keeps_range():
    df = pd.DataFrame({"x": [10, 20, 30]})
    normalized = normalize(df, "x")
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
```

## CI/CD Platforms

### GitHub Actions

The most popular choice for open-source and GitHub-hosted projects.

```yaml
# .github/workflows/ml-ci.yml
name: ML CI Pipeline

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
      - run: pip install flake8 black
      - run: flake8 src/ --max-line-length=88
      - run: black --check src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements-dev.txt
      - run: pytest tests/ --cov=src/

  train:
    needs: [lint, test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: dvc pull  # download data from S3
      - run: dvc repro  # run pipeline if needed
      - run: mlflow run .  # train with tracking
      - uses: actions/upload-artifact@v4
        with:
          name: model
          path: models/

  evaluate:
    needs: [train]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
      - run: python scripts/evaluate.py --model models/model.pkl
      - name: Check metrics
        run: |
          ACC=$(python -c "import json; print(json.load(open('metrics.json'))['accuracy'])")
          if (( $(echo "$ACC < 0.85" | bc -l) )); then
            echo "Accuracy $ACC below threshold 0.85"
            exit 1
          fi

  deploy:
    needs: [evaluate]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
      - run: docker build -t inference:latest .
      - run: docker push registry.example.com/inference:latest
      - run: kubectl set image deployment/inference inference=registry.example.com/inference:latest
```

### Jenkins

Mature, plugin-rich, widely used in enterprise environments. Uses `Jenkinsfile` for pipeline-as-code.

```groovy
pipeline {
    agent any
    stages {
        stage('Lint') { steps { sh 'flake8 src/' } }
        stage('Test') { steps { sh 'pytest tests/' } }
        stage('Train') { steps {
            sh 'dvc pull'
            sh 'dvc repro'
        }}
        stage('Evaluate') { steps {
            sh 'python scripts/evaluate.py'
        }}
        stage('Deploy') {
            when { branch 'main' }
            steps { sh './deploy.sh' }
        }
    }
}
```

### GitLab CI

Deeply integrated with GitLab, uses `.gitlab-ci.yml`. Supports autoscaling runners and GPU runners.

```yaml
train:
  stage: train
  script:
    - dvc pull
    - python train.py
  artifacts:
    paths:
      - models/
  tags:
    - gpu
```

## Conditional Promotion

Not every model should go to production. The CI pipeline tests performance and **promotes only if thresholds are met**:

```python
# evaluate.py
import json

with open("metrics.json") as f:
    metrics = json.load(f)

if metrics["accuracy"] < 0.85:
    print(f"FAIL: accuracy {metrics['accuracy']} < 0.85")
    exit(1)  # Pipeline stops — model not promoted

print(f"PASS: accuracy {metrics['accuracy']} >= 0.85")
exit(0)  # Pipeline continues to deploy
```

This **gate** prevents regressions. Combine with human review: the pipeline deploys to **Staging**, runs integration tests, then requires a manual approval button before hitting **Production**.

## Data Pipeline CI

For pipelines that use DVC or dbt:

```yaml
data-ci:
  stage: data
  script:
    - dvc pull
    - dvc repro
    - great_expectations checkpoint run train_data
```

## Best Practices

- **Pin CI runner images** — avoid `latest` tags for Python or CUDA.
- **Cache dependencies** — pip cache, DVC cache, Docker layers.
- **Use a dev/test/staging/production model registry** — never push directly to production.
- **Run data tests on every PR** — catch bad data before it wastes GPU hours.
- **Log all CI runs to the experiment tracker** — link the commit, run, and metrics.
- **Keep training under 30 minutes in CI** — full-scale training can be triggered on a schedule or on `main`.
- **Use conditional stages** — skip deploy if accuracy dropped.

## Summary

ML CI/CD extends traditional CI/CD with data validation, model performance gates, and conditional promotion. GitHub Actions provides the smoothest experience for most teams, while Jenkins and GitLab CI excel in enterprise settings. The critical discipline is **gating**: never deploy a model that degrades metrics, even if the code passes all unit tests.
