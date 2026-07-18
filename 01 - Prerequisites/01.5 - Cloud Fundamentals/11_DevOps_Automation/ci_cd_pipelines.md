# CI/CD Pipelines (Conceptual)

CI/CD automates testing and deployment of code changes.

## CI (Continuous Integration)
- Automatically test every commit
- Run lint, type check, unit tests, integration tests
- Fail fast — prevent broken code from reaching production

## CD (Continuous Delivery / Deployment)
- Automatically deploy tested code to production
- **Delivery**: deploy button after CI passes
- **Deployment**: fully automated deploy after CI passes

## ML Pipeline Example
```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline
on: [push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |  # build and push container
          docker build -t my-model:latest .
          docker push registry/my-model:latest
      - run: terraform apply  # update endpoints
```

## For AI Engineers
- CI should validate: code lint, unit tests, model tests (schema, predictions)
- CD should deploy: build container → push to registry → update Terraform
- Add a manual approval step for production model deployments
- Don't run full training in CI — train separately, CI only validates
