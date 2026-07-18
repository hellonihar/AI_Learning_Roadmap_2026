# Problem Statement — Fake News Detection

## Business Context
A social media platform needs to flag potentially false articles before viral amplification. Manual fact-checking cannot scale; an ML system acts as a first-pass filter to route suspicious content to human reviewers.

## Problem Type
Binary text classification: real (0) vs. fake (1).

## Success Metrics
- **F1 ≥ 0.85** — balance precision and recall equally.
- **Accuracy ≥ 0.88** — overall correctness benchmark.
- **Coverage across topics** — must generalize across politics, health, science.
- False positives are costly (censorship perception) but false negatives allow viral misinformation.

## Constraints
- Inference < 200 ms per article (to score at ingest time).
- Must work on article text alone (no external graph or source reputation in v1).
- Adversarial robustness — system should not be fooled by trivial rewording.
- Explainability preferred: highlight which sentences drive the prediction.
