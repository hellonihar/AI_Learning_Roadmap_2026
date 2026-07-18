# Modeling Strategy

## 4.1 Model Selection Philosophy

### Simple Models vs Complex Models

| Dimension | Simple (Linear, Logistic, Decision Tree) | Complex (GBM, Neural Network, Transformer) |
|---|---|---|
| **Interpretability** | High — you can read the coefficients / tree | Low — requires SHAP, LIME, or integrated gradients |
| **Data requirement** | Works with hundreds to thousands of samples | Needs tens of thousands+ to outperform simple models |
| **Training time** | Seconds to minutes | Hours to days |
| **Inference latency** | Microseconds | Milliseconds to seconds (depending on size) |
| **Maintenance** | Easy to debug, monitor, and retrain | Requires MLOps, GPU infra, monitoring for subtle failures |
| **Performance ceiling** | Lower (assumes linearity / simple interactions) | Higher (captures non-linear interactions, high-cardinality patterns) |

**Trade-off guide:**

| Situation | Choose |
|---|---|
| < 10K samples, need to explain each prediction | Logistic regression, decision tree |
| > 100K samples, interpretability not critical, accuracy matters | XGBoost / LightGBM |
| Text, image, audio, or sequential data | Deep learning (CNN, RNN, Transformer) |
| Need fast iteration (5 models/day), small team | Start with simple, upgrade only when simple plateaus |
| Regulatory audit required for each decision | Simple model or interpretable ML (EBM, monotonic GBM) |

**Best practice:** Always fit a simple baseline (mean prediction, linear model, rule-based) before reaching for complex models. If the complex model isn't 5–10% better, the simplicity win probably isn't worth it.

### When Deep Learning Is Overkill

Deep learning is the wrong choice when:

1. **Tabular data** — XGBoost, LightGBM, and CatBoost consistently match or beat deep learning on structured tables with less compute, less data, and less tuning.
2. **Small dataset** — DL needs 10K–100K+ samples. With 1K rows, a random forest will likely outperform a neural network.
3. **Latency-sensitive serving** — A 200-parameter logistic regression serves in microseconds; a 7B LLM needs a GPU and hundreds of milliseconds.
4. **Simple decision boundaries** — If a decision tree with depth 3 achieves 92% accuracy, there's no need for a 12-layer MLP.
5. **No GPU infrastructure** — Training deep nets on CPU is painfully slow. If your org doesn't have GPU CI/CD, you'll ship models slowly or not at all.

**Real example:** A payment fraud model with 200 tabular features and 50K transactions/month. A gradient-boosted tree (LightGBM) achieves 0.97 AUC, trains in 30 seconds, and serves in 200μs. A deep neural network achieves 0.975 AUC but takes 4 hours to train and 3ms to serve. The 0.5% lift doesn't justify the complexity — stick with the tree.

### Ensemble vs Single Model Systems

| | Single Model | Ensemble (Bagging, Boosting, Stacking) |
|---|---|---|
| **Performance** | Good | Better (usually 2–5% lift) |
| **Inference cost** | Low | N× cost (run K models) |
| **Debugging** | Easy (one model to inspect) | Hard (which model made the mistake?) |
| **Deployment** | Simple (one artifact) | Complex (K artifacts, orchestrator) |
| **Overfitting risk** | Higher (single model can memorize noise) | Lower (averaging reduces variance) |
| **Interpretability** | Easier | Harder (which prediction is the "ensemble consensus"?) |

**When to ensemble:**
- **Bagging (Random Forest):** Default for high-variance models (deep trees). Reduces overfitting with minimal tuning.
- **Boosting (XGBoost, LightGBM):** Default for structured data. Sequentially corrects errors — best performance per compute unit.
- **Stacking:** Combine complementary models (e.g., GBM + linear + NN) when they make different kinds of errors. Worth it for competition-grade accuracy or high-stakes predictions.
- **Simple averaging:** If you have 3 good models trained independently, averaging their predictions often beats any single one — free lunch.

**When NOT to ensemble:**
- Latency budget is tight (can't run 3+ models)
- Deployment complexity outweighs performance gain (1 model to maintain vs 5)
- The single model already meets the business metric target

---

## 4.2 Training Strategy

### Offline Training vs Continual Training

| | Offline (Batch Retraining) | Continual (Online / Incremental) |
|---|---|---|
| **Schedule** | Periodic (daily, weekly, monthly) | Continuous (update as data arrives) |
| **Data** | Fixed training set, full retrain | Streaming, incremental update |
| **Cost** | Low (compute once per cycle) | Higher (constant compute) |
| **Stability** | High (model changes only at retrain) | Lower (model can drift with noisy data) |
| **Complexity** | Low — standard training pipeline | High — needs monitoring for feedback loops |
| **Best for** | Stable patterns, batch features, scheduled eval | Fast-changing patterns, real-time signals |

**Hybrid approach:** Weekly full retrain + daily incremental updates. The full retrain resets any accumulated drift; the daily updates keep the model fresh between retrains. Most production systems use this pattern.

### Retraining Frequency

| Frequency | When It Works | When It Fails |
|---|---|---|
| **Real-time (per event)** | Online learning, RL, bandits | Overfits to noise, feedback loops amplify |
| **Hourly** | Fast-moving signals (fraud, news ranking) | Expensive for large models; overkill if features change slowly |
| **Daily** | Most common for production ML (recommendations, search, churn) | May miss intraday shifts (rare) |
| **Weekly** | Stable signals (credit scoring, LTV prediction) | Slow to react to market or user behavior changes |
| **Monthly / quarterly** | Regulatory models, long-term trends | Concept drift goes undetected for weeks |

**To set retraining frequency:**
1. Measure the half-life of your feature-label relationship — how long until model accuracy drops by X%?
2. Retrain at least 2× faster than that half-life
3. Monitor performance daily — if it drops below threshold, trigger an unscheduled retrain

**Real example:** An ad click-through rate model retrains daily because user behavior shifts with day-of-week, holidays, and campaigns. A mortgage default model retrains quarterly because defaults are rare events and the relationship is stable.

### Cold Start Handling

Cold start = no historical data for a new user, item, or entity.

| Strategy | How It Works | Trade-off |
|---|---|---|
| **Fallback to population average** | Predict the global mean for new entities | Safe but unhelpful — no personalization |
| **Content-based features** | Use attributes (category, device type, location) instead of historical behavior | Works immediately, but needs good attributes |
| **Explore-then-exploit** | Show random content, collect feedback, then personalize (Thompson sampling, ε-greedy) | Short-term quality hit, long-term gain |
| **Transfer from similar entities** | Use features from users/items in the same segment (city, acquisition channel) | Assumes similarity; can be wrong |
| **Meta-learning (few-shot)** | Train on related tasks; adapt to new entity with few examples | Complex; worth it at scale |

**Best practice:** Combine fallback + exploration. Serve the population average initially, then over the first few interactions, blend in personalized predictions as data accumulates.

### Bootstrapping Labels

When you have a model but no labels, you need to bootstrap training data.

| Method | Example | Risk |
|---|---|---|
| **Weak supervision** | Heuristics + label models (Snorkel) to generate noisy labels | Label noise propagates to model; evaluate on a small clean set |
| **Active learning** | Model selects the most informative examples for humans to label | Requires labeling infra; cold start problem (model needs initial data to select) |
| **Semi-supervised** | Train on labeled set, pseudo-label the unlabeled set, retrain; iterate | Pseudo-labels reinforce model biases; must filter low-confidence predictions |
| **Rules → ML transition** | Start with hand-written rules, use them as training targets for a learned model | Rules may be inconsistent; the model learns the rules' mistakes |
| **Transfer learning** | Pre-train on a related task with abundant data, fine-tune on your task | Feature / distribution mismatch between pre-training and target domain |

**Real example:** A content moderation system bootstraps with keyword rules (blocklist + regex), logs 10K human review decisions, trains a classifier on the logged data, and then phases out the rules as the classifier improves. The rules provide the initial training signal; the classifier eventually outperforms them.

---

## 4.3 Handling Data Imbalance

### Sampling Strategies

| Strategy | How It Works | Best For | Downside |
|---|---|---|---|
| **Random undersampling** | Remove samples from majority class | Large dataset, mild imbalance (10:1) | Throws away data — reduces signal |
| **Random oversampling** | Duplicate samples from minority class | Small dataset, moderate imbalance | Overfits to duplicated examples |
| **SMOTE** | Synthesize new minority samples by interpolating between nearest neighbors | Moderate imbalance, continuous features | Creates unrealistic samples for high-dimensional or categorical data |
| **ADASYN** | SMOTE variant that generates more samples for harder-to-learn minority examples | Complex decision boundaries | More noise; may amplify outliers |
| **Near-miss undersampling** | Keep majority samples closest to minority class | When boundary separation is key | Sensitive to outliers |
| **Cluster-based undersampling** | Cluster majority class, sample from each cluster | Large datasets with internal structure | Slower, needs tuning of cluster count |

**Best practice:** Try no sampling first (use class weights). If the model ignores the minority class, try class-weighted loss → oversampling → SMOTE, in that order. Over-sampling without cross-validation can give optimistically biased accuracy due to duplicate rows appearing in both train and validation splits.

### Loss Functions

| Loss Function | Imbalance Handling | When to Use |
|---|---|---|
| **Cross-entropy** | None (default) | Balanced classes or when using class weights |
| **Weighted cross-entropy** | Scale loss by inverse class frequency | Standard go-to for imbalance |
| **Focal loss** | Down-weight easy examples, focus on hard ones | Extreme imbalance (1:1000+), object detection |
| **Hinge / weighted hinge** | Margin-based + class weights | Binary classification with SVMs |
| **Log loss with threshold moving** | Standard cross-entropy + post-hoc threshold tuning | When you want to train normally and adjust the decision boundary later |

**Focal loss intuition:**
```
Standard cross-entropy: all wrong predictions get equal gradient
Focal loss: if model is already confident (right or wrong), reduce the loss
           → forces the model to focus on the hard, often minority-class examples
```

**Real example:** A rare disease classifier with 1:1000 prevalence. Standard cross-entropy produces a model that predicts "no disease" for everyone (99.9% accuracy, 0% recall). Focal loss + threshold tuning achieves 80% recall with 95% precision on the minority class.

### Metric Selection Implications

In imbalanced settings, standard accuracy is misleading. Choose metrics that reflect the imbalance:

| Metric | Sensitive to Imbalance? | Interpretation |
|---|---|---|
| **Accuracy** | Yes — a 99% accurate model can have 0% recall on minority class | Misleading on imbalanced data |
| **Precision / Recall / F1** | No (per-class) | F1 is harmonic mean of precision and recall; micro vs macro F1 matters |
| **AUC-ROC** | No — measures rank ordering, not absolute scores | Over-optimistic on highly imbalanced data (ROC may look good while precision is terrible) |
| **AUC-PR (Precision-Recall)** | Yes — reflects minority class performance | Use instead of AUC-ROC for imbalanced problems |
| **Log loss** | Partially — proper scoring rule, but dominated by majority class | Use weighted log loss |
| **Specificity / Sensitivity** | No — per-class | Good for medical: sensitivity = recall on positive class, specificity = recall on negative |

**General rule:** For imbalance < 10:1, AUC-PR + weighted log loss. For 100:1+, use precision@K / recall@K (since you only act on top predictions) and focal loss.

**Best practice:** Report confusion matrix + precision/recall per class + AUC-PR. Never report only accuracy.

---

## 4.4 Model Interpretability

### When Interpretability Is Required

| Situation | Why | Approach |
|---|---|---|
| **Regulated decisions** (loans, credit, insurance) | ECOA, GDPR right to explanation | Use inherently interpretable models (logistic regression, EBM, monotonic GBM) |
| **Medical diagnosis** | Clinicians won't trust black-box predictions | SHAP + feature importance + case-based reasoning |
| **Debugging production failures** | Need to understand why a model made a bad prediction | SHAP force plots, LIME, partial dependence |
| **Stakeholder buy-in** | Business leaders need to trust model behavior | Global feature importance + representative examples |
| **Model improvement** | Understanding weaknesses guides feature engineering | SHAP summary plot, accumulated local effects (ALE) |

**Levels of interpretability:**

| Level | What You Can Explain | Example |
|---|---|---|
| **Global** | Which features matter most overall | Feature importance rank, SHAP summary |
| **Local** | Why this specific prediction | SHAP force plot, LIME, decision tree path |
| **Per-segment** | How does the model behave for a subgroup (e.g., new users) | Stratified SHAP, partial dependence by segment |
| **Counterfactual** | What would change this prediction | "If income were 10K higher, prediction flips from deny to approve" |

### Debugging Models in Production

A model that performs well offline can fail in production. Debug systematically:

| Signal | Possible Cause | Fix |
|---|---|---|
| Accuracy drops suddenly | Feature pipeline change, data corruption | Check feature distributions, roll back feature change |
| Accuracy drifts gradually | Concept drift, user behavior change | Retrain on recent data, investigate external factors |
| Predictions shift to one class | Label drift, threshold misalignment | Recalibrate or adjust threshold |
| High variance in predictions | Feature missing → zero imputed → spike | Fix imputation, add validation for missing features |
| Model serves stale predictions | Feature freshness issue → old feature values | Alert on feature age; check streaming pipeline |
| Silent failures (no error, no alert) | Bad training data (wrong labels, data leakage) | Manual audit of training samples; data quality checks |

**Debugging workflow:**
1. Logged feature vector + prediction + actual outcome (when available)
2. Compare feature distributions (last week vs today) — PSI > 0.1 needs investigation
3. Examine high-error examples with SHAP — are they missing key features?
4. Check if the error pattern matches a known assumption that broke (e.g., new product category)

### Interpretable Proxies

When you must use a black-box model (accuracy, latency constraints) but need interpretability, train an **interpretable proxy**:

```
Black-box model (XGBoost, NN) → logits/predictions
                          ↓
Interpretable model (Linear, Shallow Tree) → learns to mimic black-box predictions
                          ↓
                      Explanations from proxy
```

| Proxy Model | Interpretability | Fidelity | When to Use |
|---|---|---|---|
| **Linear regression (on logits)** | High (coefficients) | Low (assumes linearity) | Quick, rough understanding of feature influence |
| **Decision tree (depth ≤ 3)** | High (rule set) | Medium | Non-linear interactions, few important features |
| **EBM (Explainable Boosting Machine)** | High (per-feature shape functions) | Medium-High | Best of both worlds — accurate and interpretable |
| **SHAP (model-agnostic)** | High (local + global) | High (game-theoretic) | Most reliable; works with any model; expensive for high-dim |

**Trade-off:** Simpler proxy = easier to explain but less faithful to the black box. Always validate proxy fidelity (R², % of predictions where proxy agrees with black box). If fidelity < 80%, the proxy is misleading.

**Best practice:** Use SHAP as the primary explanation tool for black-box models. It provides consistent, theoretically grounded local and global explanations.

---

## 4.5 Model Lifecycle

### Model Versioning

Every model artifact needs a unique, immutable identifier that maps back to the full training context.

**Versioning scheme:**
```
model_name: str              # e.g., "fraud_classifier"
model_version: semantic      # e.g., "2.3.1"
  major: breaking change (schema, features)
  minor: new features, non-breaking
  patch: retrain, hyperparam tune, no feature change
artifact_hash: sha256        # content-addressable
training_run_id: uuid        # maps to MLflow / experiment DB
```

**What to store with the model:**
```
model_manifest:
  - model_name, version, hash
  - git commit (training code)
  - dataset version (path, hash)
  - feature list + schemas
  - hyperparameters + random seed
  - training metrics (eval set)
  - framework + library versions
  - timestamp
  - owner / team
```

**Practice:** Never overwrite a model artifact. Write a new version. This makes rollbacks trivial and audit trails complete.

### Backward Compatibility

A model update should not break downstream consumers.

| Compatibility | Rule | Example Violation |
|---|---|---|
| **API compatibility** | Input schema (feature names, types) stays the same | Model v3 expects `click_count_7d` (was `click_7d` in v2) |
| **Output compatibility** | Prediction format, range, semantics stay the same | V2 outputs probability 0.0–1.0; V3 outputs logit -10–+10 |
| **Behavioral compatibility** | New model should not silently reverse decisions on the same input | Feature `is_premium_user=1` with V2 → approve; V3 → deny for same input without notice |
| **Operational compatibility** | Latency, memory, throughput stay within expected bounds | V3 is a 2GB model (V2 was 50MB) — breaks memory limit on serving infra |

**Shadow deployment:** Route traffic to the new model in parallel with the old model. Compare outputs. Only promote if:
- Predictions don't diverge wildly on the same inputs
- Latency / resource usage is acceptable
- Business metrics are neutral or better

**Breaking changes require a major version bump** and a migration guide for all consumers. Never sneak a breaking change into a patch release.

### Rollbacks

| Trigger | Action | Speed | Risk |
|---|---|---|---|
| **Accuracy crash** (> 20% drop) | Revert to previous model version immediately | Fast (minutes) | Model may re-expose a previously fixed bug |
| **Latency spike** (p99 > 3× baseline) | Fall back to lightweight heuristic or previous model | Fast | Heuristic is less accurate; accept temporarily |
| **Data pipeline failure** | Block model deployment, serve cached predictions or fallback | Medium | Stale predictions; document freshness age |
| **Escaped bad training data** (wrong labels in latest dataset) | Retrain on corrected data; rollback to last good model while training | Medium | Delay until new model is ready |
| **Slow degradation** (gradual metric drift) | Analyze root cause; rollback only if no fix is available | Slow | Prefer retrain + deploy over rollback |

**Rollback infrastructure:**

| Component | Requirement |
|---|---|
| **Artifact storage** | Every version is immutable and available (S3, model registry) |
| **Serving infra** | Canary deploy + instant traffic switch (Kubernetes, feature flags) |
| **Monitoring** | Dashboard shows current and previous model metrics side-by-side |
| **Automation** | Automatic rollback if error rate > threshold for > 5 minutes |

**Best practice:** Keep the previous N models (typically N=3) deployed but idle. A rollback is then a traffic-routing change, not a redeployment. Test the rollback path regularly (chaos engineering).

### End-to-End Lifecycle Summary

```
Problem definition
  → Data collection & labeling
  → Feature engineering
  → Model selection & training
  → Offline evaluation
  → Shadow deployment (parallel with current model)
  → Canary deployment (5% → 25% → 50% → 100%)
  → Production monitoring (features, predictions, outcomes)
  → Retrain or rollback on metric degradation
  → Model retirement (archive artifact + manifest)
```

Each stage has a go/no-go gate. Never skip gates under pressure — it's how bad models reach production.
