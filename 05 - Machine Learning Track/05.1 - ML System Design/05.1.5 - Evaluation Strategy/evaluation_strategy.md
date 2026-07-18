# Evaluation Strategy

Model evaluation is the process of measuring how well a model performs on data it hasn't seen before. It answers two questions: "Does this model work?" and "How well does it work?" Without evaluation, you're guessing — a model that memorized the training data (overfitting) can achieve near-perfect training metrics while failing catastrophically in production.

An evaluation strategy is essential because the answer to "does this model work?" depends on context. A fraud model with 95% precision might be excellent for a low-volume bank but unacceptable for a high-volume payment processor where 5% of legitimate transactions being blocked means millions in lost revenue. Evaluation isn't a single number — it's a multi-faceted assessment covering offline metrics (accuracy, precision, recall), online tests (A/B tests, canary releases), slice-level fairness, and long-term business impact. A well-defined strategy ensures you catch failures before they reach users, make honest comparisons between models, and connect model performance to business outcomes rather than optimizing proxy metrics in isolation.

## 5.1 Offline Evaluation

### Train / Validation / Test Splits

| Split | Purpose | Typical Size | Use |
|---|---|---|---|
| **Train** | Model learns parameters | 70–80% | Gradient descent, tree building |
| **Validation** | Hyperparameter tuning, early stopping | 10–15% | Model selection, threshold tuning |
| **Test** | Final, unbiased estimate of generalization | 10–15% | Report as "model performance" — never touch during development |

**Golden rule:** The test set must be touched exactly once — when you produce the final evaluation report. Every time you look at test set metrics and then change the model, the test set becomes a second validation set.

**Trade-off:**
- Small test set → high variance in reported metric (confidence interval is wide)
- Large test set → less data for training (model quality suffers)
- **Best practice:** Use k-fold cross-validation on the train set for model selection. Hold out a fixed test set only for final reporting. If data is scarce, use nested cross-validation.

### Time-Aware Splits

For time-series or any data where time matters, **random splits leak future information into training**.

**Wrong (random split):**
```
Training: purchases from Jan–Dec 2025 + some from Jan 2026 (randomly sampled)
Test: remaining purchases from Jan–Dec 2025 + some from Jan 2026
→ Model sees 2026 patterns during training and is evaluated on 2025 data it already saw
```

**Correct (time-aware split):**
```
Training: purchases from Jan–Nov 2025
Validation: Dec 2025
Test: Jan 2026
→ Model predicts the future using only past data
```

| Split Method | When to Use | Limitation |
|---|---|---|
| **Chronological** (train on past, test on future) | Time series, fraud, forecasting | Assumes the future resembles the past (concept drift) |
| **Rolling window** | Retraining simulations | More computationally expensive |
| **Purged walk-forward** | Finance (prevent leakage from overlapping events) | Complex to implement; needs careful gap management |

**Real example:** A fraud model trained on random splits achieved 0.98 AUC offline but 0.72 AUC in production. The random split leaked future fraud patterns (fraudsters change tactics weekly). A time-aware split gave a realistic 0.81 AUC, which matched production. The model wasn't as good as the random split suggested — it needed more frequent retraining.

### Leakage-Resistant Evaluation

| Leakage Type | How It Happens | How to Prevent |
|---|---|---|
| **Temporal leakage** | Future data in training window | Time-aware splits; never use `shift(-1)` |
| **Group leakage** | Same entity (user, session) in train and test | Group split by `entity_id`; never by rows |
| **Feature leakage** | Feature that contains label information | Audit feature importance; check for impossible correlations |
| **Data preprocessing leakage** | Scaling / imputation fit on full dataset before split | Fit scaler on train only, transform val/test separately |
| **Duplicate leakage** | Exact same row in train and test | Dedup by unique key before splitting |

**Best practice checklist:**
1. Split by time, not randomly, for any time-ordered data
2. Split by entity (user_id, session_id), not by row
3. Fit all preprocessing (mean, std, scaler, imputer) on **train only**
4. Keep a holdout test set that you evaluate exactly once
5. If using k-fold, ensure folds don't leak — `GroupKFold` or `TimeSeriesSplit`

**Trade-off:** Leakage-resistant splits often have higher variance (fewer entities in test, shorter time windows). Accept this — a slightly noisy but honest metric is infinitely more useful than a precise but inflated one.

---

## 5.2 Metrics That Matter

### Proxy Metrics vs Business Metrics

| | Proxy Metric | Business Metric |
|---|---|---|
| **Measurable** | Immediately (after each prediction) | Delayed (days to months) |
| **Owned by** | ML team | Business / product team |
| **Example** | AUC, precision@10, click-through rate | Revenue, retention, customer satisfaction |
| **Risk** | Optimizing the proxy can hurt the business metric | Too slow to use for day-to-day model decisions |

**Proxy metric failure examples:**

| Domain | Proxy Metric | What It Missed | Business Impact |
|---|---|---|---|
| **News ranking** | Click-through rate | Clickbait headlines attract clicks but reduce trust | Users churned (discovered months later) |
| **Loan approval** | Default prediction AUC | Approved safe loans that were too small to be profitable | Revenue flat despite better risk scoring |
| **Recommendations** | Precision@10 | Users clicked but didn't buy (surprised by irrelevant results) | Conversion rate dropped 5% |

**Best practice:** Track the gap between proxy and business metric. If proxy improves but business metric degrades, your proxy is wrong. Run periodic "business impact" analyses that connect model metrics to revenue, retention, or satisfaction.

### Threshold Selection

For probabilistic classifiers, the default 0.5 threshold is almost never optimal.

| Approach | How It Works | Trade-off |
|---|---|---|
| **Maximize F1** | Grid search threshold → highest F1 on validation | Good for balanced precision-recall needs |
| **Cost-based** | Minimize `cost = FP*cost_FP + FN*cost_FN` | Requires estimating costs of each error type |
| **Maximize profit** | Threshold that maximizes expected revenue | Needs business model (profit per TP, cost per FP) |
| **Target recall / precision** | Flip the constraint: fix recall at 90%, maximize precision | Good for regulated or SLA-bound systems |
| **ROC / PR curve analysis** | Pick threshold at the "elbow" of the curve | Reasonable default when costs are unknown |

**Real example — fraud detection:**
- Cost of FP: $10 (review time, customer friction)
- Cost of FN: $200 (fraudulent transaction value)
- At 0.5 threshold: 2% FP rate, 70% recall → cost = 0.02 × 10000 × $10 + 0.30 × 100 × $200 = $2K + $6K = $8K
- At 0.3 threshold: 5% FP rate, 90% recall → cost = 0.05 × 10000 × $10 + 0.10 × 100 × $200 = $5K + $2K = $7K
- **Cheaper to lower the threshold** even though FP rate increases, because FN cost dominates.

**Best practice:** Never use 0.5 as the default threshold. Compute optimal threshold from business costs on the validation set. Recompute when costs change or data distribution shifts.

### Metric Stability Over Time

A model's performance in production changes — monitor it.

| Pattern | What It Looks Like | Response |
|---|---|---|
| **Stable** | Metric within ±1% of offline eval | No action needed |
| **Gradual degradation** | Metric drops 0.1% per day | Plan retrain; investigate root cause |
| **Sudden drop** | Metric drops 5%+ in one day | Rollback or investigate immediately (pipeline change? concept drift?) |
| **Seasonal oscillation** | Metric cycles with day-of-week, holiday | Include seasonality features; retrain on recent data only |
| **Improvement** | Metric unexpectedly goes up | Investigate — could be data corruption that looks positive |

**To track stability:**
1. **Daily evaluation pipeline** — run offline eval on a fixed, fresh test set (e.g., last 7 days of data with known labels). Track metrics in a dashboard.
2. **Confidence intervals** — report 95% CI for every metric. If intervals overlap, the change isn't statistically significant.
3. **Segmented monitoring** — track metrics per segment (new users, mobile, high-value). A 1% overall drop might be a 20% drop in a critical segment.

**Best practice:** Automate weekly metric reports comparing current model vs previous model on a fixed, representative holdout set. P-value < 0.05 for regression? Investigate.

---

## 5.3 Online Evaluation

### A/B Testing for ML

A/B testing compares the current model (control) against the new candidate (treatment) on live traffic.

| Element | ML-Specific Consideration |
|---|---|
| **Randomization unit** | User, session, or request — affects network effects (if treatment changes recommendations for user A, and user B sees different content because of A's actions, standard A/B is invalid) |
| **Metric** | Primary: business metric (revenue, retention). Secondary: proxy metric (CTR, precision). Guardrail: latency, error rate |
| **Duration** | At least 1 full business cycle (1 week minimum to capture day-of-week effects) |
| **Sample size** | Power analysis: large enough to detect the minimum meaningful effect size |
| **Interference** | Two-sided markets (Uber, Airbnb) — treatment drivers affect control riders' experience. Use network A/B test designs or switchback tests |

**Best practices for ML A/B tests:**

1. **Pre-register the metric** — don't cherry-pick the metric that happened to improve. Decide primary + secondary metrics before the test starts.
2. **Hold out a validation set** — if you tuned model hyperparameters on last week's data, your A/B test isn't independent. Use a time period the model hasn't seen.
3. **Run both models offline first** — if model B doesn't beat model A on historical data, it won't win online.
4. **Monitor guardrails** — even if CTR improves, check if revenue or retention dropped. A "winning" model can still hurt the business.

### Shadow Deployments

Run the new model in parallel with the current model, but **use only the current model's predictions** for decisions. Log the new model's predictions and compare offline.

| Step | Action |
|---|---|
| 1 | Deploy new model (candidate) alongside current model (champion) |
| 2 | Route all traffic to champion; log champion predictions + candidate predictions |
| 3 | Compare: % of decisions where they disagree, direction of disagreement |
| 4 | If candidate disagrees with champion on > 10% of decisions, investigate manually |
| 5 | Only promote candidate after N days of shadow analysis with no surprises |

**What shadowing catches that offline eval misses:**
- Feature pipeline differences (candidate computed features differently than offline)
- Latency regressions (candidate is slower, causes timeout → fallback logic triggered)
- Distribution shift (candidate's test set was different from production distribution)
- Rare edge cases that weren't in the test set

**Trade-off:** Shadowing doubles serving infrastructure cost (two models, one serving). It's worth it for high-stakes models (fraud, medical, financial). For low-risk models (content ranking), skip shadow and go straight to canary.

### Canary Releases

Gradually shift traffic from champion → candidate, monitoring metrics at each step.

| Phase | Traffic % | Duration | Monitoring |
|---|---|---|---|
| **Shadow** | 0% (parallel, no decisions) | 1–7 days | Compare predictions, latency, errors |
| **Canary 1** | 5% | 1–3 days | Primary metric + guardrails |
| **Canary 2** | 25% | 1–3 days | Primary metric + guardrails + segment breakdown |
| **Canary 3** | 50% | 1–3 days | Full monitoring suite |
| **Full rollout** | 100% | — | Revert canary config, monitor for regression |

**Rollback triggers (automatic or manual):**

| Metric | Threshold | Action |
|---|---|---|
| Error rate | > 1% increase | Immediate rollback to 0% |
| Latency p99 | > 2× baseline | Rollback or fix infra |
| Primary metric | Statistically significant drop (p < 0.05) | Investigate and consider rollback |
| Guardrail metric | Any breach of minimum acceptable level | Rollback immediately |

**Real example:** A recommendation model went through canary: 5% → no issues → 25% → CTR increased 3% → 50% → noticed mobile users had 15% higher latency → fixed mobile inference → 100% rollout. The canary caught the mobile regression before it affected all users.

**Best practice:** Automate the canary process as much as possible. Manual approvals gate each phase, but monitoring, comparison, and rollback should be automatic. Infrastructure as code (Argo Rollouts, Flagger, Spinnaker).

---

## 5.4 Failure Analysis

### Slice-Based Evaluation

Overall metrics hide failures in important subgroups. Slice evaluation breaks down performance by meaningful segments.

| Slice Dimension | Why It Matters | Example |
|---|---|---|
| **Demographic** | Model may be biased against subgroups | Accuracy by age group, gender, region |
| **Behavioral** | Model may fail for power users vs new users | Precision@10 for users with < 5 sessions vs > 100 sessions |
| **Temporal** | Model degrades on certain days/times | Accuracy on weekends vs weekdays, holiday vs non-holiday |
| **Data quality** | Model fails when features are missing | Performance on rows with > 20% null features |
| **Product category** | Some items are inherently harder to predict | Recall on long-tail vs head items |
| **Traffic source** | Performance varies by acquisition channel | AUC for organic users vs paid ads users |

**Best practice:** Define slices before training, not after. Maintain a "slice catalog" with minimum performance thresholds per slice (e.g., recall > 0.8 on all demographic slices). Gate model releases on slice-level criteria, not just aggregate metrics.

**Trade-off:** More slices → more multiple comparison problems (some slice will show a significant difference by chance). Adjust for multiple testing (Bonferroni correction) or use a holdout slice analysis dataset.

### Error Analysis Workflows

A systematic workflow for understanding what the model gets wrong.

**1. Identify high-error slices:**
```
Overall error rate: 5%
Break down by:

Slice                    Error Rate  % of Data
──────────────────────────────────────────────
All users                5%          100%
New users (< 7 days)    18%          15%
Power users (> 100 d)    3%          30%
Mobile users             8%          40%
Desktop users            3%          60%
Weekend                  7%          28%
Weekday                  4%          72%
```

Worst slice: new users at 18% error rate. Focus investigation here.

**2. Deep-dive into the worst slice:**
- Plot feature distributions for errors vs correct predictions
- Are new users missing key features? (cold start → no history features)
- Are new users systematically different? (signup channel, device type)
- Does the model need a separate cold-start model?

**3. Categorize error types:**

| Error Type | Definition | Action |
|---|---|---|
| **Ambiguous label** | Human labelers disagree | Clean labels, use soft labels |
| **Missing feature** | Key signal not available at prediction time | Build a proxy feature |
| **Data error** | Wrong feature value due to pipeline bug | Fix pipeline, retrain |
| **Edge case** | Rare but systematic (e.g., holiday) | Augment training data with similar cases |
| **Model limitation** | Model can't learn the pattern (e.g., XOR with linear model) | Upgrade model complexity |

**Tools:** Error analysis dashboards (TensorFlow Model Analysis, What-If Tool, manual notebooks). Invest in tooling early — manual error analysis doesn't scale past one model.

### Identifying Systematic Failures

Systematic failures are not random — they follow a pattern. Finding the pattern is the goal.

**Method 1: Confusion matrix by segment**

For a fraud classifier:
```
              Actual Fraud    Actual Legit
Pred Fraud       600 (TP)       400 (FP)
Pred Legit       100 (FN)     8,900 (TN)
```

Slice: new accounts (< 30 days old):
```
              Actual Fraud    Actual Legit
Pred Fraud       400 (TP)       350 (FP)   ← 87.5% of all FPs
Pred Legit        20 (FN)       230 (TN)
```

**Insight:** 87.5% of false positives come from new accounts. The model doesn't have enough history to distinguish fraud from legitimate new users. **Fix:** Introduce a separate verification flow for new accounts, or add device fingerprinting / KYC features.

**Method 2: Prediction vs actual scatter plot**

Plot predicted value (x-axis) vs actual value (y-axis) for a regression model.

- **Ideal:** Points cluster around the diagonal
- **Systematic bias:** Points are above the diagonal at low values (model overpredicts) and below at high values (model underpredicts)
- **Fix:** Model is regressing toward the mean — add more discriminating features or use a different loss function

**Method 3: Residual by feature analysis**

For each feature, plot the model's residual (error) against feature value.

- If residuals are not centered at zero across feature bins, the model is systematically wrong for certain feature ranges
- Example: `order_value` feature — model under-predicts for orders > $500
- **Fix:** Add an interaction term or split encoding (`is_high_value`) to give the model more capacity on high-value orders

**Systematic failure anti-patterns:**

| Pattern | Symptom | Likely Cause |
|---|---|---|
| **Majority-class bias** | Model always predicts the most common class | Class imbalance, low threshold |
| **Feature absence failure** | Model fails when feature X is missing | Imputation default is misleading |
| **Temporal decay** | Recent data has higher error | Retraining frequency too low |
| **Segment blindness** | Model ignores a key segment (new users, international) | Segment not represented in training |
| **Threshold gaming** | Model outputs near 0.5 for most predictions | Poor calibration; features not discriminative |

**Best practice:** After every major model release, produce an error analysis report with:
1. Top 5 highest-error slices
2. Confusion matrix per slice
3. Feature distribution comparison (correct vs incorrect)
4. Action items for each systematic failure found
5. Track these findings over time — are the same failure patterns recurring?

If the same failure pattern appears across multiple model generations, the problem isn't the model — it's the data, the features, or the problem formulation itself.
