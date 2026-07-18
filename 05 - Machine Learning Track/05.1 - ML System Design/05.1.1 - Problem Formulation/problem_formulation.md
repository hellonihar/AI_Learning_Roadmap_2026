# Problem Formulation

Problem formulation is the process of translating a business need into a well-defined ML task. It is the single most important step in the ML lifecycle — a perfectly executed model solving the wrong problem delivers negative value. Formulation forces you to answer: What business outcome are we driving? Can ML actually help? How do we measure success? What constraints must the system operate within?

Skipping problem formulation is the most common reason ML projects fail. Teams jump straight to data collection and model training without clarifying the objective, and end up optimizing a proxy metric that doesn't move the business needle. A day spent framing the problem correctly — defining the objective, mapping constraints, choosing the task type, setting the system boundary — saves months of building the wrong solution. The quality of the formulation determines the ceiling on the value the ML system can deliver.

## 1.1 Business Objective → ML Objective

### Business Goals vs ML Proxy Metrics

Business goals are rarely directly optimizable by ML. You translate them into **proxy metrics** the model can actually optimize.

| Business Goal | ML Proxy Metric | Why Not Direct? |
|---|---|---|
| Increase revenue | Predicted purchase probability | Revenue depends on pricing, inventory, UX — not just ML |
| Reduce churn | Predicted 30-day cancellation probability | Churn is a delayed signal; proxy lets you act early |
| Improve search quality | NDCG on relevance judgments | "User found what they needed" is subjective; NDCG is a measurable proxy |
| Moderate content | Precision/recall on toxic content labels | "Toxic" is defined by human labels, not an objective property |

**Key rule:** A proxy metric that you can't later connect back to the real business metric is dangerous. Always track the gap.

### When ML Is Not the Right Solution

- **Simple rules work** — `if x > threshold` is cheaper, faster, and easier to debug
- **No data or garbage data** — ML with 50 samples will fail; use heuristics until you have enough
- **Perfect accuracy required** — ML is probabilistic; use it as a screener + human verifier
- **Problem changes too fast** — fraud patterns shift weekly; without automated retraining, the model decays
- **Interpretability legally mandated** — GDPR, ECOA may require exact "why" for each decision; black-box models won't work
- **Cost of failure is catastrophic** — self-driving cars need deterministic safety layers, not pure ML

### Success Metrics vs Optimization Metrics

| | Optimization Metric | Success Metric |
|---|---|---|
| **Purpose** | What the model minimizes during training | What the business cares about long-term |
| **Example** | Cross-entropy loss | Customer retention rate |
| **Feedback** | Available per batch (immediate) | Available quarterly (delayed) |
| **Owned by** | ML engineer | Product/business stakeholder |

**Common failure:** Optimizing the proxy (RMSE) while the business metric (revenue) degrades. Monitor both.

### Cost of False Positives vs False Negatives

The **asymmetry** of error costs drives model design:

| Domain | False Positive Cost | False Negative Cost | Tilt |
|---|---|---|---|
| Spam detection | Missed email (low) | Inbox full of spam (medium) | Tolerate FP, minimize FN |
| Fraud detection | Block legitimate purchase (high — angry customer) | Approve fraudulent transaction (high — financial loss) | Balance carefully |
| Medical diagnosis | Unnecessary biopsy (medium — cost + anxiety) | Missed cancer (very high — life) | Tolerate FP, minimize FN |
| Recruiting | Waste time interviewing weak candidate (low) | Miss top candidate who goes to competitor (high) | Tolerate FP, minimize FN |

**Fix:** Adjust decision threshold, weight loss function, or use cost-sensitive learning.

---

## 1.2 Framing the ML Task

### Classification vs Regression vs Ranking vs Generation

| Task | Output | Examples |
|---|---|---|
| **Regression** | Continuous number | Price, temperature, delivery time |
| **Classification** | Discrete category | Spam/not spam, digit 0–9, disease stage |
| **Ranking** | Ordered list | Search results, feed, top-K candidates |
| **Generation** | Structured output (text, image, audio) | Caption, translation, code completion |

**Real systems often combine them:**
- A fraud system might classify (fraud probability) + rank (priority queue for review) + regress (dollar amount at risk)
- A search engine ranks results but could also classify query intent and regenerate snippets

### Batch Prediction vs Online Prediction

| | Batch | Online (Real-Time) |
|---|---|---|
| **Latency** | Hours / minutes | Milliseconds |
| **Compute** | Process all at once (cheap per record) | Serve one request (needs fast inference) |
| **Model update** | Retrain daily/weekly | Can update on-the-fly |
| **Example** | Nightly churn scoring → email campaign | Real-time fraud scoring at checkout |
| **Infra** | Spark, Airflow | REST API, gRPC, streaming |

**Hybrid:** Generate batch scores nightly, but fetch them in real-time from a low-latency cache. Common for recommendation systems.

### Single Model vs Multi-Stage Pipeline

| | Single Model | Multi-Stage Pipeline |
|---|---|---|
| **Pros** | Simple to train, deploy, debug | Each stage does one thing well; can swap stages independently |
| **Cons** | Must handle all complexity alone | Latency adds up; error propagates between stages |
| **When** | Simple task, one data source, tight latency | Complex task (retrieval → ranking → reranking → blending) |

**Typical multi-stage:**
```
Candidate retrieval (lightweight, recall-focused)
  → First-stage ranking (hundreds → dozens)
  → Second-stage ranking (dozens → top-K)
  → Business logic + diversity blending
  → Final results
```

### Human-in-the-Loop vs Fully Automated

| | HITL | Automated |
|---|---|---|
| **Latency** | Hours/days (human review) | Milliseconds |
| **Scale** | Limited by human capacity | Unlimited |
| **Cost** | High per decision | Low per decision |
| **Accuracy** | High (human judgment) | Variable (but consistent) |
| **Best for** | Edge cases, ambiguous decisions, high-cost errors | High-volume, well-defined, low-cost decisions |

**Tiered approach:**
1. ML confidently correct (score > 0.95) → auto-approve
2. ML uncertain (0.7–0.95) → human review
3. ML confident it's wrong (< 0.7) → auto-reject or route to specialist

---

## 1.3 Constraints & Assumptions

### Latency, Throughput, Cost, Privacy

| Constraint | Impact on ML System |
|---|---|
| **Latency** (p99 < 100ms) | No deep ensembles, no multi-pass inference, model must fit on one GPU/CPU |
| **Throughput** (10K QPS) | Need horizontal scaling, model quantization, or batch serving |
| **Cost** ($/prediction) | Trade-off: big model (expensive inference) vs small model (less accurate) |
| **Privacy** (PII, GDPR) | Data must be anonymized; might need on-device inference or differential privacy |

**Real example:** A real-time fraud model at a payment processor (p99 < 50ms, > 5K QPS, no PII logging) — use a gradient-boosted tree (~100 features), not a 7B LLM.

### Data Availability Assumptions

- **Labeled data exists?** If not, plan for labeling pipeline (costly and slow)
- **How much?** 100 labeled samples → use heuristics; 10K+ → ML feasible
- **Distribution matches production?** If training data is from 2023 and production is 2026, beware of drift
- **Feature availability at inference?** Feature engineering must assume what's available at prediction time, not just training

### Label Delay and Label Quality

- **Label delay:** A credit default model might take 12 months to know if a loan was good. Your model trains on stale truth.
  - **Mitigation:** Use surrogate labels (early signals) or train on cohorts with known outcomes
- **Label quality:** Human labelers agree only 80% of the time on subjective tasks (toxicity, sentiment)
  - **Mitigation:** Use majority vote, Dawid-Skene, or active learning to improve labels
- **Label noise:** 10% random label flips can degrade model performance significantly
  - **Mitigation:** Use noise-robust loss functions (MAE vs cross-entropy) or clean labels iteratively

### What Happens When Assumptions Break?

| Assumption | What Breaks It | Consequence |
|---|---|---|
| Training == production distribution | COVID-19 changes user behavior | Concept drift → model degrades |
| Features always available | Downstream API fails | Prediction pipeline crashes or falls back |
| Labels arrive within 24h | Human reviewers go on strike | Retraining pipeline stalls |
| Latency stays under 100ms | Traffic spikes 10x | Timeouts, queue building, degraded UX |

**Design defensively:** Always have fallbacks, monitors, and alerting for assumption violations.

---

## 1.4 Defining the System Boundary

### What Is Inside the ML System?

The ML system owns:
- **Inference** — generating predictions from features
- **Feature computation** — transforming raw data into model inputs
- **Model lifecycle** — training, evaluation, serving, retraining
- **Prediction storage** — caching results for low-latency access

**Not inside:**
- User authentication, logging, UI rendering, payment processing, DNS routing

### What Is Handled by Rules or Heuristics?

Rules complement ML. Common splits:

| Decision | ML | Rules |
|---|---|---|
| Fraud detection | Predict fraud probability | Block if amount > $10K AND new user (hard rule) |
| Content recommendation | Rank candidates | Filter out NSFW content, block blocked users |
| Email classification | Predict spam score | Auto-delete from known spam domains (blocklist) |
| Loan approval | Predict default probability | Reject if credit score < 500 (regulatory minimum) |

**Rule of thumb:** Use rules for things you know for certain (static, deterministic, regulated). Use ML for things you need to infer (probabilistic, pattern-based).

### What Feedback Signals Exist?

Feedback is how the system improves. Design for it from day one.

| Signal | Source | Delay | Use |
|---|---|---|---|
| **Explicit feedback** | User ratings, thumbs up/down | Seconds | Direct training signal |
| **Implicit feedback** | Clicks, dwell time, scroll depth | Seconds | Proxy for engagement |
| **Business outcome** | Purchase, churn, retention | Days–months | Ground truth for offline eval |
| **Label from HITL** | Human reviewer decisions | Minutes–hours | Golden dataset for retraining |
| **Model metrics** | Latency, error rate, cache hit rate | Real-time | System health, not prediction quality |

**Critical gap:** If no feedback signal exists (e.g., a one-shot prediction with no observable outcome), you cannot improve the model after deployment. Plan for A/B testing or user studies to generate feedback.

---

## Summary

| Stage | Key Questions |
|---|---|
| **Business → ML objective** | What business metric matters? What proxy metric can we optimize? What do FP/FN cost? |
| **Framing the task** | Regression, classification, ranking, or generation? Batch or online? Single model or pipeline? How much human involvement? |
| **Constraints & assumptions** | Latency, cost, privacy limits? Do we have data? What if assumptions fail? |
| **System boundary** | What does ML own vs rules? What feedback signals can we collect to improve? |
