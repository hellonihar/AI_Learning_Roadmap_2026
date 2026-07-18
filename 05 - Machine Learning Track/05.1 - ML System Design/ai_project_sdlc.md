# AI Project SDLC

The AI project Software Development Lifecycle (SDLC) is a structured framework that takes an ML system from idea to production to retirement. Unlike traditional software SDLC, AI projects have **non-deterministic behavior** (models can fail silently), **data dependencies** (model quality depends on data quality), and **ongoing monitoring requirements** (models decay even when code doesn't change).

This document defines the full lifecycle, maps each phase to the detailed guides in this section, and specifies the gates, roles, and artifacts required at every stage.

---

## Lifecycle Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  0.  Discovery & Feasibility                                        │
│  ─────────────────────────────                                      │
│  Is ML the right solution? Is the data available?                   │
└───────────────────────────┬──────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  1.  Problem Formulation                         → 05.1.1           │
│  ──────────────────────────                                          │
│  Business objective → ML task. Constraints, success metrics.         │
└───────────────────────────┬──────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  2.  Data System Design                          → 05.1.2           │
│  ──────────────────────────                                          │
│  Data sources, collection, storage, validation, drift monitoring.    │
└───────────────────────────┬──────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  3.  Feature System Design                      → 05.1.3           │
│  ──────────────────────────                                          │
│  Feature definition, engineering, store, training-serving consistency.│
└───────────────────────────┬──────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  4.  Modeling Strategy                           → 05.1.4           │
│  ──────────────────────────                                          │
│  Model selection, training, interpretability, lifecycle management.  │
└───────────────────────────┬──────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  5.  Evaluation Strategy                         → 05.1.5           │
│  ──────────────────────────                                          │
│  Offline eval, online A/B, failure analysis, slice-based eval.       │
└───────────────────────────┬──────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  6.  Serving                                    → 05.1.6           │
│  ──────────────────────────                                          │
│  Architecture, latency, deployment patterns, reliability, cost.      │
└───────────────────────────┬──────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  7.  Feedback & Monitoring                       → 05.1.7           │
│  ──────────────────────────                                          │
│  Data/prediction/label monitoring, drift detection, retraining.      │
└───────────────────────────┬──────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  8.  Governance & Ethics                         → 05.1.8           │
│  ──────────────────────────                                          │
│  Ongoing bias audits, model cards, compliance, responsible ML.       │
└───────────────────────────┬──────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────────────┐
│  9.  Retirement                                                     │
│  ──────────────────────────                                          │
│  Decommission, archive artifacts, migrate consumers.                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Discovery & Feasibility

**Goal:** Determine whether ML is the right solution before committing resources.

| Activity | Key Questions | Artifacts |
|---|---|---|
| Business need assessment | What problem are we solving? What is the expected business impact? | Business case document |
| ML feasibility analysis | Do we have enough data? Is the problem learnable? What's the simplest baseline? | Feasibility report |
| Alternative evaluation | Can we solve this with rules/heuristics? Is an ML approach worth the complexity? | Decision: ML vs no-ML |
| Risk assessment | What's the cost of wrong predictions? What are the privacy/regulatory risks? | Risk register |

**Gate 0 → 1:** Go ahead if (a) ML is feasible, (b) data exists or can be collected, (c) expected ROI justifies investment, (d) risks are acceptable.

**Duration:** 1–4 weeks.

---

## Phase 1: Problem Formulation

**Doc:** `05.1.1 - Problem Formulation/problem_formulation.md`

**Goal:** Translate the business objective into a well-defined ML task with clear success criteria.

| Activity | References |
|---|---|
| Define business objective → ML proxy metrics | §1.1 — Business goals vs ML proxy metrics, when ML is not the right solution |
| Frame the ML task | §1.2 — Classification vs regression vs ranking vs generation, batch vs online |
| Document constraints and assumptions | §1.3 — Latency, throughput, cost, privacy, data availability |
| Define the system boundary | §1.4 — What's inside ML vs rules, feedback signals |

**Artifacts:**
- Problem formulation document (business objective, ML task, success metrics, FP/FN costs)
- Constraint inventory (latency, throughput, cost, privacy limits)
- System boundary diagram

**Gate 1 → 2:** Problem formulation reviewed and approved by business + technical stakeholders. Success metrics are unambiguous and measurable.

---

## Phase 2: Data System Design

**Doc:** `05.1.2 - Data System Design/data_system_design.md`

**Goal:** Design the infrastructure for collecting, storing, validating, and serving data for ML.

| Activity | References |
|---|---|
| Identify and classify data sources | §2.1 — First-party vs third-party, static vs streaming, training vs serving |
| Design logging and collection schema | §2.2 — What to log, schema design, handling delayed labels, avoiding leakage |
| Choose storage architecture | §2.3 — Data lakes vs warehouses, immutability, versioning, reproducibility |
| Implement data validation | §2.4 — Missing values, outliers, validation checks, distribution monitoring |
| Set up drift detection | §2.5 — Covariate vs label vs concept drift, detection strategies |

**Artifacts:**
- Data source inventory
- Logging schema specification
- Data storage architecture (lake vs warehouse, partitions, versioning)
- Data validation pipeline (Great Expectations, Pandera)
- Drift monitoring dashboard

**Gate 2 → 3:** Data pipeline is producing reliable, validated data. Data quality metrics are within acceptable range. Drift detection is operational.

---

## Phase 3: Feature System Design

**Doc:** `05.1.3 - Feature System Design/feature_system_design.md`

**Goal:** Design and build the feature computation system that transforms raw data into model inputs, consistently across training and serving.

| Activity | References |
|---|---|
| Define feature catalog | §3.1 — Static vs dynamic, batch vs real-time, aggregations and windows |
| Design feature engineering system | §3.2 — Reuse, coupling problems, ownership and governance |
| Eliminate training-serving skew | §3.3 — Time-travel correctness, same-code rule, prevention patterns |
| Evaluate feature store need | §3.4 — Online vs offline stores, when to use / not use |
| Optimize freshness vs cost | §3.5 — Freshness-latency tradeoffs, expensive features vs proxies, caching |

**Artifacts:**
- Feature registry / catalog (all features with definitions, owners, freshness)
- Feature computation code (single code path for training and serving)
- Feature store configuration (if applicable)
- Training-serving skew validation report

**Gate 3 → 4:** Feature computation is validated to be identical in training and serving. Features are documented in a registry. No known training-serving skew sources remain.

---

## Phase 4: Modeling Strategy

**Doc:** `05.1.4 - Modeling Strategy/modeling_strategy.md`

**Goal:** Select, train, and manage the model lifecycle with the right balance of complexity, performance, and maintainability.

| Activity | References |
|---|---|
| Select model family | §4.1 — Simple vs complex, when DL is overkill, ensemble vs single |
| Design training strategy | §4.2 — Offline vs continual training, retraining frequency, cold start |
| Handle data imbalance | §4.3 — Sampling, loss functions, metric selection |
| Plan for interpretability | §4.4 — When required, debugging, interpretable proxies |
| Set up model lifecycle | §4.5 — Versioning, backward compatibility, rollbacks |

**Artifacts:**
- Model selection document (candidates, comparison, chosen approach with rationale)
- Training pipeline (reproducible, versioned)
- Model manifest (hyperparameters, data version, code version, metrics)
- Model card (purpose, limitations, evaluation, fairness)
- Rollback plan

**Gate 4 → 5:** Model achieves minimum performance threshold on validation set. Interpretability requirements are satisfied (or documented exception). Model card is complete.

---

## Phase 5: Evaluation Strategy

**Doc:** `05.1.5 - Evaluation Strategy/evaluation_strategy.md`

**Goal:** Rigorously evaluate the model before and during deployment to ensure it works correctly and safely.

| Activity | References |
|---|---|
| Design offline evaluation | §5.1 — Time-aware splits, leakage-resistant evaluation |
| Choose metrics that matter | §5.2 — Proxy vs business metrics, threshold selection, stability monitoring |
| Plan online evaluation | §5.3 — A/B testing, shadow deployments, canary releases |
| Conduct failure analysis | §5.4 — Slice-based evaluation, error analysis, systematic failures |

**Artifacts:**
- Offline evaluation report (metrics on held-out test set with confidence intervals)
- Slice-based evaluation report (performance by segment)
- Shadow deployment analysis
- Canary release plan with rollback criteria
- Failure analysis report

**Gate 5 → 6:** Model passes all offline evaluation gates (overall + per-slice). Shadow deployment shows no unexpected behavior. Canary rollback criteria are defined and automated.

---

## Phase 6: Serving

**Doc:** `05.1.6 - Serving/serving.md`

**Goal:** Deploy the model to production with the right architecture, latency, reliability, and cost profile.

| Activity | References |
|---|---|
| Choose serving architecture | §6.1 — Batch vs online vs streaming |
| Meet latency and throughput targets | §6.2 — Model size optimization, hardware selection, caching |
| Select deployment pattern | §6.3 — Blue-green, canary, shadow |
| Ensure system reliability | §6.4 — Fallback strategies, graceful degradation, SLIs/SLOs |
| Optimize inference cost | §6.5 — Compression, traffic shaping, batch sizing |

**Artifacts:**
- Serving architecture diagram
- Latency budget breakdown
- Deployment pipeline (CI/CD for model rollout)
- Fallback chain specification
- Cost projection (per-prediction + monthly)
- SLI/SLO dashboard

**Gate 6 → 7:** Model is serving in production. Latency, throughput, and error rate SLIs are within SLO targets. Fallback chain is tested and operational. Cost is within budget.

---

## Phase 7: Feedback & Monitoring

**Doc:** `05.1.7 - Feedback and Monitoring/feedback_and_monitoring.md`

**Goal:** Continuously monitor the model in production and improve it over time through feedback loops.

| Activity | References |
|---|---|
| Monitor data, predictions, and labels | §7.1 — Distribution monitoring, label quality |
| Track online vs offline metrics | §7.2 — Alert fatigue, meaningful thresholds |
| Design feedback loops | §7.3 — Explicit vs implicit feedback, delay, bias |
| Set up retraining triggers | §7.4 — Scheduled, event-based, human-triggered |
| Implement governance and ethics | §7.5 — Bias detection, auditing, model cards |

**Artifacts:**
- Monitoring dashboard (data drift, prediction drift, performance metrics)
- Alert configuration (tiered: info, warning, critical)
- Feedback collection pipeline
- Retraining trigger configuration
- Regular audit report (bias, fairness, performance)

**No gate to next phase** — monitoring is continuous. The system loops back to retraining as needed.

---

## Phase 8: Governance & Ethics (Ongoing)

**Doc:** `05.1.8 - Governance and Ethics/governance_and_ethics.md`

**Goal:** Ensure the model remains fair, transparent, and compliant throughout its lifetime.

| Activity | Frequency | Artifacts |
|---|---|---|
| Bias audit (slice-based evaluation) | Quarterly or per-major-release | Fairness report |
| Model card review and update | Per-retraining | Updated model card |
| Regulatory compliance check | Per-regulatory-change | Compliance attestation |
| Audit log review | Monthly | Audit trail integrity check |
| Stakeholder communication | Quarterly | Model performance and impact summary |

**Not a gate** — governance is an ongoing practice that spans all phases.

---

## Phase 9: Retirement

**Goal:** Gracefully decommission a model that is no longer needed or has been replaced.

| Activity | Details |
|---|---|
| Notify consumers | Inform all downstream systems and teams with N-days notice (30–90 days typical) |
| Migrate traffic | Route traffic to replacement model or fallback |
| Verify no remaining dependencies | Check logs for requests hitting the old endpoint |
| Archive artifacts | Store final model version, manifest, training data version, evaluation reports in long-term storage |
| Update documentation | Mark model as retired in registry; close model card |
| Remove serving infrastructure | Decommission endpoints, free resources |

**Artifacts:**
- Retirement notice and migration plan
- Archived model artifacts (immutable)
- Final model card with end-of-life date

---

## Stage Gates Summary

| # | Phase | Gate Condition | Reviewer |
|---|---|---|---|
| 0→1 | Discovery → Formulation | Problem is ML-feasible, ROI positive, risks acceptable | PM + Tech Lead |
| 1→2 | Formulation → Data | Problem spec approved, metrics defined, constraints documented | Stakeholders + Eng |
| 2→3 | Data → Features | Data pipeline reliable, quality metrics passing, drift detection ready | Data Eng + ML Eng |
| 3→4 | Features → Modeling | Feature registry complete, no training-serving skew | ML Eng |
| 4→5 | Modeling → Evaluation | Model meets accuracy threshold, model card complete | ML Eng + Reviewer |
| 5→6 | Evaluation → Serving | Offline + shadow eval passed, canary rollback criteria defined | ML Eng + Ops |
| 6→7 | Serving → Monitoring | Latency, throughput, error rate SLOs met, fallback tested | Ops + ML Eng |
| 7→8 | Monitoring | Continuous — no gate (loops back to retrain) | — |
| 8→9 | Governance | Ongoing — no gate (runs in parallel) | — |
| 9→R | Retirement | Model replaced or no longer needed | PM + Eng |

---

## Roles & Responsibilities

| Role | Responsibilities Across SDLC |
|---|---|
| **Product Manager** | Business objective definition, success metrics, stakeholder communication, gate approval |
| **Data Engineer** | Data pipeline, storage, validation, monitoring infrastructure |
| **ML Engineer** | Problem formulation, feature engineering, modeling, evaluation, serving, monitoring |
| **MLOps Engineer** | Deployment pipeline, serving infrastructure, reliability, cost optimization |
| **Data Scientist** | Exploratory analysis, feature discovery, model evaluation, failure analysis |
| **Legal / Compliance** | Privacy, regulatory requirements, bias auditing, model card review |
| **Software Engineer** | API integration, system boundary implementation, fallback logic |

---

## Handoff Artifacts Between Phases

Each phase produces artifacts consumed by the next:

| Phase | Produces | Consumed By |
|---|---|---|
| 0 — Discovery | Feasibility report, ML/no-ML decision | Phase 1 |
| 1 — Formulation | Problem spec, success metrics, constraints | Phase 2, 4 |
| 2 — Data | Data pipeline, validation checks, data catalog | Phase 3 |
| 3 — Features | Feature registry, feature computation code | Phase 4 |
| 4 — Modeling | Trained model, model card, model manifest | Phase 5 |
| 5 — Evaluation | Eval report, shadow analysis, canary plan | Phase 6 |
| 6 — Serving | Deployed model, SLO dashboard, fallback chain | Phase 7 |
| 7 — Monitoring | Drift alerts, retraining triggers, audit logs | Loop back to Phase 2/4 |
| 8 — Governance | Bias reports, compliance docs, model cards | All phases (ongoing) |
| 9 — Retirement | Archived artifacts, migration plan | — |
