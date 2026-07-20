# Case Study: AI-Powered Claims Intelligence & Fraud Detection for Insurance

## 1. Business Problem & Opportunity

### The Scenario
A national P&C (Property & Casualty) insurer with $8B annual premium processes **1.2M claims per year** across auto, home, and commercial lines. The claims operation employs 2,500 adjusters, examiners, and investigators. Annual claim leakage (overpayments + fraud) is estimated at **$420M**.

Key pain points:
- **Fraud detection lag**: 80% of fraudulent claims are identified post-payment; recovery rate is only 30%
- **Manual triage**: Every claim is manually reviewed by a human adjuster within 48h, costing $85/triage and delaying legitimate payouts
- **Inconsistent settlement**: Settlement amounts vary up to 35% for identical claim profiles depending on adjuster experience
- **Document processing bottleneck**: 65% of adjuster time spent on data entry from PDFs, photos, and handwritten notes
- **Litigation exposure**: 12% of denied claims result in lawsuits; insurer loses 55% of those cases

### Business Case

| Metric | Current | Target | Annual Impact |
|--------|---------|--------|---------------|
| Fraud detection rate (pre-payment) | 20% | 65% | -$180M fraud loss |
| Claim triage time | 48h | 15min (auto), 4h (escalated) | -$40M ops cost |
| Claims process cycle time | 21 days | 5 days (auto), 12 days (complex) | +$25M CSAT |
| Settlement consistency | 35% variance | < 8% variance | -$60M leakage |
| Adjuster productivity (claims/case) | 120/yr | 400/yr | -$50M FTE cost |
| Litigation rate on denials | 12% | < 3% | -$15M legal cost |
| **Total addressable benefit** | | | **$370M/yr** |

### Success Criteria
- Pre-payment fraud detection rate > 65% with < 5% false positive rate
- Auto-adjudication of 40%+ of simple claims (straight-through processing)
- Claims cycle time reduced by 60%+
- Settlement consistency (coefficient of variation) < 8%
- Net Promoter Score for claims experience > 50 (currently -10)
- **ROI 6:1 within 24 months** (project cost $25M, expected annual benefit $150M+)

---

## 2. Problem Framing & AI Solution Assessment

### Is AI the Right Approach?

| Factor | Assessment |
|--------|-----------|
| Problem structure | Multi-faceted: classification (fraud/legit), regression (reserve estimation), NLP (document understanding), graph (network analysis) |
| Data availability | 15 years of claims, policies, payments, customer data, external data — robust |
| Decision complexity | High — thousands of claim attributes, evolving fraud patterns, regulatory constraints |
| Business readiness | EVP of Claims is sponsor, existing data lake, claims modernization program underway |
| Alternative | Rule-based systems (legacy) catch only 20% of fraud; manual processes don't scale |

**Decision**: AI is essential. The combinatorial complexity of fraud patterns, the volume of unstructured data (photos, PDFs, notes), and the need for real-time triage at scale cannot be addressed with traditional rules.

### AI Use Case Canvas

| Element | Definition |
|---------|-----------|
| Use case name | AI Claims Intelligence & Fraud Detection Platform |
| Business objective | Reduce fraud loss by 60%+, cut cycle time by 60%+, improve settlement consistency |
| Primary stakeholders | Chief Claims Officer, VP Fraud, Head of Operations, General Counsel |
| AI tasks | Fraud classification, claim triage routing, reserve estimation, document extraction, network analysis |
| Inputs | Claim forms, policy data, photos, adjuster notes, third-party data, historical claims |
| Outputs | Fraud score (0–100), triage category (auto/simple/complex), recommended reserve, suspicious network alerts |
| Decisions powered | Claim routing, payment approval, investigation assignment, reserve setting |
| Success metrics | Fraud detection rate, FPR, cycle time, settlement variance, NPS |
| Fallback | Rule-based triage + manual review if ML pipeline degrades |

---

## 3. Data Strategy & Engineering

### Source Systems & Data Inventory

| Source | Data | Volume | Freshness | Method |
|--------|------|--------|-----------|--------|
| Claims Management System | Claim details, status, payments, reserves | 1.2M claims + updates | Real-time (events) | CDC → Kafka |
| Policy System | Policy details, coverage, endorsements, history | 4M active policies | Real-time | CDC → Kafka |
| Customer 360 / CRM | Customer profile, tenure, prior claims, satisfaction | 8M customers | T+1 batch | DB snapshot |
| Document Management | PDFs, photos, handwritten notes (FNOL, estimates) | 7M documents/year | Event-driven | Object storage trigger |
| Third-party data | ISO ClaimSearch, vehicle history, weather, credit | 100+ sources | Varies (API/batch) | API gateway + S3 |
| Adjuster notes & actions | Free-text notes, decisions, override history | 50M entries | Real-time | Kafka |
| Payment system | Payment transactions, vendor details, check images | 3M payments/year | Real-time | CDC → Kafka |

### Data Pipeline Architecture

```
                   ┌──────────────────────────────────────────────────────┐
                   │                    Ingestion Layer                    │
                   ├─────────┬─────────┬──────────┬──────────┬────────────┤
                   │ CDC     │ API     │ File     │ Batch    │ Streaming  │
                   │ (Kafka) │ Gateway │ Upload   │ (Airflow)│ (Kafka)    │
                   │ Connect)│ (Kong)  │ (S3/Blob)│          │            │
                   └────┬────┴────┬────┴────┬─────┴────┬─────┴─────┬──────┘
                        │         │         │          │           │
                        ▼         ▼         ▼          ▼           ▼
                   ┌──────────────────────────────────────────────────────┐
                   │                   Bronze Layer (raw)                  │
                   │  Parquet partitioned by date + source               │
                   └────────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
                   ┌──────────────────────────────────────────────────────┐
                   │              Silver Layer (cleaned, enriched)         │
                   │  • Deduplication (claim_id, policy_id)               │
                   │  • Schema validation & type coercion                  │
                   │  • Data quality scoring per record                    │
                   └────────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
                   ┌──────────────────────────────────────────────────────┐
                   │               Gold Layer (features + ML ready)        │
                   │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │
                   │  │ Tabular  │ │ Text     │ │ Graph    │             │
                   │  │ Features │ │ Embeddings│ │ Nodes/   │             │
                   │  │ (Spark)  │ │ (GPU pod) │ │ Edges    │             │
                   │  └──────────┘ └──────────┘ └──────────┘             │
                   └────────────────────────┬─────────────────────────────┘
                                            │
                                            ▼
                   ┌──────────────────────────────────────────────────────┐
                   │              Feature Store (online + offline)         │
                   │  Online: Redis (sub-ms serving)                       │
                   │  Offline: S3 Parquet + Delta Lake (training)         │
                   └──────────────────────────────────────────────────────┘
```

### Feature Engineering

**Claim-level features** (tabular):
- Claim attributes: type, cause, severity, location, time-of-day, day-of-week
- Claimant history: prior claims count, frequency, amount, time since last claim
- Policy tenure, coverage type, premium amount, deductible
- Time-to-report (claim occurrence to filing)
- Injury vs. property-only indicator

**Text features** (NLP):
- TF-IDF + BERT embeddings of adjuster notes
- Sentiment analysis of claimant communication
- Topic modeling of claim description (unsupervised clusters)
- Semantic similarity to past fraudulent claims
- Named entity extraction (body shops, medical providers, attorneys)

**Graph features** (network analysis):
- Degree centrality of claimant in provider network
- Shared address/phone/IP between claimants
- Shared provider across multiple claimants
- Ring density (multiple claims sharing overlapping attributes)
- Bipartite graph distances (claimant ↔ provider ↔ adjuster)

**Behavioral features** (temporal):
- Claimant filing patterns (night/weekend filings correlate with fraud)
- Provider billing velocity (sudden spike in referrals)
- Communication frequency spikes
- Premium change timing relative to claim

**External features**:
- ISO ClaimSearch index match score
- Vehicle valuation (NADA/Kelley Blue Book)
- Weather severity at time of loss (for CAT claims)
- Crime stats in claim location (for theft claims)
- Economic indicators (unemployment correlates with fraud uptick)

### Data Quality Framework

| Check | Tool | Action on Failure |
|-------|------|-------------------|
| Claim ID uniqueness | Great Expectations | Reject batch, deduplicate at source |
| Required fields present | Schema validation | Route to manual data correction queue |
| Policy active at time of loss | Cross-system validation | Flag for investigation hold |
| Amount within expected range | Statistical bounds | Flag as anomaly for review |
| Document readable (OCR confidence > 70%) | Tesseract / DocAI | Request re-upload from claimant |
| Feature drift (PSI > 0.15 for 3+ features) | Evidently | Alert data team, retrain models |

---

## 4. Model Development Lifecycle

### The AI System: Multi-Model Ensemble Architecture

```
                           ┌──────────────────────────────┐
                           │      Claim Ingestion Event    │
                           └─────────────┬────────────────┘
                                         │
                                         ▼
                           ┌──────────────────────────────┐
                           │    Document Intelligence      │
                           │  ┌──────────┐ ┌────────────┐  │
                           │  │ OCR +    │ │ Photo       │  │
                           │  │ Document │ │ Damage      │  │
                           │  │ Parsing  │ │ Assessment  │  │
                           │  │ (LayoutLM)│ │ (ViT/CNN)  │  │
                           │  └──────────┘ └────────────┘  │
                           └─────────────┬────────────────┘
                                         │
                                         ▼
                           ┌──────────────────────────────┐
                           │        Triage Classifier      │
                           │  (XGBoost: Simple / Complex / │
                           │   Investigate / Auto-Pay)     │
                           └──────┬───────────┬───────────┘
                                  │           │
                    ┌─────────────┘           └─────────────┐
                    ▼                                       ▼
           ┌──────────────────┐                  ┌──────────────────┐
           │ Fraud Detection  │                  │ Reserve Estimation│
           │ Ensemble         │                  │                  │
           │ ┌──────────────┐ │                  │ ┌──────────────┐ │
           │ │ GNN (network)│ │                  │ │ Gradient      │ │
           │ │ TabNet       │ │                  │ │ Boosting +    │ │
           │ │ (tab feat)   │ │                  │ │ Quantile Reg  │ │
           │ │ BERT (text)  │ │                  │ │ (3 models)    │ │
           │ │ Fusion       │ │                  │ └──────────────┘ │
           │ └──────────────┘ │                  └──────────────────┘
           └──────────────────┘
                    │                                       │
                    └──────────────┬────────────────────────┘
                                   ▼
                    ┌────────────────────────────────────┐
                    │         Decision Engine             │
                    │  Combines scores, reserves, rules   │
                    │  1. Auto-pay (score > 0.95 legit,   │
                    │     reserve < $5k)                  │
                    │  2. Simple review (human verify)    │
                    │  3. Complex review (senior adjuster) │
                    │  4. Investigate (fraud > 0.8)       │
                    └─────────────┬──────────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────────────┐
                    │  Outcome Feedback Loop              │
                    │  (human decisions → retrain data)   │
                    └────────────────────────────────────┘
```

### Model Components Detail

| Model | Task | Algorithm | Training Data | Features |
|-------|------|-----------|---------------|----------|
| Document Parser | Extract fields from PDFs/photos | LayoutLM v3 + Tesseract | 500k labeled documents | Image + text |
| Damage Assessment | Estimate repair cost from photos | EfficientNet + ViT | 200k labeled photos | Images |
| Triage Classifier | Route claim to correct workflow | XGBoost (multi-class) | 800k historical claims | 120 tabular features |
| Fraud Classifier (Tabular) | Flag fraudulent claims | TabNet + XGBoost ensemble | 300k labeled (15% fraud) | 95 tabular features |
| Fraud Graph (GNN) | Network fraud detection | GraphSAGE + GAT | 50M node graph | Graph structure + node attrs |
| Fraud Text (NLP) | Text-based fraud signals | BERT fine-tune | 200k labeled notes + documents | Claim notes text |
| Fraud Fusion | Combine all fraud signals | Logistic regression (meta) | 300k labeled | 32 model scores + cross-features |
| Reserve Estimator | Predict ultimate claim cost | LightGBM quantile (3 models: P10/P50/P90) | 600k closed claims | 100+ tabular features |

### Training Pipeline

```
┌────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Raw Data   │──▶│ Feature  │──▶│ Train/   │──▶│ Train    │──▶│ Model    │
│ Gold Layer │   │ Assembly │   │ Val/Test │   │ (GPU for │   │ Registry │
│ (S3/Delta) │   │ (Spark)  │   │ Split    │   │  DL/CPU  │   │ (MLflow) │
│            │   │          │   │ (time)   │   │  for GB) │   │          │
└────────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                                    │
                                                    ▼
                                           ┌──────────────────┐
                                           │ Evaluation Suite │
                                           │ • AUC-ROC, PR-AUC │
                                           │ • Lift at 5% FPR │
                                           │ • P50 coverage   │
                                           │ • Business sim    │
                                           │ • Fairness audit  │
                                           └──────────────────┘
```

### Evaluation & Validation

**Offline validation protocol**:
- **Temporal split**: train on claims 2018–2023, validate on 2024, test on 2025
- **Time-series walk-forward**: 6 expanding windows of 6 months each
- **Adversarial validation**: detect train/test distribution shift

**Key metrics**:

| Model | Primary Metric | Target | Business Metric |
|-------|---------------|--------|-----------------|
| Fraud detection | Recall@5%FPR | > 65% | $ fraud prevented |
| Triage | Macro F1 | > 0.88 | Auto-STP rate, cycle time |
| Reserve estimate | Pinball loss | — | Reserve accuracy ±15% |
| Document parser | Field-level accuracy | > 95% | Adjuster time saved |

**Fraud-specific metrics** (beyond standard ML):

| Metric | Definition | Target |
|--------|-----------|--------|
| Detection rate | Fraudulent claims flagged pre-payment / total fraud | > 65% |
| False positive rate (FPR) | Legitimate claims flagged as fraud / total legitimate | < 5% |
| Precision at investigation | % of investigated claims confirmed fraudulent | > 40% |
| Cost saved per investigation | Avg $ prevented per investigation case | > $8,000 |
| Time to detection (TTD) | Avg days from filing to fraud flag | < 1 day |
| Net fraud detection value | $ prevented − $ investigation cost | Maximized |

**Fairness & bias checks**:
- Parity in fraud flag rate across demographic groups (no disparate impact)
- Equal false positive rates across geographies, income levels
- No systematic over-flagging of specific claimant profiles
- Reserve estimation accuracy parity across policy types and regions
- Regular bias audits published quarterly to compliance committee

---

## 5. Deployment Architecture

### Target Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     API Gateway (Kong + WAF)                         │
└──────────┬──────────────────────┬────────────────────────┬───────────┘
           │                      │                        │
           ▼                      ▼                        ▼
┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
│ Real-time          │ │ Batch              │ │ Investigator       │
│ Claim Intake       │ │ Document Backfill  │ │ Portal (Graph UI)  │
│ (Kserve inference) │ │ (Airflow + Spark)  │ │ (React + D3.js)    │
└─────────┬──────────┘ └─────────┬──────────┘ └─────────┬──────────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │   Model Router          │
                    │  (by claim type +       │
                    │   model freshness)      │
                    └──────┬─────────────────┘
                           │
          ┌────────────────┼──────────────────┐
          ▼                ▼                   ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ Fraud Ensemble │ │ Triage         │ │ Reserve        │
│ (3 inference   │ │ (XGBoost CPU)  │ │ (LightGBM CPU) │
│  pods per      │ │                │ │                │
│  sub-model)    │ │                │ │                │
│ • GNN (GPU)    │ │                │ │                │
│ • TabNet (GPU) │ │                │ │                │
│ • BERT (GPU)   │ │                │ │                │
│ • Fusion (CPU) │ │                │ │                │
└──────┬─────────┘ └──────┬─────────┘ └──────┬─────────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
                    ┌────────────────────────┐
                    │   Decision Engine       │
                    │  (Rules + ML scores     │
                    │   → action)            │
                    └──────┬─────────────────┘
                           │
                           ▼
                    ┌────────────────────────┐
                    │  Claims System         │
                    │  (workflow + queue)    │
                    └────────────────────────┘
```

### Serving Strategy

| Workload | Pattern | Infrastructure | SLA | Volume |
|----------|---------|---------------|-----|--------|
| Claim triage | Real-time (event-driven) | Kserve on K8s, auto-scaled | p99 < 2s | 3,500/day |
| Fraud scoring | Real-time (event-driven) | Kserve, GPU for DL pods | p99 < 5s | 3,500/day |
| Document processing | Async (S3 trigger → queue) | GPU pods, batch size 32 | p99 < 30s | 20k docs/day |
| Reserve estimation | Real-time | CPU pods, pre-loaded model | p99 < 500ms | 3,500/day |
| Fraud investigator search | On-demand (portal) | GNN inference, cached graph | p99 < 3s | 200/day |
| Batch model retraining | Scheduled (weekly) | Spark + GPU cluster | Complete in 8h | 1.2M claims |

### Deployment Platform

```
┌──────────────────────────────────────────────────────────────────┐
│                     Kubernetes (EKS/AKS)                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │ Kserve   │ │ Airflow  │ │ Spark    │ │ Kafka    │ │ Redis │ │
│  │ (model   │ │ (pipeline│ │ Operator │ │ (event   │ │ (feat │ │
│  │ serving) │ │  orchest)│ │          │ │  broker) │ │ store)│ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────┘ │
├──────────────────────────────────────────────────────────────────┤
│  IaC: Terraform + Helm                          GitOps: ArgoCD   │
│  CI/CD: GitHub Actions → ECR/ACR → K8s                          │
│  Feature Store: Redis (online) + Feast (offline)                │
│  Model Registry: MLflow (S3 backend)                            │
│  Monitoring: Prometheus + Grafana + Evidently + WhyLabs         │
└──────────────────────────────────────────────────────────────────┘
```

### Rollout Strategy

| Phase | Scope | Duration | Validation |
|-------|-------|----------|------------|
| Shadow | Run all models alongside existing process, no decision impact | 6 weeks | Compare AI decisions vs. human decisions |
| Document AI only | Auto-extract documents, human review extraction | 4 weeks | Extraction accuracy > 95%, adjuster time saved |
| Triage + Reserve | Auto-routing + reserve recommendation (human must approve) | 8 weeks | A/B test: pilot vs. control queues |
| Fraud scoring | Fraud scores visible to investigators (advisory only) | 8 weeks | Detection rate improvement, false positive rate |
| Auto-adjudication | Straight-through processing for simple claims (< $5k, low risk) | 6 weeks | Customer satisfaction, error rate, appeal rate |
| Full autonomy | All models live, human exception-only | Ongoing | Business impact tracking |

### Rollback Plan
- **Per-model rollback**: each model independently deployable; can revert individual models without affecting others
- **Automated triggers**: fraud FPR > 5%, triage accuracy drop > 10%, document accuracy < 85%, reserve bias > 5%
- **Shadow parallel**: all models run in shadow during and after rollout; instant fallback to previous version
- **Human override**: every critical decision (deny, investigate, auto-pay > $5k) has human review option
- **Circuit breaker**: if model scores are unavailable, fall back to rule-based triage + mandatory human review

---

## 6. Monitoring & Observability

### Monitoring Stack

```
┌──────────────────────────────────────────────────────────────────┐
│                     Unified Observability                         │
├──────────────┬─────────────────┬───────────────┬─────────────────┤
│ Data         │ Model           │ Operational   │ Business        │
│ Monitoring   │ Monitoring      │ Monitoring    │ Impact          │
├──────────────┼─────────────────┼───────────────┼─────────────────┤
│ Feature drift│ Fraud score     │ Inference     │ Fraud detection │
│ (PSI/KS per  │ distribution    │ latency p50/  │ rate (daily)    │
│  feature)    │ (JSD)           │ p95/p99       │ FPR (daily)     │
│ Label drift  │ Calibration     │ Throughput    │ Cycle time      │
│ (delayed     │ (reliability    │ (models/min)  │ (median by      │
│  ground      │  diagram)       │ GPU/CPU util  │  queue)         │
│  truth)      │ Override rate   │ Error rate    │ Auto-STP rate   │
│ Data         │ Reserve bias    │ Queue depth   │ Settlement      │
│ freshness    │ (actual vs.     │ Memory        │ variance        │
│              │  predicted)     │               │ Litigation rate │
├──────────────┼─────────────────┼───────────────┼─────────────────┤
│ Evidently +  │ WhyLabs /       │ Prometheus +  │ Looker /        │
│ Great        │ Arize AI /      │ Grafana /     │ PowerBI +       │
│ Expectations │ NannyML         │ Datadog       │ custom dash     │
└──────────────┴─────────────────┴───────────────┴─────────────────┘
```

### Alerting Thresholds

| Alert | Severity | Threshold | Action |
|-------|----------|-----------|--------|
| Fraud FPR spike | P0 | FPR > 5% sustained for 1h | Auto-disable fraud model, revert to rules-only |
| Fraud detection rate drop | P1 | Detection rate < 50% | Investigate feature drift, retrain model |
| Document extraction accuracy | P1 | Field accuracy < 85% | Rollback document AI, manual processing fallback |
| Reserve bias > 10% | P2 | Avg (predicted - actual) / actual > 10% | Investigate, recalibrate |
| Model latency degradation | P2 | p99 > 5s for fraud, > 500ms for reserve | Scale pods, optimize inference |
| Data pipeline failure | P1 | No new claim events in 30min | Page on-call engineer |
| Feature drift (critical features) | P2 | PSI > 0.2 for journey_time, claim_amount | Investigate source, plan retraining |
| Auto-STP rate deviation | P2 | STP rate ± 20% from expected | Check distribution of incoming claims |
| Fairness drift | P1 | Disparate impact ratio < 0.8 for any group | Pause affected model, compliance review |

### Feedback Loops

- **Immediate (per-claim)**: When a human overrides the AI decision (fraud flag, reserve, triage category), that override is logged with the human's decision and rationale. This becomes immediate training signal.
- **Daily**: Automated model retraining for high-volatility models (fraud text, triage) using the latest override data.
- **Weekly**: Fraud confirmed/rejected labels backfill from investigations → retrain fraud models.
- **Monthly**: Full evaluation suite against holdout data; challenger models tested in shadow.
- **Quarterly**: Business impact review; adjust success criteria, model architecture, or feature set.

---

## 7. Business Integration & Change Management

### Decision Workflow

```
Claim Filed (FNOL)
        │
        ▼
┌──────────────────┐        ┌─────────────────────────────────────────┐
│ Document AI      │───────▶│ AI Triage + Fraud + Reserve             │
│ Auto-extract     │        │ (compute in < 5s)                       │
│ (15s)            │        └────────────┬────────────────────────────┘
└──────────────────┘                     │
                                         ▼
                              ┌──────────────────────────┐
                              │     Decision Engine       │
                              ├──────────┬────────┬───────┤
                              │ Auto-Pay │ Simple │Invest.│
                              │ (40%)    │ Review │(15%)  │
                              │          │ (45%)  │       │
                              └──────────┴────────┴───────┘
                                    │         │        │
                                    ▼         ▼        ▼
                              ┌────────┐ ┌────────┐ ┌────────┐
                              │ Auto-  │ │ Adjuster│ │ Fraud  │
                              │ paid   │ │ reviews │ │ invest │
                              │ in 2min│ │ in 4h   │ │ portal │
                              └────────┘ └────────┘ └────────┘
```

### Exception & Escalation Rules

| Scenario | Action |
|----------|--------|
| Fraud score > 85 | Auto-assign to fraud investigation team, hold payment |
| Fraud score 65–85 | Flag for adjuster review with recommended investigation checks |
| Reserve > $50k | Mandatory senior adjuster approval |
| Claimant is policyholder < 30 days | Mandatory fraud check + proof of loss verification |
| Auto-STP confidence < 0.9 | Route to human review despite auto-pay recommendation |
| AI confidence low (uncertainty high) | Flag "AI recommends manual review" in claims queue |

### Adoption Metrics

| Metric | Month 1 | Month 3 | Month 6 | Month 12 | Target |
|--------|---------|---------|---------|----------|--------|
| Auto-STP rate | 5% | 18% | 30% | 40% | 40%+ |
| Fraud alert override rate | 60% | 35% | 22% | 15% | < 15% |
| Reserve override rate | 55% | 40% | 25% | 18% | < 15% |
| Adjuster time on data entry | 65% | 45% | 30% | 20% | < 20% |
| Claim cycle time (median) | 21 days | 14 days | 9 days | 6 days | < 5 days |
| NPS (claims experience) | -10 | 5 | 25 | 42 | 50+ |

### Stakeholder Communication Cadence

| Audience | Frequency | Channel | Content |
|----------|-----------|---------|---------|
| Executive (CCO, CFO, CEO) | Monthly | Dashboard + 1-pager | Fraud $ prevented, STP rate, NPS, cost savings |
| Claims operations | Weekly | Standup + dashboard | Cycle time, queue health, override trends |
| Fraud investigators | Daily | Team huddle + case list | High-scoring claims, network alerts |
| Data Science team | Daily | Standup + run log | Model metrics, pipeline health |
| Regulators / Compliance | Quarterly | Report | Model validation, bias audit, explainability |
| Call center agents | Monthly | Email + training | What changed, how to handle AI-recommended decisions |

---

## 8. Results & Continuous Improvement

### Measured Outcomes (18 months post-deployment)

| Metric | Baseline | 6 Months | 12 Months | 18 Months | Variance |
|--------|----------|----------|-----------|-----------|----------|
| Fraud detection rate (pre-payment) | 20% | 48% | 60% | 68% | +48pp |
| Fraud false positive rate | — | 8% | 6% | 4.2% | -3.8pp (trending) |
| Auto-STP rate | 0% | 18% | 32% | 42% | +42pp |
| Claims cycle time (median) | 21 days | 14 days | 8 days | 5 days | -16 days |
| Settlement variance (CV) | 35% | 22% | 12% | 7% | -28pp |
| Adjuster productivity (claims/yr) | 120 | 180 | 290 | 380 | +260 |
| Litigation rate on denials | 12% | 9% | 5% | 2.5% | -9.5pp |
| Fraud $ prevented (annualized) | — | $48M | $112M | $165M | +$165M |
| Ops cost savings (annualized) | — | $15M | $32M | $48M | +$48M |
| Customer NPS (claims) | -10 | 5 | 28 | 45 | +55pts |

### Lessons Learned

| Lesson | Impact | Mitigation for Future |
|--------|--------|----------------------|
| Graph fraud model initially had high FPR (15% vs target 5%) | Many false investigations, adjuster frustration | Add precision gate to graph model; use graph scores as supplementary signal only, not primary decision |
| Document extraction accuracy dropped on mobile phone photos | 20% of claims needed manual correction | Add image quality pre-check; route low-quality images to human with auto-suggested fields |
| Fraudsters adapted within 3 months: changed filing patterns | Detection rate dropped 8% | Implement weekly retraining cycle + adversarial validation |
| Adjusters resisted AI fraud flags initially ("I know my claimants") | Override rate 60% in month 1 | Invested in explainability: show top-3 fraud signals per alert. Trust built over 4 months |
| Auto-STP claims had higher-than-expected supplement rate | 12% of auto-paid claims had supplements > 30% of original payment | Add supplement prediction model; set conservative auto-pay threshold |
| Model performance varied by state (different regulations, fraud patterns) | Accuracy gap of 15% between best and worst states | Train state-specific calibration layers on top of global model |

### Evolution Roadmap

```
Year 1                  Year 2                     Year 3
│                       │                          │
▼                       ▼                          ▼
───────────────────────────────────────────────────────────
Claims Intelligence     Fraud Network              Autonomous
(phase 1 complete)      Expansion                  Claims
                        │                          │
                        ▼                          ▼
                ┌──────────────┐           ┌──────────────┐
                │ Real-time     │           │ Predictive   │
                │ provider      │           │ Claims       │
                │ network watch │           │ Prevention   │
                │ (graph +      │           │ (flag risky  │
                │  streaming)   │           │  policy +    │
                │ Vendor fraud  │           │  recommend   │
                │ detection     │           │  mitigation) │
                │ Medical bill  │           │ Fully auto   │
                │ review AI     │           │ subrogation  │
                └──────────────┘           └──────────────┘
```

---

## 9. Key Takeaways for AI Architects

1. **Multi-model ensembles beat monolithic models**: Fraud detection requires different views (tabular, text, graph, behavioral). Each captures different signals; the fusion meta-model learns which expert to trust for which claim type.

2. **Graph neural networks are a game-changer for fraud**: Single-claim models miss network patterns. A GNN detecting a ring of claimants sharing providers, addresses, and vehicles catches fraud patterns invisible to pointwise models.

3. **Document AI is the highest-ROI first step**: 65% of adjuster time was data entry. Automating document extraction freed capacity for high-value investigation. Start with document AI, build trust, then add more complex models.

4. **Fraud detection requires an adversarial mindset**: Fraudsters adapt quickly. Weekly retraining, adversarial validation, and shadow-mode challengers are essential. What works today may not work next quarter.

5. **Explainability is non-negotiable for human-in-the-loop**: Adjusters and investigators need to understand *why* a claim was flagged. SHAP values, counterfactual explanations, and a "top signals" display built trust and reduced override rates from 60% to 15%.

6. **Regulatory compliance must be architected in, not bolted on**: Insurance is heavily regulated. Fairness audits, model documentation, explainability, and human review loops must be built into the platform from day one, not added as an afterthought.

7. **Phase rollout by model maturity, not by timeline**: Document AI was production-ready in 6 weeks; fraud network took 6 months. Let model readiness drive the rollout schedule, not the project plan.

8. **Feedback loops are the highest-leverage investment**: Every human override, every investigation outcome, every supplement request is training data. Architecturing these feedback loops properly was the single biggest driver of long-term accuracy improvement.
