# Case Study: AI-Driven Demand Forecasting & Inventory Optimization for Omnichannel Retail

## 1. Business Problem & Opportunity

### The Scenario
A national fashion retailer with 500+ stores, an e-commerce platform, and 3 distribution centers loses **$120M annually** due to:
- **Stockouts** (25% of lost revenue): trending items sell out within hours; restocking takes 7–14 days
- **Overstock & markdowns** (35% of lost margin): seasonal inventory sits for months, sold at 60% discount
- **Excess working capital**: $200M tied up in slow-moving inventory across warehouses and stores
- **Channel fragmentation**: e-commerce and physical retail use separate forecasting systems, causing misallocation

### Business Case
| Metric | Current | Target | Annual Impact |
|--------|---------|--------|---------------|
| Stockout rate | 12% of SKU-days | < 3% | +$45M revenue |
| Markdown depth | 60% avg discount | < 30% | +$30M margin |
| Inventory turns | 4.2× per year | 7.5× | -$50M working capital |
| Forecast accuracy (MAPE) | 55% | < 20% | (enables above) |

### Success Criteria
- Forecast accuracy (MAPE) < 20% at SKU-store-week level
- Stockout rate reduction to < 3% across top 20% of revenue SKUs
- Markdown rate reduction of 50% through better initial buy and replenishment
- Inventory turnover improvement to > 7× per year
- **ROI 5:1 within 18 months** (project cost $8M, expected annual benefit $40M+)

---

## 2. Problem Framing & AI Solution Assessment

### Is AI the Right Approach?

| Factor | Assessment |
|--------|-----------|
| Problem structure | Time-series forecasting with hierarchical structure (SKU → category → region → total) |
| Data availability | 5 years of historical sales, promotions, pricing, weather, events — sufficient |
| Decision complexity | High — thousands of SKUs across stores with complex demand patterns (seasonality, trends, promotions) |
| Business readiness | Executive sponsor (COO) identified, data team exists but no ML experience |
| Alternative | Rule-based (moving average) cannot capture complex interactions; traditional statistical (ARIMA) fails with intermittent demand |

**Decision**: AI is appropriate due to the scale (500+ stores × 10k SKUs), complexity (promotional lift, seasonality, trend interactions), and need for automated decision-making at SKU-store granularity.

### AI Use Case Canvas

| Element | Definition |
|---------|-----------|
| Use case name | AI Demand Forecasting & Inventory Optimization |
| Business objective | Reduce stockouts and overstock by 50%+ each |
| Primary stakeholders | COO, Supply Chain VP, Store Operations, Finance |
| AI task | Hierarchical time-series forecasting with uncertainty quantification |
| Inputs | Historical sales, inventory, promotions, pricing, weather, events, store attributes |
| Outputs | Forecast (mean + prediction intervals) at SKU-store-day, recommended replenishment quantity |
| Decisions powered | Inventory buy quantities, allocation to stores, markdown timing |
| Success metric | MAPE, stockout rate, inventory turns, margin improvement |
| Fallback | Statistical ensemble (Prophet + ARIMA) if ML pipeline fails |

---

## 3. Data Strategy & Engineering

### Source Systems & Data Inventory

| Source | Data | Volume | Freshness | Method |
|--------|------|--------|-----------|--------|
| POS / Transaction DB | Daily sales by SKU-store, returns, customer counts | 5M rows/day | T+1 batch | Kafka → S3 |
| Inventory System | Stock levels, shipments, DC inventory, inbound | 2M updates/day | Near-real-time | CDC → Kafka |
| Promotions Calendar | Promo type, discount %, duration, media spend | 200 events/year | Scheduled | API pull |
| Pricing System | Regular price, promo price, markdown schedule | 500k price-points | T+1 batch | DB export → S3 |
| Weather API | Temperature, precipitation, holiday flags | 365 days × 500 stores | Daily pull | External API |
| Store Attributes | Square footage, region, urban/rural, store tier | 500 rows | Static | DB table |
| E-commerce | Online sales, browsing, cart adds by SKU | 2M events/day | Real-time | Kafka → S3 |

### Data Pipeline Architecture

```
┌──────────────┐    ┌──────────┐    ┌────────────────┐    ┌─────────────┐
│ Source       │───▶│ Kafka    │───▶│ Stream         │───▶│ Bronze      │
│ Systems      │    │ (event   │    │ Processing      │    │ (raw,       │
│              │    │  log)    │    │ (Flink/Spark)   │    │  append)    │
└──────────────┘    └──────────┘    └────────────────┘    └──────┬──────┘
                                                                │
                                                                ▼
┌──────────────┐    ┌──────────┐    ┌────────────────┐    ┌─────────────┐
│ External     │───▶│ API      │───▶│ Batch          │───▶│ Silver      │
│ Weather/     │    │ Gateway  │    │ Ingestion      │    │ (cleaned,   │
│ Events       │    │          │    │ (Airflow)      │    │  validated) │
└──────────────┘    └──────────┘    └────────────────┘    └──────┬──────┘
                                                                │
                                                                ▼
┌──────────────┐    ┌──────────┐    ┌────────────────┐    ┌─────────────┐
│ Feature      │◀───│ Feature  │◀───│ Aggregation +  │◀───│ Gold        │
│ Store        │    │ Pipeline │    │ Feature Eng    │    │ (features,  │
│ (Redis/S3)   │    │ (Spark)  │    │ (Spark Jobs)   │    │  ready)     │
└──────────────┘    └──────────┘    └────────────────┘    └─────────────┘
```

### Feature Engineering

**Temporal features** — derived from sales + calendar:
- Lagged sales (t-7, t-14, t-28, t-365)
- Rolling statistics (7-day, 28-day, 90-day mean/std)
- Day-of-week, month, quarter, holiday flags
- Seasonal decomposition (trend, seasonal, residual)

**Promotional features**:
- Promo type indicator (BOGO, % off, $ off)
- Promo depth (discount %)
- Weeks since last promo, promo duration
- Media spend during promo period

**External features**:
- Temperature deviation from seasonal norm
- Precipitation indicator
- Competitor promotion intensity (via web scrape)
- Economic indicators (consumer confidence, unemployment)

**Store & product features**:
- Store tier (high/medium/low volume)
- Product category life-cycle stage (new, mature, declining)
- Product seasonality cluster
- Store-SKU affinity (how this SKU sells in this store)

### Data Quality Framework

| Check | Tool | Action on Failure |
|-------|------|-------------------|
| Null rate < 5% per feature | Great Expectations | Alert + impute using median |
| Sales > 0 for active SKUs | Soda | Flag for manual review |
| Promo dates within valid range | Custom validator | Reject batch, alert source owner |
| Forecast vs actual variance < 3σ | Automated monitor | PagerDuty escalation for pipeline |
| Feature drift (PSI > 0.2) | Evidently | Retrigger training pipeline |

---

## 4. Model Development Lifecycle

### Approach: Hybrid ML + Statistical

```
                     ┌──────────────────────┐
                     │ Input Features (all)  │
                     └──────────┬───────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
        ┌──────────────┐               ┌──────────────┐
        │ High-volume   │               │ Low-volume    │
        │ SKUs          │               │ / New SKUs   │
        │ (>5 sales/wk) │               │ (<5 sales/wk) │
        └──────┬───────┘               └──────┬───────┘
               │                              │
               ▼                              ▼
        ┌──────────────┐               ┌──────────────┐
        │ Deep Learning │               │ Probabilistic  │
        │ (TFT / MQF2)  │               │ (Prophet +     │
        │               │               │  Croston)      │
        └──────┬───────┘               └──────┬───────┘
               │                              │
               └───────────────┬──────────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Reconciliation Layer │
                    │ (MinT / ERM bottom-up│
                    │  + top-down blend)   │
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │ Uncertainty Wrap     │
                    │ (prediction intervals │
                    │  via quantile loss)   │
                    └──────────────────────┘
```

### Model Selection Experiments

| Model | MAPE (holdout) | Train Time | Inference | Pros | Cons |
|-------|---------------|-----------|-----------|------|------|
| Seasonal Naive | 38% | — | Instant | Baseline | Poor with promotions |
| Prophet | 28% | 2min/100 SKUs | Fast | Handles seasonality + holidays | No interactions |
| LightGBM | 22% | 30min | Fast | Feature interactions, fast | No uncertainty native |
| TFT (Temporal Fusion Transformer) | 18% | 6hrs | Moderate | Quantile outputs, attention | Requires GPU training |
| MQF2 (Multi-Quantile) | 17% | 8hrs | Moderate | Calibrated intervals | Complex setup |

**Final selection**: TFT for top 20% SKUs (80% of revenue), LightGBM for mid-volume, Prophet + Croston for long-tail and new SKUs.

### Training Pipeline

```
Raw Data (S3) ──▶ Feature Engineering (Spark) ──▶ Train/Val/Test Split (time-based)
                                                        │
                                                        ▼
                                              ┌──────────────────┐
                                              │ Hyperparameter    │
                                              │ Optimization      │
                                              │ (Optuna, 100     │
                                              │  trials)         │
                                              └───────┬──────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │ Distributed Train │
                                              │ (Ray / SageMaker  │
                                              │  with GPU for TFT,│
                                              │  CPU for others)  │
                                              └───────┬──────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │ Model Evaluation  │
                                              │ 18 metrics: MAPE, │
                                              │ sMAPE, MASE,     │
                                              │ pinball loss,    │
                                              │ coverage         │
                                              └───────┬──────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │ Model Registry   │
                                              │ (MLflow: params,  │
                                              │ metrics,        │
                                              │ artifacts)      │
                                              └──────────────────┘
```

### Evaluation & Validation

**Offline validation protocol**:
- **Time-series cross-validation**: expanding window with 6 folds (6 months each)
- **Holdout**: most recent 12 months (never seen during training)
- **Backtesting**: simulate decisions using forecast vs. actual for 24-month period

**Key metrics** (beyond MAPE):

| Metric | Definition | Target |
|--------|-----------|--------|
| Weighted MAPE (wMAPE) | Revenue-weighted forecast error | < 15% |
| Pinball loss (P50, P90) | Quantile loss for probabilistic forecast | Minimized |
| Prediction interval coverage | % of actuals within 80% PI | 75–85% |
| Bias | Mean signed deviation | ±2% |
| MASE | Scaled by naive forecast MAE | < 1.0 |
| Business simulation ROI | $ impact of using forecast vs. current method | +$40M |

**Fairness & bias checks**:
- Forecast accuracy parity across store tiers (urban vs. rural, high vs. low volume)
- Accuracy parity across product categories (fashion vs. basics, seasonal vs. staple)
- No systematic underpredicting for stores in low-income areas

---

## 5. Deployment Architecture

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         API Gateway (Kong)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                 ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ Batch        │ │ Real-time    │ │ What-If      │
    │ Forecast API │ │ Replenishment│ │ Simulator    │
    │ (daily)      │ │ (on-demand)  │ │ (for planners)│
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                 │
           └────────────────┼─────────────────┘
                            ▼
                ┌──────────────────────┐
                │ Model Router         │
                │ (route by SKU volume │
                │  & freshness)        │
                └──────┬───────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│ TFT        │ │ LightGBM   │ │ Prophet    │
│ Inference  │ │ Inference  │ │ (fallback) │
│ (GPU pod)  │ │ (CPU pod)  │ │ (CPU pod)  │
└──────┬─────┘ └──────┬─────┘ └──────┬─────┘
       │              │              │
       └──────────────┼──────────────┘
                      ▼
            ┌──────────────────┐
            │ Reconciliation   │
            │ Layer (Spark)    │
            └──────┬───────────┘
                   ▼
            ┌──────────────────┐
            │ Output           │
            │ (Forecast DB +   │
            │  S3 Parquet)     │
            └──────────────────┘
```

### Serving Strategy

| Workload | Pattern | Infrastructure | SLA | Volume |
|----------|---------|---------------|-----|--------|
| Weekly forecast | Batch (Sat 2am) | Spark on K8s, 50 pods | Complete by 6am | 5M SKU-store combos |
| Daily replenishment | Batch (midnight) | Spark on K8s, 20 pods | Complete by 4am | 500k SKU-store |
| On-demand what-if | Real-time API | CPU pods, pre-loaded model | p99 < 500ms | 100 req/min |
| Store dashboard sync | Micro-batch (15min) | Light pipeline | Fresh within 15min | Incremental |

### Deployment Platform

```
┌─────────────────────────────────────────────────────┐
│                   Kubernetes (EKS/AKS)              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │  Batch   │ │  Online  │ │  Monitor │ │  UI    │ │
│  │(Argo)   │ │(Kserve)  │ │(Grafana) │ │(Streamlit)│
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ │
├─────────────────────────────────────────────────────┤
│  Infrastructure as Code: Terraform + Helm Charts     │
│  GitOps: ArgoCD                                      │
│  CI/CD: GitHub Actions → Container Registry → K8s    │
└─────────────────────────────────────────────────────┘
```

### Rollout Strategy

| Phase | Scope | Duration | Validation |
|-------|-------|----------|------------|
| Shadow | Run forecasts alongside existing system, no decision impact | 4 weeks | Compare accuracy, business simulation |
| Pilot | 50 stores (high, mid, low volume mixes) | 8 weeks | A/B test: pilot vs. control stores |
| Rollout | All stores, category by category | 12 weeks | Phase gates: stockout < 5%, MAPE < 25% |
| Optimize | Full production + markdown optimization | Ongoing | Business impact tracking |

### Rollback Plan
- **Automated rollback** triggers: forecast MAPE > 30%, system latency > 5× baseline, stockout rate increase > 2%
- **Manual rollback**: one-click helm rollback to previous model version
- **Fallback system**: statistical ensemble (Prophet + ARIMA) runs in parallel, ready to cut over in < 5 minutes
- **Data quality circuit breaker**: if feature pipeline fails, use cached features from last successful run

---

## 6. Monitoring & Observability

### Monitoring Stack

```
┌──────────────────────────────────────────────────────────────┐
│                    Unified Observability                      │
├────────────┬───────────────┬──────────────┬──────────────────┤
│ Data       │ Model         │ Operational  │ Business         │
│ Monitoring │ Monitoring    │ Monitoring   │ Impact           │
├────────────┼───────────────┼──────────────┼──────────────────┤
│ Feature    │ Prediction    │ Latency p50/ │ Stockout rate    │
│ drift (PSI)│ drift (JSD)   │ p95/p99      │ (daily)          │
│ Label drift│ Accuracy      │ Throughput   │ Markdown rate    │
│ Freshness  │ (delayed)     │ Error rate   │ Inventory turns  │
│ Null rate  │ Coverage of   │ GPU util     │ Revenue impact   │
│ Distribution│ PIs          │ Memory       │ Forecast bias    │
├────────────┼───────────────┼──────────────┼──────────────────┤
│ Evidently  │ WhyLabs /     │ Prometheus + │ Custom dashboard │
│ + Great    │ Arize AI      │ Grafana      │ (Looker /        │
│ Expectations│              │              │ PowerBI)         │
└────────────┴───────────────┴──────────────┴──────────────────┘
```

### Alerting Thresholds

| Alert | Severity | Threshold | Action |
|-------|----------|-----------|--------|
| Feature drift | P2 | PSI > 0.2 for 3+ features | Investigate data source, retrain if persistent |
| Accuracy degradation | P1 | Rolling 7d MAPE > 25% | Auto-rollback to previous model version |
| Data pipeline failure | P1 | No new data in 2h | Page on-call engineer |
| Forecast bias | P2 | |bias| > 5% for 3+ consecutive days | Adjust post-processing |
| Service latency | P2 | p99 > 1s | Scale pods, investigate bottleneck |

### Feedback Loops

- **Short-term**: Weekly model retraining with new data (automated)
- **Medium-term**: Monthly evaluation of model architecture; compare challenger models
- **Long-term**: Quarterly business impact review; adjust use case, feature set, or approach
- **Human feedback**: Planners can flag forecasts as over/under; flags become training data for corrective model

---

## 7. Business Integration & Change Management

### Decision Workflow

```
Forecast Generated (AI)
        │
        ▼
┌──────────────────┐     Accept      ┌──────────────────┐
│ Review Dashboard │──────────────▶  │ Auto-push to     │
│ (Planner reviews  │                │ Replenishment    │
│  exception flags) │                │ System           │
│                   │◀────────────── │                  │
│ Threshold: only   │    Override    │ Manual adjustment│
│ flag if |error|   │               │ logged for       │
│ > 15% or stockout │               │ feedback loop    │
│ risk > 10%        │               │                  │
└──────────────────┘                └──────────────────┘
```

### Adoption Metrics

| Metric | Month 1 | Month 3 | Month 6 | Target |
|--------|---------|---------|---------|--------|
| % auto-accepted orders | 40% | 70% | 85% | 90% |
| Planner override rate | 35% | 20% | 12% | < 10% |
| Time saved per planner/week | 5 hrs | 12 hrs | 18 hrs | 20 hrs |
| Net Promoter Score (planners) | — | 25 | 55 | 60+ |

### Stakeholder Communication Cadence

| Audience | Frequency | Channel | Content |
|----------|-----------|---------|---------|
| Executive (COO, CFO) | Monthly | Dashboard + 1-pager | Business impact ($), adoption %, risks |
| Supply chain team | Weekly | Standup + dashboard | Forecast accuracy, action items |
| Data Science team | Daily | Standup + run log | Model performance, pipeline health |
| Store managers | Quarterly | Town hall + email | What changed, how it helps them |
| Vendors/suppliers | As needed | Forecast share portal | Share forecast for supplier planning |

---

## 8. Results & Continuous Improvement

### Measured Outcomes (12 months post-deployment)

| Metric | Baseline | 6 Months | 12 Months | Variance |
|--------|----------|----------|-----------|----------|
| Forecast MAPE (weighted) | 55% | 24% | 18% | -37pp |
| Stockout rate | 12% | 5% | 2.8% | -9.2pp |
| Markdown depth | 60% | 38% | 28% | -32pp |
| Inventory turns | 4.2× | 5.8× | 7.2× | +3.0× |
| Working capital reduction | — | $22M | $48M | +$48M |
| Revenue recovered | — | $18M | $38M | +$38M |
| Planner productivity gain | — | 8h/wk | 16h/wk | +16h/wk |
| AI model retrain frequency | — | Weekly | Weekly | Stable |

### Lessons Learned

| Lesson | Impact | Mitigation for Future |
|--------|--------|----------------------|
| Feature drift in promotional response post-holiday season | MAPE spiked 8% in January | Add season-specific feature sets; automatic drift retrigger |
| New SKU cold-start problem underperformed | 3 months to reach mature accuracy | Use product-attribute similarity transfer learning |
| Planner trust took longer than expected | Override rate remained high for 4 months | Invest in explainability (SHAP per SKU-store forecast card) |
| Batch pipeline couldn't keep up with flash sales | 2-hour delay in forecast update | Add incremental prediction for flash-sale items (real-time stream) |

### Evolution Roadmap

```
Q1        Q2          Q3          Q4          Q5+
│         │           │           │           │
▼         ▼           ▼           ▼           ▼
Forecast → Replenish → Markdown    Price     Autonomous
only       optimize    optimize    elastic   supply chain
                      (AI agent)  (real-time)
```

---

## 9. Key Takeaways for AI Architects

1. **Start with business value, not technology**: The MAPE improvement is meaningless unless it translates to stockout reduction and margin improvement. Define business metrics before ML metrics.

2. **Hybrid models beat single models**: No single algorithm works for all SKU types. Tier your approach — deep learning for high-volume, gradient boosting for mid-volume, statistical for long-tail.

3. **Data engineering is 70% of the work**: The model architecture was straightforward; the hard part was reliable, fresh, clean data pipelines from 7+ source systems with consistent schemas.

4. **Uncertainty quantification is mandatory for business decisions**: A point forecast without prediction intervals leads to overconfident decisions. Quantile forecasts enabled risk-based inventory decisions (e.g., safety stock = P90 - P50).

5. **Plan for human-in-the-loop from day one**: The system succeeds because planners trust it. Invest in explainability, exception flagging, and override logging. The override data itself is valuable training feedback.

6. **Monitoring must span data, model, operations, and business**: Siloed monitoring misses cross-domain issues. A feature drift spike might explain a business metric decline.

7. **Rollout in stages, measure at every gate**: Shadow → Pilot → Phased rollout with clear go/no-go criteria at each stage. This builds confidence and allows course correction before full commitment.

8. **Change management is half the project**: The ML model was built in 8 weeks; the organizational adoption took 12 months. Budget for training, communication, and trust-building.
