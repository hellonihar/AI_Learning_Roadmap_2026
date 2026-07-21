# Business & Leadership — Q&A

## Q1: Describe a time you convinced a skeptical executive to invest in AI.

**STAR framework:**

**Situation:** The VP of Product was skeptical about AI investment for a customer support product. Previous AI experiments had failed.

**Task:** Convince leadership to fund an LLM-powered support assistant despite skepticism.

**Action:**
1. Researched pain points — analyzed support tickets, found 40% were repetitive tier-1 issues
2. Built a focused proof of concept on 2 weeks of historical tickets. Demonstrated 70% accurate automated resolution
3. Made a business case — cost per ticket ($8 human vs. $0.50 AI), $3M annual savings at 30% deflection
4. Addressed skepticism directly — acknowledged past failures, showed RAG prevents hallucination, proposed shadow mode first
5. Phased rollout: AI assists agents in 3 months, then automation in 6 months

**Result:** VP approved $2M pilot. After 6 months: 35% ticket deflection, $1.2M annualized savings, NPS improved by 5 points.

---

## Q2: How would you prioritize between three competing AI project proposals?

**Prioritization framework (weighted scoring):**

| Criterion | Weight | Chatbot | Churn Prediction | Document Automation |
|-----------|--------|---------|-----------------|-------------------|
| Strategic alignment | 25% | 8 | 9 | 6 |
| ROI (18-month) | 25% | 9 | 7 | 8 |
| Feasibility | 20% | 7 (RAG ready) | 8 (existing model) | 5 (needs custom model) |
| Data readiness | 15% | 9 | 6 | 4 |
| Risk | 10% | 7 | 5 | 8 |
| Team capability | 5% | 8 | 7 | 6 |
| **Total** | **100%** | **8.05** | **7.25** | **6.4** |

Beyond scoring: also consider dependencies (does Chatbot enable other projects?), strategic urgency (competitor moves), and organizational capacity (can't do all three at once).

**Recommendation:** Start with Chatbot (highest score, quickest wins), then Churn Prediction (highest strategic alignment), defer Document Automation.

---

## Q3: What metrics would you use to measure the success of an AI platform team?

**Tier 1 (Platform Health):**
- Uptime / reliability (SLA achievement) for inference and training services
- Model deployment frequency (deployments per week)
- Time from model registration to production deployment
- GPU utilization % across the cluster
- Cost per inference or per training hour

**Tier 2 (Team Productivity):**
- Number of models in production
- Number of teams/use cases served
- Self-service adoption rate (% of deployments done without platform team help)
- Average time to onboard a new team

**Tier 3 (Business Impact):**
- Revenue or cost savings attributed to platform-supported models
- User satisfaction survey (platform NPS)
- Platform contribution to company AI strategy goals

**Key insight:** Don't measure the platform team by model accuracy. Measure by how well they enable other teams to build and deploy models.

---

## Q4: Walk through your build-vs-buy decision process for an LLM serving stack.

**Decision framework:**

| Factor | Build | Buy |
|--------|-------|-----|
| **Core differentiator?** | If LLM serving is your product's secret sauce | If it's a commodity capability |
| **Scale** | >100M inference requests/month | Lower volume |
| **Latency requirements** | <50ms (needs tight optimization) | <500ms (acceptable for most) |
| **Team capability** | Strong infra/MLOps team | Limited infra team |
| **Customization needed** | Custom quantization, custom kernels | Standard APIs suffice |
| **Vendor risk** | Want to avoid lock-in | Accept vendor dependency |
| **Time to market** | 3-6 months to build | Immediate |

**Decision process:**
1. Start with buy (API) — validate product-market fit quickly
2. Evaluate at scale — if cost exceeds $X/month and usage is growing, evaluate build
3. Build the wrapper layer — even on top of APIs, build abstraction layer for future migration
4. Hybrid — build for latency-sensitive, buy for burst/cost-sensitive

---

## Q5: How do you handle a model that achieves great accuracy but fails in production?

**Root causes to investigate:**
1. **Training-serving skew** — different data distribution between training and production
2. **Feature availability** — production features computed differently than training
3. **Label leakage** — training data contained information not available at inference time
4. **Feedback loops** — model predictions change future data distribution

**Action plan:**
1. Shadow deploy current model alongside production — compare predictions, find discrepancies
2. Log production features and predictions; reconstruct training-time features for comparison
3. Check for data drift and covariate shift between training and production data
4. Implement automated retraining with production data (once labels arrive)
5. Add feature validation gates — if a feature distribution shifts beyond threshold, alert
6. Consider simpler model (less prone to hidden assumptions) for production stability

---

## Q6: Design an AI strategy roadmap for a company that has never used AI.

**Phase 1 (0-3 months): Foundation**
- Assess data maturity — what data exists, quality, accessibility
- Build AI governance framework (ethics, risk, compliance)
- Hire first ML engineer or partner with ML consultancy
- Run AI discovery workshop with business stakeholders
- **Goal:** Understanding and readiness

**Phase 2 (3-9 months): Quick wins**
- Identify 1-2 high-ROI, low-complexity use cases (e.g., document classification, chatbot for IT support)
- Buy vs. build: prefer SaaS APIs for fast validation
- Establish data pipelines for these use cases
- **Goal:** Prove value, build organizational confidence

**Phase 3 (9-18 months): Scale**
- Expand to 3-5 use cases across departments
- Build ML platform/infrastructure (feature store, model registry, monitoring)
- Hire dedicated ML team
- **Goal:** Embed AI into business processes

**Phase 4 (18+ months): Innovate**
- Develop custom models for core differentiators
- Explore advanced AI (GenAI, agents, multimodal)
- Build internal AI Center of Excellence
- **Goal:** AI as competitive advantage

---

## Q7: How do you structure AI teams for maximum effectiveness?

**Common models:**

| Model | Description | Best For |
|-------|-------------|----------|
| **Centralized** | Single AI/ML team serves all products | Small-mid companies, consistent standards |
| **Embedded** | ML engineers embedded in product teams | Product-aligned development, fast iteration |
| **Hybrid** | Platform team + embedded ML engineers | Large orgs, multiple products |
| **Federated** | Each business unit has own ML team | Highly autonomous business units |

**Recommended (Hybrid):**
- **Central AI Platform team:** Builds infrastructure (feature store, model registry, MLOps), sets standards, enables self-service
- **Embedded ML Engineers:** Sit in product teams, build models for specific use cases, use platform infrastructure
- **AI Research team (optional):** For frontier work, new capability exploration

**Ratios:** Platform team should be ~20% of total ML headcount. Ideal size: 5-8 platform engineers supporting 20-30 ML engineers.

---

## Q8: Calculate TCO for an AI system — what factors do you include?

**Compute costs:**
- Training: GPU hours, spot vs. on-demand, storage for checkpoints
- Inference: GPU hours, reserved vs. serverless, model size determines cost
- Development: lower-tier GPUs for experimentation

**Data costs:**
- Storage: object store, feature store, vector DB
- Labeling: internal vs. outsourced vs. programmatic
- Ingestion: data pipeline compute (Spark, Airflow)
- Egress: transferring data between regions/providers

**Platform costs:**
- Cloud infrastructure (K8s, networking, load balancers)
- ML platform licenses (W&B, MLflow, DataRobot)
- Monitoring tools (Prometheus, Grafana, Arize)

**People costs:**
- ML engineers, data engineers, MLOps engineers
- Domain experts for labeling/validation
- Governance, compliance, legal review time

**Hidden costs:**
- Cost of bad predictions (false positives/negatives)
- Model maintenance and retraining over time
- Technical debt from experimentation (abandoned projects)
- GPU idle time during low traffic periods

---

## Q9: How do you communicate technical constraints to non-technical executives?

**Framing principles:**

1. **Translate to business language:**
   - "We need 8 GPUs" ? "We need $X/month compute. This is the same as running Y servers."
   - "The model has 95% accuracy" ? "We'll automate 80% of cases perfectly, the remaining 20% need human review."
   - "Latency is 500ms" ? "The user will wait half a second for a response."

2. **Use analogies:**
   - Model drift ? "It's like a GPS that works in familiar neighborhoods but needs recalibration when roads change."
   - Training data quality ? "Garbage in, garbage out. The model is only as good as its training material."

3. **Present trade-offs, not absolutes:**
   - "We can launch in 2 weeks with 80% accuracy, or in 3 months with 95%. What's more important for this quarter?"

4. **Bring options, not problems:**
   - Not "We don't have enough GPUs" but "With current GPU budget, we can serve X requests/hour. To reach Y, we need $Z or these optimizations."

---

## Q10: How do you run an AI discovery workshop with business stakeholders?

**Agenda (half-day):**

**Hour 1: Education**
- 15 min: What AI can and cannot do (myth-busting)
- 20 min: Examples from the same industry
- 15 min: Process overview (how AI projects work)
- 10 min: Q&A

**Hour 2-3: Ideation**
- Brainstorm use cases per department
- Group by: pain point, data availability, complexity
- For each idea: what's the input, output, success metric?

**Hour 3-4: Prioritization**
- Score each idea on: business value (1-5), feasibility (1-5), data readiness (1-5)
- Plot on value vs. effort matrix
- Select top 3 for deep-dive

**Output:**
- Prioritized list of AI use cases with scoring
- Data readiness assessment
- Data access owners identified per use case
- 90-day action plan

---

## Q11: How do you handle vendor evaluation for AI platforms?

**Evaluation criteria:**

1. **Model quality** — benchmark on your specific task (not just vendor benchmarks). Run blind A/B test.
2. **Latency & reliability** — P50/P95/P99 latency, SLA, uptime history
3. **Pricing model** — per-token, per-hour, reserved. Calculate at your projected volume
4. **Data handling** — does vendor train on your data? Where is data stored? Deletion policy?
5. **Security & compliance** — SOC 2, ISO 27001, data residency, encryption
6. **Customization** — fine-tuning availability, RAG integration, custom model hosting
7. **Vendor lock-in** — is the API standard? Can you migrate? Data portability?
8. **Support** — SLA for support tickets, technical account manager?
9. **Roadmap** — upcoming features that matter to you

**Process:**
- Step 1: RFI to 4-6 vendors
- Step 2: Hands-on POC with top 2-3 (real use case, real data)
- Step 3: Security review
- Step 4: Commercial negotiation (leveraging POC results)
- Step 5: 3-6 month pilot, then commit

---

## Q12: What KPIs would you use to track AI initiative success?

**Business KPIs (for executives):**
- ROI / cost savings attributable to AI
- Revenue uplift (e.g., from recommendation system)
- Customer satisfaction change (NPS, CSAT)
- Time saved (employee hours or customer wait time)
- Automation rate (percentage of tasks fully automated)

**Product KPIs (for product managers):**
- User adoption rate (% of users engaging with AI feature)
- User retention (do AI users stay longer?)
- Feature usage frequency
- User satisfaction with AI feature (rating or survey)

**Technical KPIs (for engineers):**
- Model accuracy/performance (as validated offline)
- Inference latency (P50/P95/P99)
- Service uptime / availability
- Data freshness (time from event to feature availability)
- Model staleness (time since last retraining)

**Leading vs. lagging indicators:**
- Leading (predict future success): adoption rate, data quality score, retraining frequency
- Lagging (measure past success): ROI, revenue impact, cost savings

---

## External Resources

- [Interview Guys: 10 AI Solutions Architect Questions (2026)](https://blog.theinterviewguys.com/ai-solutions-architect-interview-questions/)
- [Future Skills Academy: Top AI Architect Questions (2026)](https://futureskillsacademy.com/blog/top-ai-architect-interview-questions-and-answers/)
- [Tredence: Top 20 AI & ML Questions (2026)](https://www.tredence.com/blog/top-20-ai-machine-learning-interview-questions-2026)
- [Index.dev: 50 Principal AI/ML Architect Questions](https://www.index.dev/interview-questions/principal-ai-ml-architect)
- [Maywise: 5 AI Interview Questions Guide (2026)](https://maywise.in/blog/5-ai-interview-questions-2026-guide)
