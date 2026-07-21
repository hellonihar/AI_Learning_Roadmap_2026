# Phase 3: The Drawing Board
## Months 6–9

---

### Chapter 8: The Consultant

Marcus brought in a consultant. Not one of the big firms — a woman named Priya who'd built ML platforms at two FAANG companies and now ran her own practice.

She spent a week interviewing the team, reading their post-mortem, and reviewing their architecture.

On Friday, she presented her findings.

"You have a team problem," she said. "Not a technology problem."

She pointed at Raj. "You're a brilliant ML engineer. But you skip infrastructure because it's boring. You optimize for model accuracy the way a chef optimizes for taste — ignoring that the kitchen is on fire."

Raj opened his mouth, then closed it.

"Elena is your best asset. She's the only one thinking about operations. But she's one person trying to build infrastructure for a team of five. She needs a team."

"What about Anika?" Marcus asked.

"Anika is a good architect. She's also trying to be the PM, the tech lead, and the executive communicator. That's four full-time jobs."

Priya turned to the whiteboard.

"Here's my recommendation. You need to separate concerns:

1. **Platform team** — builds and maintains infrastructure. That's Elena, plus two more engineers.
2. **ML team** — builds and trains models. That's Raj, plus one more ML engineer.
3. **Product team** — scopes problems, measures success, manages stakeholders. That's David, plus a dedicated PM.
4. **Architecture** — Anika oversees all three, reports to Marcus.

"And you need a clear **build vs. buy** strategy. Stop trying to build everything yourselves. Use APIs where they're good enough. Build where you need differentiation."

"Example?" David asked.

"Document processing. You could build a custom LayoutLM pipeline. Or you could call Azure Document Intelligence or AWS Textract. The API is 90% as good as your model and costs less in engineering time."

"But we want to differentiate," Marcus said.

"Then differentiate where it matters. For ticket triage — the core customer experience — build your own RAG system. That's your moat. For document parsing, buy the API. Don't waste engineering time on commodity capabilities."

---

**Concepts introduced:** Platform vs. ML vs. Product team structure, build vs. buy framework, commodity vs. differentiation analysis, API-first approach, engineering time as a resource, organizational scaling patterns.

---

### Chapter 9: The MLOps Foundation

With Priya's framework adopted, Elena finally had the team she needed. Two new engineers joined — Sarah, a platform engineer from a fintech company, and Miguel, who'd run ML infrastructure at a Series B startup.

Their first task: build the MLOps foundation.

Elena drew the CI/CD pipeline on the whiteboard:

```
Git Push → Tests → Data Validation → Train → Eval → Registry → Canary → Deploy
                           ↓                                  ↓
                     Quality Gates                       Rollback
```

"Every code change goes through this pipeline," she explained. "No exceptions."

"What's in each stage?" Sarah asked.

1. **Tests** — unit tests for data processing, feature engineering, inference code
2. **Data validation** — Great Expectations suite checks schema, ranges, null ratios
3. **Training** — kicks off on K8s with spot GPUs, auto-checkpoints every 500 steps
4. **Evaluation** — compares against current production model. Must beat or tie on all metrics
5. **Registry** — MLflow stores every model version with metrics, params, dataset hash
6. **Deployment** — automated canary: 5% → 25% → 100%, with automatic rollback

"What about monitoring?" Miguel asked.

Elena added another layer:

```
Production → Metrics → Dashboard → Alerts
    ↓
Predictions → Feature Store → Drift Detection → Retrain Trigger
```

"Every prediction gets logged. Features, model version, confidence, actual outcome (when available). We monitor:
- **Data drift** — does the input distribution match training?
- **Concept drift** — does the relationship between features and labels change?
- **Model performance** — accuracy, precision, recall on labeled feedback
- **System metrics** — latency, throughput, error rate, GPU utilization"

Raj whistled. "That's a lot of infrastructure before we write any model code."

"That's the point. Infrastructure first, model second. We build the factory before we decide what to manufacture."

---

**Concepts introduced:** CI/CD for ML, automated model testing, data validation with Great Expectations, model registry (MLflow), automated canary deployment, monitoring for data drift, concept drift, and system metrics, prediction logging, feedback loop for model improvement, infrastructure-first approach.

---

### Chapter 10: Choosing the Right Problem

While Elena built the platform, David and Anika worked on problem selection.

They interviewed 30 stakeholders across 8 departments. They asked three questions:
1. What decision do you make repeatedly that could be automated?
2. What data do you have to support that decision?
3. What's the cost of being wrong?

The answer emerged clearly: **customer support ticket triage.**

"Here's the data," David said. "300,000 tickets per month. 40% are tier-1 — password resets, account status, billing inquiries. These take 5 minutes for a human but cost $12 per ticket. If we can automate even 30%, that's $4.3M annual savings."

"How automatic?" Anika asked.

"Route to the right team. Suggest a response. If confidence is high, auto-respond."

"The cost of being wrong?"

"Wrong routing wastes time. Wrong auto-response frustrates customers. Both are recoverable — unlike our document extraction failure where we almost signed a wrong contract."

The risk profile was better. The data was available. The business case was clear.

"One constraint," David added. "We can't use customer data to train models that leave our cloud. Legal requires data residency."

"That rules out external APIs," Raj said. "We need to deploy locally."

"Not entirely. We can use APIs for inference as long as no training data leaves our boundary. But it's easier to self-host."

"Let's go with self-hosted," Anika decided. "We'll use an open-source LLM. Mistral or Llama, quantized, deployed on our own GPU cluster."

"RAG architecture?" Raj asked.

"RAG architecture. Vector database for knowledge base retrieval. Prompt engineering for response generation. Guardrails for content safety."

Anika drew the high-level design:

```
TICKET ──→ Classifier ──→ Retrieve (Vector DB) ──→ Generate (LLM) ──→ Route/Respond
                ↑                      ↑
         Embedding Model        Knowledge Base
```

"Phase 1: Classify and route. Phase 2: Suggest responses. Phase 3: Auto-respond."

"What's the timeline?"

"If the platform is ready in two months, Phase 1 in production by month 9. Full triage by month 15."

"That's aggressive."

"That's realistic. We have the platform now. We just need to build on it."

---

**Concepts introduced:** Problem selection framework, tier-1 vs. high-risk AI use cases, cost-benefit analysis for automation, data residency constraints, self-hosted vs. API-based LLMs, open-source LLMs (Mistral, Llama), model quantization for deployment, RAG architecture high-level design, phased delivery roadmap.

---

Raj and Anika stayed late that night.

"Remember when we thought building the model was the hard part?" Raj said.

"Yeah. We were naive."

"We didn't know what we didn't know. The first failure taught us more than the next ten wins will."

"Then it was worth it."

Raj looked at the infrastructure diagram Elena had left on the board. Data lakehouse. Feature store. Model registry. CI/CD. Monitoring.

"Six months ago I had a Jupyter notebook," he said. "Now I have an ML platform."

"The notebook still works," Anika said. "It's just not production anymore."
