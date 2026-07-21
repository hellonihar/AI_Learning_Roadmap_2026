# Phase 2: The Fall
## Months 3–6

---

### Chapter 4: The Deep Dive

The post-mortem lasted three days. Anika insisted they investigate every failure, no matter how small. Elena brought printouts of all system metrics for the past month. Raj brought his notebooks. David brought stakeholder complaints.

They pinned them to the wall.

By noon on day one, the wall was covered.

"This is our autopsy," Anika said.

Raj started. "I'll own the model issues. We trained on 50K Kaggle invoices — clean, single-page, high-res. Production has handwritten docs, multi-page, varying DPI, watermarks, folds, coffee stains. We never checked the production data distribution."

"Let's quantify that," Elena said. "How different?"

Raj pulled up histograms. "Training images: average resolution 1200x1600, single page, black text on white. Production: average resolution ranges from 600x800 to 2000x3000, 30% multi-page, 15% handwritten, 5% contain stamps overlapping text."

"That's covariate shift," Elena said. "The input distribution changed. The model generalized poorly because it never saw this variation."

*Covariate shift.* Anika wrote it on the board. "What else?"

"Label leakage," Raj said quietly.

"What?"

"In the Kaggle dataset, the 'Total Amount' field was almost always in the bottom-right quadrant of the page. The model learned location, not content. When we showed it documents where the total was in a different position — like multi-page contracts where the total is on page 3 — it grabbed whatever looked like a number from the bottom-right."

"That's cheating," David said.

"That's what label leakage looks like," Raj said. "A spurious correlation. The model found a shortcut."

Anika added it to the board. *Label leakage — spurious correlations.*

"What else?" she asked.

Elena stepped up. "Our infrastructure was a house of cards. Single GPU, no fallback, no monitoring, no drift detection. We didn't know anything was wrong until users complained. We were flying blind."

"What do you need?"

"Feature store. Model registry. A/B testing framework. Automated monitoring. CI/CD pipeline. And a data pipeline that validates input before it reaches the model."

"That's a lot."

"That's the minimum for production ML," Elena said. "We skipped it because we were rushing. And it cost us."

David added his piece. "I also failed. I took stakeholder requests at face value without understanding their real needs. Every department said 'AI for X,' but none of them had clean data, clear success metrics, or realistic expectations. I should have called that out."

Anika looked at the wall. Fifteen failure modes, each with a root cause, each preventable with proper engineering.

"We need to go back to fundamentals," she said. "We're going to rebuild. But this time, we're going to do it right."

---

### Chapter 5: The Diagnosis

Elena spent the next two weeks building a complete map of NovaTech's data landscape. What she found was sobering.

Data lived in 12 different systems. CRM, ERP, ticketing system, document management, email archives, legacy databases, spreadsheets on shared drives, SharePoint sites abandoned in 2019. Some data was structured. Most was not. None of it was documented.

"There's no single source of truth," she told the team. "Every department has their own version of the data. Sales has customers in Salesforce. Support has them in Zendesk. Billing has them in SAP. They don't agree on what a 'customer' even is."

"Standard problem," Anika said. "Data silos."

"It gets worse. The document processing system we built — the training data came from one department's scanned files. But production documents come from all departments. Different formats, different templates, different quality."

"That's why the model failed on multi-page. We trained on one distribution, served on another."

"Exactly. The fix isn't more model training. The fix is a unified data layer — a data lakehouse — that ingests, validates, and catalogs all documents before they ever reach a model."

Elena drew the architecture:

```
Sources (CRM, ERP, Docs, Tickets)
    ↓
Ingestion Pipeline (Kafka + Spark)
    ↓            ↓
Bronze (raw) → Silver (validated) → Gold (curated)
    ↓                    ↓
Raw Storage         Feature Store
                    Model Training
```

"A lakehouse with medallion architecture," Anika said. "Bronze, silver, gold."

"Delta Lake on S3," Elena confirmed. "ACID transactions, schema enforcement, time travel. Every document gets cataloged before it's used."

"How long to build this?"

"Six months if I had a team. Twelve months with just me."

"Then we need to prioritize. What's the smallest version that helps?"

Elena thought. "Forget the full lakehouse. We need three things right now:
1. Know where production documents come from
2. Validate input quality before inference
3. Log every prediction with input features for debugging"

"That's a lot simpler."

"One month. But it means delaying the next model deployment."

Anika looked at Marcus, who had been quietly listening.

"We don't have a choice," Marcus said. "We rush again, we fail again. And next time, I won't get another budget."

---

**Concepts introduced:** Data silos, data lakehouse architecture, medallion architecture (bronze/silver/gold), Delta Lake, ACID transactions on data lakes, schema enforcement, data cataloging, input validation pipeline, prediction logging, covariate shift diagnosis.

---

### Chapter 6: The Feature Store Debate

Raj had been reading. Papers, blog posts, documentation for open-source projects. He came to the next meeting with a realization.

"We've been thinking about this wrong," he said. "The model isn't the product. The features are."

He pulled up a diagram of a ride-sharing recommendation system. "In production ML, feature computation is the hardest problem. Every model needs features — embeddings, aggregates, rolling statistics. If you compute them differently in training and inference, you get training-serving skew. That's what happened to us."

"What do you propose?" Anika asked.

"A feature store. A central repository where features are computed once and reused across models. Feast is open source. Tecton is managed. Either way, we stop computing features ad-hoc in notebooks."

Elena nodded. "A feature store also gives us:
- Point-in-time correctness for training data (no data leakage)
- Consistent computation between training and inference
- Feature sharing across teams
- Monitoring for feature drift"

"Point-in-time what?" David asked.

"When you train a model, you need features as they existed at prediction time, not as they exist now. If you join today's data with yesterday's labels, you leak future information. A feature store handles that automatically."

David shook his head. "I didn't even know that was a problem."

"Most people don't. And it's why ML projects fail in production."

"Is this necessary for a first model?" Marcus asked.

"It's necessary for any model that goes to production," Raj said. "Otherwise, your training data is wrong, your evaluation is wrong, and your production metrics are meaningless."

"Do we build or buy?"

"Build for now," Elena said. "We'll start with a simple Redis-backed feature store for real-time serving, and Feast for batch feature computation. If we outgrow it, we migrate."

"Cost?"

"Engineer time. The tools are open source."

Marcus nodded. "Approved."

---

**Concepts introduced:** Feature store, feature computation consistency, point-in-time joins, data leakage in time-series ML, training-serving skew root cause, Feast (open-source feature store), Redis for real-time feature serving, batch vs. real-time features.

---

### Chapter 7: The Reset

Six months in, they had nothing in production.

Marcus had to report progress to the executive team. He chose brutal honesty.

"Here's what we learned," he told the room. "AI is not a library you import. It's an infrastructure problem. We failed because we treated model training as the hard part and everything else as an afterthought."

He walked them through the root causes:
- Training-serving skew
- Data silos across 12 systems
- No input validation
- No monitoring
- No feature consistency
- Unrealistic timelines

"We spent $2 million and have nothing to show for it," the CFO said.

"We have knowledge," Marcus countered. "And we have a plan. But we need to change how we think about this work. The question isn't 'how do we build AI.' The question is 'how do we build a system that can safely run AI in production.'"

The CFO wasn't convinced, but the CEO was. "Give me the plan for next year."

Anika presented. She didn't show slides. She drew the architecture on a whiteboard:

**Layer 1 — Data Foundation:** Data lakehouse with Delta Lake. Unified ingestion from all sources. Data quality validation. Feature catalog.

**Layer 2 — Training Infrastructure:** Feature store. Model registry. Experiment tracking. Distributed training on spot GPUs.

**Layer 3 — Serving Infrastructure:** Model deployment with canary releases. A/B testing. Autoscaling. Monitoring and drift detection.

**Layer 4 — Governance:** Model cards. Bias audits. Compliance. Input/output guardrails.

**The First Use Case:** Start over with a single, well-scoped problem. Customer support ticket triage with a RAG system. Lower risk, higher business value, measurable success metrics.

"One use case," Anika said. "We prove it works end-to-end. Then we scale."

"How long?" the CEO asked.

"Nine months to first production model. Eighteen months to scale."

"That's twice what you said last time."

"Last time I was wrong. This time, I've seen how it breaks."

The CEO studied the board. "Approved. But I want monthly progress reports. And if we're going to take two years to get this right, I want to see progress, not promises."

"That's fair."

---

**Concepts introduced:** Layered AI architecture (data → training → serving → governance), RAG as a safer starting point, phased delivery, realistic timelines, executive communication, learning from failure, the organizational cost of rushing.

---

When Anika returned to the team room, the whiteboard was covered. Elena had drawn the infrastructure. Raj had sketched the RAG pipeline. David had written the success metrics.

"Everyone's already working," Anika observed.

"We decided not to wait," Raj said. "I've been fine-tuning an embedding model for ticket classification."

Elena nodded. "I'm setting up the feature store POC."

David held up a spreadsheet. "And I have 10,000 labeled support tickets from the last year. Cleaned. Validated. Ready."

Anika smiled. "Then let's stop failing and start building."
