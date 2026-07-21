# Phase 5: The Scale
## Months 15–20

---

### Chapter 16: The Second System Effect

Success brought more success. Three more teams wanted AI solutions:

1. **Fraud Detection** — real-time flagging of suspicious account activity
2. **Recommendation Engine** — personalized product suggestions for the sales portal
3. **Document Extraction** — second attempt, this time with lessons learned

The team grew from 5 to 25. Elena's platform team had 8 engineers. Raj's ML team had 6. Anika added a governance team with 2 ML engineers focused on responsible AI.

But with growth came chaos.

"Each team is making independent infrastructure choices," Elena reported. "Fraud team deployed on SageMaker. Recommendations team built on Vertex AI. Document extraction wants Bedrock. We have three clouds, three serving stacks, no governance."

"The second system effect," Anika said.

"The what?"

"When teams get autonomy without standards, each one builds their own version of the same thing. Different clouds, different MLOps tools, different monitoring. It's the opposite of our first project — instead of one fragile system, we have four incompatible ones."

"What's the fix?"

"A **unified AI platform** with a **routing layer**. Teams retain autonomy over their models, but infrastructure is shared."

Anika drew the design:

```
User Request
    │
    ▼
Orchestration Layer
    ├─→ Intent routing (which model?)
    ├─→ Load balancing (which GPU node?)
    └─→ Cost tracking (who pays?)
    │
    ▼
Serving Layer
    ├─→ vLLM (LLMs)
    ├─→ Triton Inference Server (non-LLM models)
    └─→ SageMaker / Vertex AI (if existing investment)
    │
    ▼
Shared Components
    ├─→ Feature Store (Feast)
    ├─→ Model Registry (MLflow)
    ├─→ Monitoring (Prometheus + Evidently)
    ├─→ Vector DB (Qdrant)
    └─→ GPU Cluster (K8s with KubeFlow)
```

"Each team controls their own model and training pipeline. But inference goes through our shared platform. This gives us reliability, cost control, and governance."

"And the teams that already deployed on their own cloud?"

"We migrate them. Gently. One at a time. Over three months."

---

**Concepts introduced:** Second system effect, decentralized infrastructure sprawl, unified AI platform architecture, model orchestration/routing layer, multi-model serving (vLLM + Triton Inference Server), shared vs. team-owned infrastructure, migration strategy for existing deployments.

---

### Chapter 17: The Cost Wake-Up

Marcus got the monthly cloud bill and called an emergency meeting.

"$450,000. Last month it was $180,000. What happened?"

Elena pulled up a cost breakdown. "Three things:
1. Fraud detection team is running 8x A100s 24/7 for real-time inference. They only need them during business hours.
2. Each team reserved their own GPU instances. We're paying for 40 GPUs but only using 60% capacity.
3. Nobody optimized model sizes. Some teams are running FP16 models when INT4 would suffice."

"What do we change?"

"First, **right-sizing**," Elena said. "Autoscaling GPU node pools. Spot instances for non-critical workloads."

"When?"

"Immediately."

Raj added: "Second, **model optimization**. We can quantize most models to INT4 with minimal quality loss. 4x memory reduction, 2-3x speedup. The fraud model was serving on 8 GPUs. Quantized, it runs on 2."

"What's the quality impact?"

"We tested. <1% accuracy drop on the fraud model. 3% quality drop on the LLM. Acceptable for most use cases."

"Third?"

"**GPU scheduling**," Elena said. "Not all models need 24/7 availability. Batch inference runs on spot. Development runs on preemptible nodes. Only critical real-time inference gets reserved capacity."

She showed the optimized architecture:

```
GPU Pool Types:
├─ Reserved (30%): Critical real-time inference
├─ On-demand (30%): Standard production workloads
├─ Spot (30%): Training, batch inference, dev
└─ Preemptible (10%): Experimentation, CI/CD
```

"Expected savings?"

"60%. $450K → $180K/month. And we'll add autoscaling so unused GPUs don't cost anything."

Marcus nodded. "Do it."

---

**Concepts introduced:** GPU cost optimization, right-sizing instances, autoscaling GPU pools, spot vs. reserved vs. on-demand GPU instances, model quantization for cost reduction (FP16 → INT4), quality-cost trade-off in quantization, GPU scheduling for different workload priorities, cloud cost allocation.

---

### Chapter 18: Governance Awakening

The CEO's office sent a memo: "We need to talk about responsible AI."

A customer had complained that the recommendation engine was surfacing higher-priced products to women than to men for identical searches. The press hadn't picked it up yet, but legal was nervous.

Anika's governance team investigated.

The finding: the model had learned historical purchase patterns where women in the training data had been upsold more aggressively. The model replicated this bias.

"We have a fairness problem," Anika told the team. "And we have no framework for detecting it."

She created the **NovaTech Responsible AI Framework**:

**Design Phase:**
- Fairness goals defined per use case
- Data audit for representation bias
- Sensitive attribute identification

**Development Phase:**
- Bias evaluation across demographic slices
- Model cards with fairness metrics
- Documentation of limitations

**Deployment Phase:**
- Input guardrails (toxicity, PII injection)
- Output guardrails (toxicity, hallucination, bias)
- Red-teaming before launch

**Monitoring Phase:**
- Fairness metrics tracked in production
- Drift detection per demographic group
- Quarterly bias audits

"Starting immediately," Anika announced. "Every model gets a model card. Every deployment goes through a fairness check. No exceptions."

Raj raised his hand. "Our RAG system for support — it handles tickets from all customers. Should we check that?"

"Great question," Anika said. "Let's do the audit."

They ran the audit. Found two issues:
1. Tickets in Spanish were handled 12% less accurately (fewer Spanish-language support articles in the knowledge base)
2. Enterprise customers got faster responses (their tickets used more technical terms that matched the knowledge base better)

Both were fixable. But they would never have known without the audit.

"If the press had found this before we did," Anika said, "we'd be in crisis mode. Instead, we fix it proactively."

She looked at the team. "Governance isn't a blocker. It's a shield."

---

**Concepts introduced:** Algorithmic bias in production, training data historical bias, fairness evaluation framework, model cards, responsible AI lifecycle (design → develop → deploy → monitor), bias detection across demographic groups, red-teaming, proactive vs. reactive governance, language bias in RAG systems, enterprise vs. consumer bias.

---

The governance framework revealed another gap: the **EU AI Act** applied to NovaTech's European customers. Their system was classified as "limited risk" — transparency obligations applied.

Anika worked with legal to create a compliance checklist:
- **Transparency:** All AI interactions clearly labeled to users
- **Explainability:** Ability to explain why a recommendation was made
- **Human oversight:** Human review for high-stakes decisions
- **Documentation:** Model cards + system cards for every deployment
- **Data governance:** Clear data sourcing and retention policies

"Six months ago, I thought compliance was someone else's problem," David admitted.

"It's everyone's problem now," Anika said. "That's what it means to have AI in production."
