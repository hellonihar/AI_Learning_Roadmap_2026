# Behavioral & Scenario-based — Q&A

## Q1: Tell me about a time you disagreed with a product manager on a technical decision.

**STAR:**

**Situation:** PM wanted to use a single large LLM for both chatbot responses and content summarization. I argued for two separate models optimized per task.

**Task:** Resolve the disagreement without blocking the project.

**Action:**
1. Scheduled a 1-hour data-driven debate, not an opinion argument
2. Built a side-by-side comparison: one model vs. two models on latency, cost, and quality
3. Demonstrated: two-task model had 2x latency for summarization and 30% lower quality on chatbot, while being 20% cheaper to run with separate models
4. Proposed a compromise: start with single model for MVP, split in v2 when scale demands it

**Result:** PM agreed to single-model for MVP but added the split to v2 roadmap. MVP shipped on time, v2 split delivered 40% latency improvement.

---

## Q2: Describe a model that performed well in testing but failed in production.

**STAR:**

**Situation:** A fraud detection model had 99% precision and 95% recall in offline testing. In production, false positive rate was 5x higher.

**Task:** Diagnose and fix the production degradation.

**Action:**
1. Shadow-deployed the model alongside existing system — compared predictions
2. Found training-serving skew: production features had 2-second delay, but training assumed real-time
3. Identified label leakage: training data included information only available 24h after transaction
4. Retrained with production-accurate feature pipeline and corrected labels
5. Added feature distribution monitoring (KS test) and automated retraining trigger

**Result:** False positive rate returned to expected levels. Built a monitoring system that caught similar issues in future deployments.

**Lesson:** Offline metrics can lie. Always shadow-deploy and compare before cutting over.

---

## Q3: Tell me about a technical decision you made that turned out wrong.

**STAR:**

**Situation:** I chose to fine-tune a small LLM (7B) for a document extraction task instead of using RAG with a large model.

**Task:** Extract structured data from legal documents.

**Action:** Fine-tuned Llama-2-7B on 10K labeled documents. Invested 3 weeks in data prep, training, and evaluation.

**Result:** The fine-tuned model worked well on seen document formats but failed on unseen formats. Maintenance cost was high (needed new training data for each format change). The team eventually switched to GPT-4 with RAG, which handled unseen formats immediately.

**Lesson:** Before choosing a complex solution, evaluate if a simpler approach (like prompt engineering with a capable model) can solve the problem. I now start with the simplest viable approach and escalate only when proven insufficient.

---

## Q4: Describe how you handled a stakeholder wanting unrealistic AI capabilities.

**STAR:**

**Situation:** An executive expected a GenAI chatbot to handle 100% of customer queries autonomously within 2 months, including complex billing disputes and legal questions.

**Task:** Reset expectations without demoralizing the team or losing executive sponsorship.

**Action:**
1. Presented a data-driven breakdown: 40% simple queries (doable in 2 months), 35% medium (needs RAG + human fallback), 25% complex (requires legal review)
2. Proposed phased rollout: Phase 1 (automate 40%) in 2 months, Phase 2 (automate 75%) in 4 months, Phase 3 (handle complex with escalation) in 6 months
3. Showed competitive analysis — no company achieves full autonomy; best-in-class automates 60-70%
4. Demonstrated a prototype on 2 simple query types to build confidence

**Result:** Executive agreed to phased approach. Phase 1 delivered 42% automation in 2 months, building trust for Phase 2.

---

## Q5: Tell me about a time you mentored a junior engineer.

**STAR:**

**Situation:** A new ML engineer joined my team, strong in theory but inexperienced with production ML systems.

**Task:** Help them become independently productive within 3 months.

**Action:**
1. First 2 weeks: pair programming sessions — building a simple inference endpoint together
2. Created a structured onboarding guide covering our MLOps stack, debugging patterns, and monitoring dashboards
3. Assigned progressively independent tasks: first a bug fix (supervised), then a feature addition (reviewed), then a new model deployment (autonomous)
4. Weekly 30-minute mentoring sessions for questions, code review feedback, career advice
5. Taught debugging methodology: "When something breaks, explain to me what you know, what youve checked, and what you suspect"

**Result:** Engineer was independently deploying models by month 3. Promoted to mid-level at 6 months. Later became a mentor themselves.

---

## Q6: Describe a cross-functional initiative you led across multiple teams.

**STAR:**

**Situation:** Our company had 3 product teams building separate AI features, each maintaining their own infrastructure. Cost was high, best practices were inconsistent.

**Task:** Unify AI infrastructure across teams to reduce cost and improve standards.

**Action:**
1. Interviewed each team lead — understood their pain points and non-negotiables
2. Proposed a shared ML platform team with common infrastructure (feature store, model registry, GPU cluster)
3. Built a proof of concept showing 30% cost reduction and 2x faster deployment
4. Navigated resistance: "we'll lose control" ? addressed by keeping model logic in product teams, only infrastructure shared
5. Led the migration: one team at a time, with 2-week overlap period

**Result:** $500K annual cost savings, deployment time reduced from weeks to days, consistent monitoring and governance across all teams.

---

## Q7: How do you approach a problem when requirements are unclear?

**Framework:**

1. **Ask clarifying questions:**
   - "Who are the users and what problem are we solving for them?"
   - "What does success look like? How will we measure it?"
   - "What is the minimum useful version?"
   - "By when do we need this?"

2. **Build understanding iteratively:**
   - Create a lightweight spec (1-page) and get sign-off
   - Build the smallest possible prototype (hours, not days)
   - Put it in front of real users or stakeholders
   - Iterate based on feedback

3. **Manage ambiguity with structure:**
   - Make assumptions explicit and document them
   - Use time-boxed exploration (e.g., "I will spend 2 days investigating 3 approaches, then recommend")
   - Escalate blockers: "I have 2 viable paths but need a decision on X to proceed"

**Key mindset:** "I don't need perfect requirements. I need to know the direction and the first step."

---

## Q8: Scenario — A production model starts generating harmful outputs. Walk through your response.

**Immediate (<5 minutes):**
1. **Disable the offending output path** — route to a safe fallback or block responses
2. **Assess blast radius** — how many users affected? What kind of harmful content?
3. **Notify on-call + security team**

**Short-term (<1 hour):**
4. **Investigate root cause** — was it a prompt injection? Model drift? Data poisoning? Configuration error?
5. **Deploy temporary fix** — add stricter guardrails, block specific prompt patterns
6. **Communication** — if user-facing, prepare incident communication

**Medium-term (<1 week):**
7. **Patch permanent fix** — update model, retrain, or add permanent guardrails
8. **Conduct post-mortem** — what happened, why, how to prevent
9. **Implement new testing** — add this scenario to red-teaming suite
10. **Review guardrail coverage** — are there other attack surfaces?

**Key principle:** Safety over revenue. Don't hesitate to take the system down if needed.

---

## Q9: Scenario — Your training cluster goes down 2 weeks before a major deliverable.

**Response:**

1. **Diagnose** — is it hardware failure, networking, software, or cloud provider issue?
2. **Recover** — if hardware, switch to backup nodes. If cloud, move region. If software, fix the bug.
3. **Mitigate:**
   - Checkpoint available? Resume from last checkpoint on healthy cluster
   - Can you reduce model size to train faster? (Smaller model, shorter context)
   - Can you use a cheaper/available GPU type?
   - Split training across multiple smaller clusters (elastic training)
4. **Communicate** — be transparent with stakeholders: "We have X% complete, delayed by Y days. Here's our recovery plan."
5. **Prevent recurrence:**
   - Multi-region training capability for critical projects
   - Regular checkpointing (every N steps to S3, not local disk)
   - Backup cluster agreement with cloud provider
   - Retry logic in training orchestration

---

## Q10: Scenario — A key vendor changes pricing overnight. What do you do?

**Immediate:**
1. Assess impact — how much will cost increase? How much buffer in budget?
2. Lock in current pricing if possible (annual commitment before new pricing kicks in)

**Short-term (1 week):**
3. Evaluate alternatives — competitive APIs, open-source self-hosted options
4. Implement abstraction layer if not present (decouple code from vendor API)
5. Negotiate with vendor — volume discounts, multi-year commitment, alternative pricing tiers

**Medium-term (1-3 months):**
6. Implement model routing — use vendor for some traffic, open-source for other
7. Add caching layer to reduce vendor API calls
8. Evaluate fine-tuning smaller open-source models to replace vendor model

**Long-term:**
9. Reduce vendor dependency — diversify across providers, build in-house fallbacks
10. Budget for price volatility as a standard risk factor

**Key principle:** Always have a Plan B. Every vendor dependency should have a documented migration path.

---

## Q11: Scenario — Your model has 95% overall accuracy but 70% on a minority subgroup.

**Investigation:**
1. **Confirm the gap** — is it statistically significant? How large is the subgroup?
2. **Identify the cause:**
   - Data imbalance: subgroup underrepresented in training data
   - Label quality: are labels less reliable for this subgroup?
   - Feature representation: do features capture relevant signals for this subgroup?
   - Model architecture: is the model optimized for majority by default?

**Remediation:**
3. **Data:** Oversample subgroup, collect more diverse data, reweight training samples
4. **Model:** Add fairness constraint to loss, train separate model for subgroup, adjust decision threshold
5. **Process:** Add subgroup evaluation to standard model validation checklist
6. **Monitoring:** Track accuracy per subgroup in production, alert on drift

**Communication:**
7. Be transparent with stakeholders. Don't ship a model with known bias without disclosure.
8. If launching anyway (with mitigations), document clearly and commit to improvement timeline.

---

## Q12: How do you stay current with AI advancements? What's your system?

**My system:**

**Daily (15 min):**
- Skim ArXiv Sanity or Twitter/Bluesky for important papers
- Read 1-2 posts from curated RSS feed (The Batch, TLDR AI, Import AI)

**Weekly (1-2 hours):**
- Read 2-3 papers deeply (abstract, method, results, limitations)
- Experiment with new tools/models (try a new API, run a small notebook)
- Review summaries from AI newsletters

**Monthly (half day):**
- Build a small project with a new technique (e.g., try DPO on a custom dataset)
- Attend or watch recording of a conference talk
- Write/reflect on what I learned

**Key approach:** Depth over breadth. I don't try to follow everything. I pick 2-3 areas relevant to my work and go deep. For everything else, summaries and high-level awareness suffice.

**Filtering:** Apply the "will this matter in 6 months?" test before investing time.

---

## External Resources

- [HireFlow: LLM Engineer Interview Questions (2026)](https://hireflow.net/interview-questions/llm-engineer)
- [Interview Guys: AI Solutions Architect Questions (2026)](https://blog.theinterviewguys.com/ai-solutions-architect-interview-questions/)
- [CoPrep: Top AI Engineer Questions (2026)](https://www.coprep.ai/blog/top-ai-engineer-interview-questions-in-2026-llms-rag-agents-and-langchain)
- [Maywise: 5 AI Interview Questions Guide (2026)](https://maywise.in/blog/5-ai-interview-questions-2026-guide)
- [IGotAnOffer: ML System Design Interview Guide](https://igotanoffer.com/en/advice/machine-learning-system-design-interview)
- [GitHub: aakriti1318/interview_questions](https://github.com/aakriti1318/interview_questions)
