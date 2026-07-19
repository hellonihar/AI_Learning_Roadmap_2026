# Lesson 06.4.5: Governance for LLMs

## Learning Objectives
- Understand model cards, system cards, and their role in governance
- Identify the key regulatory frameworks affecting LLM deployment
- Implement usage policies and audit trails
- Analyze ethical considerations: bias, fairness, and transparency

---

## 1. The Governance Challenge

LLMs introduce governance challenges that traditional software does not:

- **Non-deterministic outputs:** The same input can produce different outputs.
- **Training data opacity:** Even the model creator cannot enumerate every fact in a 10TB dataset.
- **Emergent capabilities:** Models can do things the creators did not explicitly design for.
- **Distributed responsibility:** Who is liable — the model creator, the deployer, or the end user?

Governance is the framework of policies, documentation, and controls that answers these questions.

---

## 2. Documentation: Model Cards & System Cards

### Model Cards
Introduced by Mitchell et al. (2019), a model card is a standardized document for a trained model.

**Required sections:**
- **Model details:** Name, version, architecture, training data, hyperparameters.
- **Intended use:** What the model is designed for — and what it is NOT designed for.
- **Evaluation results:** Accuracy, bias metrics, performance across demographic groups.
- **Ethical considerations:** Known limitations, potential for misuse.
- **Training data:** Source, size, preprocessing, filtering.
- **Quantitative analysis:** Disaggregated evaluation (by gender, race, language, etc.).

**Example providers:**
- OpenAI: GPT-4 System Card (45 pages covering capabilities, risks, mitigations)
- Google: Gemini Technical Report
- Meta: Llama 3 Model Card

### System Cards
A system card extends the model card to the full deployed system, including:

- Guardrails and mitigations in place
- Deployment context (API, chatbot, embedded device)
- Monitoring and feedback mechanisms
- Human-in-the-loop processes
- Known failure modes and fallback behavior

---

## 3. Usage Policies

Define who can use the system and how.

**Acceptable Use Policy (AUP):**
- Prohibited use cases (e.g., generating malware, impersonation, automated decision-making in high-risk domains)
- Rate limits and user verification for high-volume access
- Licensing terms for commercial vs research use

**Internal Use Policy (for the organization deploying the LLM):**
- Approved model list (no shadow IT — employees cannot use unauthorized models)
- Data handling rules: never send sensitive data to external APIs without PII redaction
- Disclosure requirements: when and how to inform users they are interacting with AI

---

## 4. Audit Trails

Every LLM interaction should be logged for auditability.

| Field | Purpose |
|-------|---------|
| Request ID | Unique identifier for each interaction |
| User ID | Who made the request (anonymized if needed) |
| Timestamp | When the request occurred |
| Input (redacted) | The user prompt, with PII removed |
| Output (redacted) | The model response, with PII removed |
| Model version | Exact model + version used |
| Latency | Time taken for the response |
| Guardrail actions | Was the input or output modified / blocked? |
| Human review | Was this escalated to a human? What was the outcome? |

**Storage requirements:**
- Retain logs for a minimum period (varies by regulation; GDPR suggests 12 months)
- Immutable storage (write-once, append-only)
- Access controls: only authorized personnel can read raw logs

---

## 5. Regulatory Landscape

### EU AI Act (2024, enforcement phased through 2027)
The world's first comprehensive AI regulation. Key impacts on LLM governance:

| Risk Tier | Examples | Requirements |
|-----------|----------|--------------|
| **Unacceptable** | Social scoring, real-time biometric surveillance | Banned |
| **High-risk** | CV screening, credit scoring, medical devices | Conformity assessment, human oversight, risk management |
| **Limited risk** | Chatbots, AI-generated content | Transparency (label AI-generated content) |
| **Minimal risk** | AI in games, spam filters | No obligations |

**For LLM providers:** GPAI (General Purpose AI) models face additional requirements — publish training data summaries, implement copyright policy, conduct systemic risk assessments for models above 10²⁵ FLOPs.

### US Executive Order on AI (October 2023)
- Requires developers of "dual-use foundation models" to share safety test results with the US government
- Directs NIST to develop AI standards
- Mandates watermarking of AI-generated content
- Introduces the AI Safety Institute (AISI)

### Copyright Concerns
- **Training data:** Is using copyrighted material for training fair use? Multiple lawsuits pending (NYT v. OpenAI, Getty v. Stability AI).
- **Outputs:** Can AI-generated content be copyrighted? US Copyright Office says "no" for fully AI-generated works.
- **Mitigation:** Use opt-out mechanisms (e.g., robots.txt for web crawlers, OpenAI's opt-out form). Maintain training data provenance records. Implement output filters that detect and block copyrighted content (e.g., "David vs Goliath" style protection).

### Other Notable Regulations
- **China's AI Regulation:** Algorithm registration, content control, mandatory safety reviews.
- **Canada's AIDA (Artificial Intelligence and Data Act):** Similar tiered approach to EU AI Act.
- **Brazil's Bill 2338/2023:** Rights for individuals affected by AI systems.

---

## 6. Ethical Considerations

### Bias
LLMs can propagate and amplify biases present in training data.

**Detect:**
- Use disaggregated evaluation: how does the model perform across genders, dialects, and cultures?
- Template-based bias probing (e.g., "The [engineer/nurse] arrived late because ____")
- Real-world monitoring: Are certain user groups receiving worse answers?

**Mitigate:**
- Balanced fine-tuning datasets
- Controlled decoding (reduce probability of stereotypical next tokens)
- Regular bias audits

### Fairness
Fairness means different things in different contexts:

- **Equal treatment:** The model gives equally good answers to all users.
- **Equal opportunity:** The model does not disadvantage any group in high-stakes decisions.
- **Procedural fairness:** The process for handling AI failures is transparent and consistent.

### Transparency
- Users should know when they're interacting with AI.
- Users should understand what data is collected and how it's used.
- Users should have recourse when the model produces a bad outcome (a human review process).

---

## 7. Governance Checklist

- [ ] Model card published for every model in use
- [ ] System card published for every deployed application
- [ ] Acceptable Use Policy documented and enforced
- [ ] Audit trail implemented (immutable logs of all LLM calls)
- [ ] Data retention and deletion policies in place
- [ ] Bias evaluation completed (disaggregated by relevant demographics)
- [ ] Transparency disclosure ready ("this is AI-generated")
- [ ] Human review process defined for escalations
- [ ] Copyright compliance (training data provenance, opt-out mechanisms)
- [ ] Regulatory mapping complete (EU AI Act tier classification done)

---

## Summary

Governance for LLMs is about managing risk through documentation (model cards, system cards), policies (acceptable use, internal controls), audit trails, and regulatory compliance. Start with transparency — tell users they're talking to AI, log every interaction, and have a plan for when things go wrong. As regulations like the EU AI Act take effect, governance will move from best practice to legal requirement.

---

## Key Terms
- **Model Card:** Standardized documentation of a model's intended use, performance, and limitations
- **System Card:** Extended documentation that includes deployment context and safeguards
- **GPAI (General Purpose AI):** EU AI Act term for models like GPT-4 that can perform a wide range of tasks
- **Disaggregated evaluation:** Measuring model performance separately for different demographic groups
- **Procedural fairness:** Fairness of the processes and recourse mechanisms, not just the outcomes
