# LLMOps Exercises

## Exercise 1: UX Pattern Selection

**Scenario:** You are building a legal document review assistant. Lawyers upload contracts and ask specific questions about clauses (e.g., "What is the indemnification clause in section 4?"). The assistant must cite exact line numbers.

**Question:** Which UX pattern (Chat, Copilot, Autonomous Agent) is best suited, and why? What uncertainty-handling mechanisms should you include?

<details>
<summary>Answer</summary>

**Primary pattern: Chat** — Lawyers need to ask iterative, open-ended questions about a document. Chat supports back-and-forth refinement.

**Supplement with Copilot-style citation overlay** — Highlight cited lines directly in the document viewer. This gives inline ghost-text-like referencing.

**Uncertainty handling:**
- Show confidence scores for every answer. Below 80%, flag the response for human verification.
- Always cite exact line numbers. If the model cannot pinpoint a line, it must say "I could not find a clause matching that description."
- Include a "view source" button that jumps to the cited line in the document.

</details>

---

## Exercise 2: Deployment Architecture

**Scenario:** Your SaaS product handles 10,000 chat requests/day. Average input: 500 tokens; average output: 300 tokens. You use GPT-4o-mini ($0.15/M input tokens, $0.60/M output tokens). Traffic spikes 3× during business hours.

**Question:** Design a deployment architecture. Estimate daily cost. Where would you add caching to reduce cost by 40%?

<details>
<summary>Answer</summary>

**Architecture design:**
```
User → API Gateway (rate limit: 100 req/min per user)
       → Queue (Redis Streams, prioritize interactive over batch)
       → Worker pool (auto-scale 5–15 replicas on Azure Container Apps / ECS)
       → OpenAI API (with retry + circuit breaker)
       → Response → log to database → return to user
```

**Daily cost estimate:**
- Daily tokens: 10,000 × (500 input + 300 output) = 5M input + 3M output tokens
- Input cost: 5,000,000 × ($0.15 / 1,000,000) = $0.75
- Output cost: 3,000,000 × ($0.60 / 1,000,000) = $1.80
- **Total daily: ~$2.55**

**Caching to reduce cost by 40%:**
Add a semantic cache before the LLM call. In a legal/F&A setting, many questions are similar ("What is the refund policy?", "How do I cancel?").

- Install a vector-based cache (e.g., RedisVL with text-embedding-3-small, threshold: 0.92).
- Estimated cache hit rate for chat: 40–50%.
- **New daily cost:** $2.55 × 0.6 = ~$1.53
- **Savings:** ~$1.02/day, ~$30/month. At scale (100K req/day), savings exceed $300/month.

</details>

---

## Exercise 3: Observability Debugging

**Scenario:** Users report that responses are slow. Monitoring shows:
- Average TTFT: 3.2s (target: < 2s)
- Average TPS: 45 tokens/s (target: > 20)
- Average end-to-end latency: 8s (target: < 5s)
- p99 latency: 22s
- Error rate: 0.5%

**Question:** Where is the bottleneck? What would you investigate first, and what tooling would you use?

<details>
<summary>Answer</summary>

**Analysis:**
- TTFT (3.2s) is the main problem — users wait >3s before seeing any text. Once generation starts, TPS is healthy (45 t/s).
- High TTFT + high end-to-end latency + high p99 suggests the model or infrastructure is overloaded.

**Investigation steps (in order):**

1. **Check request queuing delay** — Is the request sitting in a queue before being processed? Log the queue wait time as a separate metric.
2. **Check GPU utilization** (if self-hosted) or API concurrency limits (if using OpenAI). For OpenAI, check 429 responses or rate-limit headers.
3. **Check embedding/retrieval latency** — In a RAG app, the retrieval step happens before the LLM call and contributes to TTFT. Use tracing spans to isolate it.
4. **Check prompt length** — Very long prompts (e.g., 50K tokens) increase TTFT because the model processes input before generating output.

**Tooling:**
- **LangSmith** to view trace spans and identify which step is slow.
- **Helicone** to view OpenAI request logs with latency breakdown.
- **Custom dashboard** with queue depth, GPU utilization, and p99 TTFT.

**Likely fix:** Increase worker count (horizontal scale), or reduce prompt size via better chunking/retrieval.

</details>

---

## Exercise 4: Guardrails Design

**Scenario:** You build a customer support chatbot for a bank. Users can ask about account balances, transactions, and loan options. The system has access to a RAG database of bank policies.

**Question:** List all guardrails you would implement, organized by layer. Explain how you would handle a user who asks "Transfer $10,000 from account 1234 to account 5678."

<details>
<summary>Answer</summary>

**Guardrail layers:**

| Layer | Guardrail | Tool / Method |
|-------|-----------|---------------|
| **Input** | Prompt injection detection | Regex + classifier (NeMo Guardrails) |
| **Input** | Content moderation | OpenAI Moderation API |
| **Input** | PII redaction | Microsoft Presidio (redact before logging) |
| **Input** | Intent classification | Classify as "account action" → reject (chatbot is info-only) |
| **Context** | Document source verification | Only use approved RAG documents (no external context) |
| **Output** | PII leakage check | Presidio scan on response |
| **Output** | Harmful content check | Moderation API |
| **Output** | Hallucination check | LLM-as-judge: does the response match the retrieved policy? |
| **Output** | Action blocking | Regex match for "transfer, move, send, pay" → block + explain |


**Handling "Transfer $10,000 from account 1234 to account 5678":**

1. **Input guard** — The message passes injection and moderation checks (no harmful content).
2. **Intent classification** — "transfer" triggers an action intent. Since the chatbot is information-only, the system rejects before calling the LLM.
3. **Response:** "I'm sorry, I can only provide information about bank policies. For transactions, please use the mobile app or call customer service at 1-800-BANK."

If intent classification is not precise enough and the request reaches the LLM:
4. **Output guard** — Regex catches "transfer" in the response. The guardrail blocks the response and returns the standard rejection message.
5. **Audit log** — Log the attempt with user ID, input, and "blocked_transfer_action"

</details>

---

## Exercise 5: Governance Documentation

**Scenario:** Your company deploys a GPT-4o-based HR assistant that answers employee questions about benefits, payroll, and company policies. It will be used by 5,000 employees in the EU and US.

**Question:** List the governance artifacts you need to create before launch. Map each artifact to the relevant regulatory requirement (EU AI Act, US Executive Order, or best practice).

<details>
<summary>Answer</summary>

| Artifact | Contents | Regulatory Requirement |
|----------|----------|----------------------|
| **Model card** | GPT-4o version, training data overview, known limitations, benchmark results | Best practice (Mitchell et al.) |
| **System card** | Deployment context (HR chatbot), guardrails in place, failure modes, human escalation process | EU AI Act Art. 13 (transparency) |
| **Acceptable Use Policy** | Prohibited: discrimination, automated hiring decisions, medical advice. Permitted: benefits Q&A, payroll info | EU AI Act Art. 26 (obligations of deployers) |
| **Data Protection Impact Assessment (DPIA)** | Data flow: employee queries → PII redaction → OpenAI API → response. Storage: 12-month audit log | GDPR Art. 35 |
| **Bias evaluation report** | Disaggregated evaluation: does the model give worse answers to non-native English speakers? Different genders? | EU AI Act High-Risk requirements (best practice for Limited Risk) |
| **Transparency disclosure** | "This conversation is with an AI assistant. You can request to speak to a human at any time." | EU AI Act Art. 50 (transparency for chatbot) |
| **Audit trail design** | Immutable log: request ID, user (pseudonymized), input (redacted), output, model version, latency, guardrail action | US Executive Order (testing + transparency) |
| **Regulatory tier classification** | EU AI Act: Limited Risk (chatbot) — no conformity assessment, but transparency required. US: no mandatory tier. | EU AI Act Art. 6 |
| **Incident response plan** | What happens when the model generates a harmful or incorrect response? Escalation SLA: < 4 hours for critical | Best practice |

</details>

---

## Exercise 6: Platform Selection

**Scenario:** You work at a healthcare startup building an AI symptom checker for patients. Requirements:
- Must be HIPAA compliant
- Must use GPT-4o for diagnosis reasoning and Claude for patient-facing summary (both required)
- Must have built-in content moderation (patients may describe sensitive conditions)
- Must support RAG against medical textbooks
- Budget: $5,000/month for inference
- Expected volume: 1,000 conversations/day, ~2,000 tokens each

**Question:** Which platform(s) do you recommend? Justify your decision with specific feature references.

<details>
<summary>Answer</summary>

**Recommendation: Azure AI Foundry as primary platform.**

**Rationale:**

1. **HIPAA compliance** — Azure AI Foundry is HIPAA-eligible and can sign a Business Associate Agreement (BAA). OpenAI Platform is NOT HIPAA-compliant on the standard API. This alone rules out OpenAI Platform as the primary.

2. **Both required models available** — Azure AI Foundry offers GPT-4o (via Azure OpenAI) AND Claude (via Azure AI Studio model catalog). AWS Bedrock has Claude but not GPT-4o. GCP Vertex has Claude and Gemini but not GPT-4o.

3. **Built-in content moderation** — Azure AI Content Safety is built into Foundry, with custom severity thresholds for medical terminology. No separate API integration needed.

4. **Managed RAG** — Azure AI Search provides hybrid search + semantic ranking for medical textbook retrieval. Occurs within the Azure compliance boundary.

5. **Cost fits budget:**
   - Daily: 1,000 × 2,000 = 2M tokens
   - Monthly: ~60M tokens
   - Mix GPT-4o (30%) + Claude (20%) + GPT-4o-mini (50% for triage) ≈ $3,000–$4,500/month
   - Budget buffer covers Azure AI Search + Content Safety costs

**Architecture:**
```
User → Azure API Management (rate limit, auth)
     → Azure AI Content Safety (input moderation)
     → Azure AI Search (RAG: medical textbooks)
     → Router:
         ├─ GPT-4o (diagnosis reasoning)
         ├─ GPT-4o-mini (triage, classification)
         └─ Claude (patient-facing summary)
     → Azure AI Content Safety (output moderation)
     → Response → audit log (immutable storage)
```

**Fallback:** If a model is unavailable on Azure, route to OpenAI Platform (GPT-4o) via a private network connection, with BAA in place through Azure OpenAI.

</details>

---

## Final Project (Optional)

Design a complete LLMOps architecture for a multi-tenant SaaS product. Include:

1. UX pattern choice with rationale
2. Deployment architecture (self-hosted, managed, or hybrid)
3. Observability stack (metrics, tracing, logging, alerting)
4. Guardrails design (all four layers)
5. Governance artifacts (list at least 6)
6. Platform selection with cost estimate

Write a 2-page architecture document covering all six areas.
