# Lesson 06.4.1: UX & Product Design for LLM-Powered Products

## Learning Objectives
- Understand when to choose prompting vs fine-tuning from a product perspective
- Identify the three core UX patterns for LLM applications
- Design for uncertainty, streaming, and feedback loops
- Structure prompt chains for multi-step user workflows

---

## 1. Prompting vs Fine-Tuning: A Product Decision

The first architectural decision in any LLM-powered product is whether to rely on prompting or to fine-tune a base model. This choice has profound UX implications.

| Factor | Prompting | Fine-Tuning |
|--------|-----------|-------------|
| Iteration speed | Minutes (edit prompt) | Hours–days (training job) |
| Cost per use | Higher (more tokens) | Lower (smaller model possible) |
| Reliability | Variable; prompt-injection surface | More consistent behavior |
| Data privacy | Prompts sent to API provider | Model weights retain training data |
| Skill required | Prompt engineering | ML engineering, data curation |

**Product heuristic:** Start with prompting. Only fine-tune when you need to (a) reduce latency/cost at scale, (b) enforce a narrow output style that prompting cannot reliably produce, or (c) operate in a regulated setting where a static, auditable model is required.

---

## 2. Three Core UX Patterns

### 2.1 Chat Interface
- **Best for:** Open-ended tasks, iterative refinement, customer support.
- **Design considerations:** Clear turn indicators, editable previous messages (to fix misunderstandings), suggested follow-ups, and "regenerate" affordance. Users expect conversational memory—be transparent about context windows.
- **Anti-pattern:** Making the chat feel like a person while simultaneously disclaiming it is not a person. Avoid anthropomorphic avatars; use bot indicators instead.

### 2.2 Copilot (Inline Assistance)
- **Best for:** Writing, coding, design tools where the user stays in control.
- **Design considerations:** Invocation on demand (Tab/Enter to accept, Esc to dismiss), ghost text / inline suggestions, "explain this" side panels. The user remains the active driver; the model assists.
- **Anti-pattern:** Automatically inserting text without user confirmation—erodes trust and creates cleanup work.

### 2.3 Autonomous Agent
- **Best for:** Multi-step tasks, background processing, workflow automation.
- **Design considerations:** Transparent reasoning (show the plan, show tool calls), ability to pause/interrupt, confirmation gates before destructive actions, and persistent status indicators. Agents that "think in loops" need time-visibility (spinner with step count, ETA estimates).
- **Anti-pattern:** Black-box execution where the user has no insight into failures or intermediate decisions.

---

## 3. Handling Uncertainty

LLMs are probabilistic—every response carries uncertainty. Your UX must acknowledge this.

- **Confidence scores:** When classifying or extracting, surface a confidence score. Below a threshold, ask the user to verify ("I'm only 60% sure this is a support ticket — does this look right?")
- **Disclaimers & citations:** Always cite sources for factual claims. Add subtle disclaimers at the point of friction, not as blanket banners users ignore.
- **"I don't know":** Teach the model to say "I'm not sure" rather than hallucinate. Reward uncertainty expression via RLHF or system prompts.

---

## 4. Streaming Responses for UX

Perceived latency matters more than actual latency. Streaming tokens one-by-one reduces time-to-first-token perception from ~3s to ~200ms.

**Implementation checklist:**
- Use Server-Sent Events (SSE) or WebSockets for real-time token delivery.
- Render tokens incrementally with a stable layout (avoid layout shift as text appears).
- Show a cursor/blinking indicator while tokens are being generated.
- Allow the user to "stop generation" mid-stream.
- Buffer punctuation to smooth rendering (render full sentences, not half-words).

**Fallback:** If streaming is not possible (e.g., legacy infrastructure), show a skeleton loader that mimics the expected output shape (e.g., bullet-point placeholders).

---

## 5. Feedback Loops

Feedback is the lifeblood of LLM product improvement.

- **Explicit feedback:** Thumbs up/down per response. Collect a free-text follow-up when thumbs-down. Store these as labeled datasets for future fine-tuning or prompt tuning.
- **Implicit feedback:** Copy-paste events, time spent reading, regeneration rate, edit distance between suggestion and final output. These signals are noisier but abundant.
- **Loop closure:** Route low-scoring responses to a human-in-the-loop review queue. Periodically sample high-scoring responses to validate that quality hasn't regressed.

---

## 6. Prompt Chains in UX

Complex tasks often require multiple LLM calls. How you expose this to the user matters.

| Pattern | Description | UX Example |
|---------|-------------|------------|
| Linear chain | Step A → B → C, all auto | "Summarize this article" (extract → summarize → format) |
| Conditional branching | Route based on intermediate result | "Analyze sentiment" → positive (suggest products) / negative (escalate to support) |
| User-in-the-loop | Pause chain for user input at a decision point | "Draft email" → user edits tone → "Send" |
| Map-reduce | Split input, process in parallel, combine | "Summarize 10 documents" (parallel summaries → merge) |

**UX principles for chains:**
- Show progress: "Step 2 of 4: Analyzing document..."
- Allow cancellation or rollback to a previous step.
- Cache intermediate results so users can explore branches without re-running the full chain.

---

## Summary

Great LLM product design bridges the gap between what models can do and what users can trust. Start simple (chat), iterate with feedback, and only add complexity (agents, chains, fine-tuning) when the product need demands it. Always design for uncertainty—transparency about model limitations builds more trust than pretending the model is perfect.

---

## Key Terms
- **Ghost text:** Inline suggestion shown in dimmed text before user accepts
- **SSE:** Server-Sent Events, a standard for streaming text over HTTP
- **Confidence threshold:** The probability cutoff below which the model defers to the user
- **Anthropomorphic:** Attributing human characteristics to AI
