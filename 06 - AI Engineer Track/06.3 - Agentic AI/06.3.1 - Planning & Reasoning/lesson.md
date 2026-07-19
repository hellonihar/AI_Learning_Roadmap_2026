# 06.3.1 — Planning & Reasoning

Agentic AI systems must do more than generate text — they must reason about their environment, decompose complex tasks, and adapt when plans fail. This lesson covers the core reasoning paradigms that power modern agents.

## The ReAct Paradigm

**ReAct (Reasoning + Acting)**, introduced by Yao et al. (2023), interleaves reasoning traces with action steps. Unlike a standard LLM that produces a single answer, a ReAct agent cycles through:

1. **Thought**: "I need to find the user's order status. The order ID is #4421."
2. **Action**: `search_orders("4421")`
3. **Observation**: "Order #4421 is marked as shipped on June 12."
4. **Thought**: "The order has shipped. I should provide the tracking link."
5. **Action**: `get_tracking("4421")`
6. **Observation**: "Tracking: 1Z999AA10123456784"
7. **Final Answer**: "Your order #4421 shipped on June 12. Tracking: 1Z999AA10123456784"

The key insight is that the *thought trace* lets the LLM reason about what it knows and what it still needs, while the *action* lets it gather real-world information. This dramatically reduces hallucination compared to answering from parametric knowledge alone.

### ReAct Prompt Structure

A typical ReAct prompt includes:

- A description of available tools (name, description, parameters)
- Few-shot examples showing Thought/Action/Observation cycles
- Instructions to stop and produce a Final Answer when sufficient information is gathered

When using function-calling APIs (OpenAI, Anthropic), the "Thought" is implicit in the model's reasoning, and tool calls are structured JSON.

## Chain-of-Thought (CoT) Planning

**Chain-of-Thought** prompting, introduced by Wei et al. (2022), asks the model to produce intermediate reasoning steps before arriving at an answer. In the agentic context, CoT serves two purposes:

1. **Plan Generation**: The agent writes a step-by-step plan before executing any tool calls.
2. **Decomposition**: Complex goals are broken into atomic steps the agent can verify.

Example CoT plan:

```
Goal: "Book a flight to Tokyo for next Friday, budget under $800."
Plan:
1. Search flights from user's home airport to Tokyo-Narita for next Friday
2. Filter results by price under $800
3. If multiple options exist, compare duration and layovers
4. Present top 3 options to user
5. Wait for user selection
6. Book the selected flight
```

CoT works well when the plan can be determined upfront. Its limitation is that it does not naturally handle *contingencies* — if step 2 returns zero results, the original plan breaks.

## Tree-of-Thoughts (ToT)

**Tree-of-Thoughts** (Yao et al., 2023) extends CoT by exploring multiple reasoning paths simultaneously. At each step, the model generates several candidate "thoughts," evaluates them, and decides which branches to pursue.

ToT introduces three operations:
- **Branch**: Generate N possible next steps from the current state
- **Evaluate**: Score each branch using the LLM itself (e.g., "sure/likely/impossible")
- **Search**: BFS or DFS over the tree, keeping only the most promising paths

This is especially useful for tasks requiring search or planning with irreversible decisions — for example, solving puzzles, writing multi-file code changes, or navigating complex multi-step workflows where early mistakes compound.

**Practical trade-off**: ToT requires many more LLM calls than ReAct or CoT, increasing both latency and cost. It is best reserved for high-stakes planning where correctness justifies the overhead.

## Plan-and-Solve

**Plan-and-Solve** (Wang et al., 2023) separates planning from execution more explicitly. The agent first produces a complete plan, then executes it step by step, verifying each step before proceeding.

```
Phase 1 — Plan:
  "I will:
    1. Extract all customer emails from the CRM
    2. Categorize by sentiment
    3. Draft responses for negative sentiment emails
    4. Send drafts for approval"

Phase 2 — Solve (with verification):
  Step 1: extract_emails() → success (342 emails)
  Step 2: categorize(emails) → success
  Step 3: draft_responses(negatives) → success
  Step 4: send_for_approval(drafts) → success
```

The advantage is that the plan acts as a **checklist** — if execution deviates, the agent can detect the drift. The disadvantage is rigidity: if the environment changes during execution, the original plan may become invalid.

## Task Decomposition

All reasoning paradigms rely on **task decomposition** — breaking a high-level goal into smaller, executable subtasks.

### Hierarchical Decomposition

Decompose the top-level task recursively:

```
"Prepare quarterly report"
  ├── Gather data from Salesforce
  ├── Analyze Q2 metrics
  │   ├── Revenue growth
  │   ├── Customer acquisition cost
  │   └── Churn rate
  ├── Generate charts
  └── Write executive summary
```

Each subtask can itself be an agent call or a tool invocation.

### Dynamic Decomposition

Not all tasks decompose cleanly ahead of time. Dynamic decomposition lets the agent decide the next subtask based on the current state — essential for open-ended tasks like research or debugging.

## Subgoal Planning

Subgoal planning identifies intermediate states that must be reached before the final goal. Each subgoal becomes a checkpoint:

- **Subgoal 1**: Retrieve user profile → **check**: profile found?
- **Subgoal 2**: Retrieve recent transactions → **check**: within date range?
- **Subgoal 3**: Calculate refund amount → **check**: matches policy?
- **Subgoal 4**: Process refund → **check**: success confirmation?

Checkpoints allow rollback: if Subgoal 3 fails, the agent can ask for clarification without repeating Subgoals 1-2.

## Dynamic Replanning

Real-world agents must handle failure. When a tool returns an error, an API is down, or data is missing, the agent should not crash — it should **replan**.

### Replanning Strategies

1. **Retry with backoff**: If a tool call fails due to rate limiting, wait and retry.
2. **Alternative tool**: If `search_flights` fails, try `search_flights_v2` or a different provider.
3. **Partial completion**: If only some subtasks succeed, return partial results and explain.
4. **Human handoff**: If the agent cannot recover, escalate to the user with context.

### Replanning in ReAct

ReAct naturally supports replanning because the Thought-Action-Observation loop continues until the goal is met. After a failed observation, the next Thought can be:

```
Thought: "The search API returned a 503 error. I'll try the backup API."
Action: search_flights_backup(origin="JFK", destination="NRT", date="2026-07-24")
```

### Replanning with CoT/Plan-and-Solve

For plan-based approaches, replanning requires a specific **replan trigger**:

```python
if observation indicates failure:
    revise_plan(original_plan, failed_step, observation)
```

The LLM is prompted to analyze which step failed, why, and how to adjust the remaining steps.

## Choosing a Reasoning Paradigm

| Paradigm | Strengths | Weaknesses | Best For |
|----------|-----------|------------|----------|
| ReAct | Flexible, handles dynamic environments well | Can get stuck in loops without termination checks | Customer support, research, data extraction |
| CoT | Simple, interpretable, low cost | Brittle when plan needs to change | Well-understood tasks with predictable steps |
| ToT | Explores alternatives, finds optimal paths | Expensive, slow | Puzzles, code generation, strategy |
| Plan-and-Solve | Verifiable, good for compliance | Rigid, assumes static environment | Regulated workflows, document processing |

## Key Takeaways

- ReAct is the foundational paradigm for most modern agent systems — it interleaves reasoning with action.
- CoT improves answer quality by forcing intermediate reasoning; combine it with ReAct for best results.
- ToT explores multiple paths but is costly; use it selectively.
- Plan-and-Solve separates planning from execution for auditability.
- Every agent needs a replanning strategy — failure handling is not optional.
- Task decomposition is the core skill; an agent that cannot break down a task cannot execute it.

In the next section (06.3.2), we explore how agents remember past interactions and maintain state across turns.
