# Chat-Based Coding Assistants

Conversational AI tools used for code generation, debugging, and explanation.

## Claude (Anthropic)

| Strength | Detail |
|---|---|
| Context window | 200K tokens (can process entire codebases) |
| Code quality | Strong at architecture, design patterns, refactoring |
| Reasoning | Excellent step-by-step debugging |
| Artifacts | Generates runnable code in a sandbox |

**Prompt:** "Claude, here's my full error log and code. Find the root cause."

## ChatGPT (OpenAI)

| Strength | Detail |
|---|---|
| Versatility | Code, data analysis, image generation |
| GPT-4o | Strong general-purpose coding |
| Code interpreter | Run Python in-browser, upload files |

## Gemini (Google)

| Strength | Detail |
|---|---|
| Integration | Deep with Google ecosystem (Colab, GCP) |
| Context | 1M token context window |
| Multimodal | Understands screenshots, diagrams, UI mockups |

## When to Use Chat vs IDE Assistant

| Scenario | Tool |
|---|---|
| Quick inline completion | IDE assistant (Copilot) |
| Debugging a complex error | Chat (Claude, ChatGPT) |
| Refactoring across multiple files | Chat with full codebase |
| Generating a new project structure | Chat |
| Learning a new codebase | Chat |
| Writing boilerplate | IDE assistant |
