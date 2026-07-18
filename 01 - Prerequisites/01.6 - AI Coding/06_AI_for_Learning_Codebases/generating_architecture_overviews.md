# Generating Architecture Overviews

Ask AI to analyze codebases and produce architectural summaries.

## Diagram Generation

```
Prompt: "Analyze the data flow in this project and describe it
as a Mermaid sequence diagram"
```

```mermaid
sequenceDiagram
    User->>API: POST /chat {message, user_id}
    API->>Auth: Validate token
    Auth-->>API: user_id
    API->>Memory: Get conversation history
    Memory-->>API: [{role, content}, ...]
    API->>Retriever: Search relevant docs
    Retriever->>VectorDB: query(embedding)
    VectorDB-->>Retriever: top_k chunks
    Retriever-->>API: [Document, ...]
    API->>LLM: Generate response
    LLM-->>API: {content, tokens_used}
    API->>Memory: Save interaction
    API-->>User: {response, sources}
```

## Architecture Questions

```
- "What is the dependency graph of the main modules?"
- "What design patterns are used in this project?"
- "How do the microservices communicate?"
- "What's the authentication flow?"
- "How are errors propagated through the call stack?"
- "What's the data transformation pipeline from input to storage?"
```

## When to Use

- Onboarding to a new team
- Planning a major refactor
- Writing documentation
- Identifying architectural debt (circular dependencies, god classes)
