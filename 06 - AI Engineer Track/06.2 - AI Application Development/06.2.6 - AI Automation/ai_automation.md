# AI Automation

## 1. Introduction to AI Automation

### Automation vs Intelligence

| | Traditional Automation (Rule-Based) | AI-Driven Automation (LLM-Based) |
|---|---|---|
| **Decision making** | Hardcoded if-else rules | Probabilistic model output |
| **Input handling** | Structured, predictable (CSV, API, forms) | Unstructured, variable (email, chat, documents) |
| **Adaptability** | Breaks when input format changes | Adapts to new phrasing, formats, contexts |
| **Error mode** | Obvious (crash, wrong branch) | Subtle (hallucination, plausible wrong answer) |
| **Setup time** | Hours–days (writing rules) | Minutes–hours (writing prompts) |
| **Maintenance** | Manual rule updates | Prompt tweaks, model updates, RAG refreshes |

**Traditional automation** excels at predictable, repetitive tasks with clear rules. **AI automation** excels at tasks requiring understanding, judgment, or handling variety. The strongest systems combine both — rules for what's known, AI for what's ambiguous.

### Rule-Based Automation vs LLM-Driven Automation

```
Rule-based email triage:
  IF subject contains "order" AND body contains "cancel"
  THEN route to "cancellation" team
  → Works for exact matches. Fails for "I want to cancel my order" written in the body with no subject.

LLM-driven email triage:
  Prompt: "Classify this customer email into one of: cancellation, refund, support, sales, other"
  → Handles any phrasing. Understands intent, not just keywords. But may misclassify.
```

| Aspect | Rule-Based | LLM-Driven |
|---|---|---|
| **Precision** | 100% for matched inputs | 85-98% (depends on prompt and model) |
| **Recall** | Low (misses variants) | High (catches all phrasings) |
| **Handling ambiguity** | Fails (no match → error or default) | Graceful (best guess, uncertainty expression) |
| **Explainability** | Perfect (trace exact rule) | Partial (can ask, may be inaccurate) |
| **Cost per execution** | Negligible | $0.001–$0.01 per LLM call |
| **Best for** | High-volume, stable, structured inputs | Variable, unstructured, judgment-based tasks |

**Best practice:** Use rules for what you know for certain. Use LLM for what requires understanding. Combine: rule-based pre-filter → LLM for ambiguous cases → rule-based post-validation.

### Deterministic vs Probabilistic Workflows

```
Deterministic workflow (all rules):
  Input → Validate → Transform → Save → Output
  → Every step is predictable. Same input = same output.
  → Easy to test, debug, and audit.

Probabilistic workflow (LLM involved):
  Input → LLM Classify → Route → LLM Generate → Validate → Output
  → Output varies. Same input may produce different results.
  → Hard to test, but flexible and adaptive.
```

| Characteristic | Deterministic | Probabilistic |
|---|---|---|
| **Repeatability** | Same input → same output | Same input → similar but not identical |
| **Testability** | Unit tests catch all failures | Statistical evaluation needed |
| **Error handling** | Known failure modes | Unexpected failures possible |
| **Audit trail** | Complete trace | Black box (model reasoning is inferred) |
| **LLM role** | None | Classification, generation, routing, validation |

**Hybrid pattern:** Use deterministic steps for reliability-critical operations (saving to database, sending emails, financial calculations) and probabilistic LLM steps for judgment operations (classification, extraction, content generation). Validate LLM outputs with deterministic rules before acting on them.

### Human-in-the-Loop Automation

AI automation is rarely fully autonomous. Humans handle edge cases, high-stakes decisions, and training data generation.

```
Confidence-based handoff:
  LLM confidence > 0.95 → Auto-execute (no human needed)
  LLM confidence 0.70-0.95 → Flag for human review (suggest action)
  LLM confidence < 0.70 → Escalate to human (no action taken)

  → 80% of cases auto-executed
  → 15% reviewed quickly (human confirms or corrects)
  → 5% escalated for manual handling
```

| HITL Pattern | How It Works | Best For |
|---|---|---|
| **Review before action** | Human approves LLM output before execution | High-stakes (financial transactions, legal decisions) |
| **Review after action** | LLM executes, human audits later | Low-stakes (content suggestions, email drafts) |
| **Exception handling** | LLM routes only edge cases to humans | Customer support tier 1 vs tier 2 |
| **Training loop** | Human corrections → retrain/improve LLM | Continuous improvement |
| **Fallback** | Human handles when LLM confidence is low | Safety-critical systems |

---

## 2. Automation Design Patterns

### A. Trigger-Based Automation

#### Event-Driven Workflows

An event triggers an automated sequence. No polling, no scheduled checks — the system reacts when something happens.

```
Event sources:
  ┌──────────────────────┐
  │ New email arrives     │
  │ New file in S3 bucket │
  │ Webhook received      │
  │ Database row inserted │
  │ Slack message posted  │
  │ Form submitted        │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ Trigger → LLM Workflow│
  │  (classify, extract,  │
  │   generate, act)      │
  └──────────────────────┘
```

```python
# Event-driven automation: new support ticket
def on_new_ticket(ticket_data: dict):
    """Triggered by webhook when a new support ticket is created."""

    # Step 1: Classify urgency
    urgency = llm_classify(
        f"Classify urgency (high/medium/low): {ticket_data['description']}"
    )

    # Step 2: Route based on urgency
    if urgency == "high":
        send_slack_alert(f"URGENT: {ticket_data['title']}")
        assign_to_priority_queue(ticket_data)
    else:
        assign_to_standard_queue(ticket_data)

    # Step 3: Generate initial response
    response = llm_generate(
        f"Draft a response acknowledging this ticket: {ticket_data['description']}"
    )

    # Step 4: Post response (with human review if high urgency)
    if urgency == "high":
        save_for_review(ticket_data["id"], response)
    else:
        post_response(ticket_data["id"], response)
```

#### Webhooks

Webhooks are HTTP callbacks — when an external system has an event, it sends an HTTP request to your endpoint.

```python
from flask import Flask, request

app = Flask(__name__)

@app.route("/webhook/slack", methods=["POST"])
def handle_slack_webhook():
    data = request.json

    # Parse Slack message
    text = data["event"]["text"]
    channel = data["event"]["channel"]
    user = data["event"]["user"]

    # LLM determines intent
    intent = llm_classify(f"Classify this Slack message intent: {text}")

    if intent == "question":
        answer = llm_generate(f"Answer this question concisely: {text}")
        post_to_slack(channel, answer)
    elif intent == "request":
        create_ticket(text, user)
        post_to_slack(channel, "Ticket created!")
    elif intent == "status_check":
        status = query_status(text)
        post_to_slack(channel, status)

    return {"ok": True}, 200
```

| Aspect | Webhooks | Polling |
|---|---|---|
| **Trigger** | Real-time (push) | Scheduled (pull) |
| **Latency** | Milliseconds | Seconds–minutes (interval) |
| **Resource usage** | Low (only when events occur) | Constant (checking even when nothing changes) |
| **Reliability** | May miss events if endpoint is down | Eventually catches all events |
| **Implementation** | Provide URL, handle POST | Run cron job, check for changes |

**Best practice:** Use webhooks for real-time automation. Use polling as a fallback when webhooks aren't supported or for resilience (catch missed webhooks).

#### Scheduled Jobs (Cron-Based)

For tasks that run on a fixed schedule rather than reacting to events.

```python
# cron configuration (run every day at 9 AM)
# 0 9 * * * python daily_report.py

def daily_report():
    """Generate and send daily automated report."""

    # 1. Gather data
    sales_data = query_database("SELECT * FROM sales WHERE date = CURRENT_DATE")
    support_tickets = query_database("SELECT * FROM tickets WHERE created_at > NOW() - INTERVAL '1 day'")

    # 2. LLM analyzes and summarizes
    summary = llm_generate(f"""
    Summarize today's performance:
    Sales: {sales_data}
    Support tickets: {support_tickets}
    Highlight key trends, anomalies, and action items.
    """)

    # 3. Send report
    send_email(
        to="management@company.com",
        subject=f"Daily Report - {date.today()}",
        body=summary
    )
```

| Schedule Pattern | Example | Use Case |
|---|---|---|
| **Fixed interval** | Every hour | Data syncing, cache warming |
| **Daily** | Every day at 9 AM | Report generation, data backup |
| **Weekly** | Every Monday | Performance review, cleanup |
| **Conditional** | Run when condition is met | "Check every 5 minutes, but only act if new data exists" |

### B. Sequential Task Automation

#### Multi-Step Pipelines

A sequence of steps where each step depends on the previous one.

```
Step 1: Ingest raw data (PDF, email, form)
Step 2: LLM extracts structured data
Step 3: Validate extracted data against schema
Step 4: Transform to target format
Step 5: Write to database / send to API
Step 6: Notify user of completion
```

```python
def invoice_processing_pipeline(raw_invoice: str) -> dict:
    """Process an invoice from raw text to database record."""

    # Step 1: Extract fields using LLM
    extracted = llm_extract(f"""
    Extract the following fields from this invoice:
    - invoice_number
    - vendor_name
    - date (YYYY-MM-DD)
    - total_amount (number)
    - line_items (array of {description, amount})

    Invoice: {raw_invoice}
    """)

    # Step 2: Validate extracted data
    validation_errors = validate_invoice_schema(extracted)
    if validation_errors:
        return {"status": "failed", "errors": validation_errors, "data": extracted}

    # Step 3: Transform
    transformed = {
        "invoice_id": extracted["invoice_number"],
        "vendor": normalize_vendor_name(extracted["vendor_name"]),
        "amount_cents": int(extracted["total_amount"] * 100),
        "line_items": extracted["line_items"],
        "received_at": datetime.now().isoformat()
    }

    # Step 4: Store
    db.insert("invoices", transformed)

    # Step 5: Route for approval
    if transformed["amount_cents"] > 100000:  # $1,000+
        route_for_approval(transformed)
        return {"status": "pending_approval"}
    else:
        auto_approve(transformed)
        return {"status": "approved"}
```

#### Data → Process → Output Flows

A three-stage architecture that separates concerns.

```
[Data Layer]          [Process Layer]        [Output Layer]
Raw emails     →      LLM classify +     →  Route to queue
Webhook data           extract + generate    Send email/Slack
Database rows                                 Update CRM
File uploads                                  Create ticket
                                              Store in DB
```

**Data layer responsibilities:**
- Ingest from various sources (API, webhook, file, DB)
- Normalize to a common format
- Validate basic structure (not null, expected types)

**Process layer responsibilities:**
- LLM classification, extraction, generation
- Rule-based validation and transformation
- Business logic execution

**Output layer responsibilities:**
- Deliver to destination (API, email, DB, queue)
- Handle failures (retry, DLQ)
- Log and notify

#### Template-Based Automation

Use templates for common automation patterns, parameterized by input.

```python
AUTOMATION_TEMPLATES = {
    "email_response": {
        "steps": [
            {"action": "llm_classify", "prompt": "Classify email intent: {input}"},
            {"action": "llm_generate", "prompt": "Draft response with intent: {classification}"},
            {"action": "send_email", "to": "{sender}", "body": "{draft}"}
        ]
    },
    "ticket_triage": {
        "steps": [
            {"action": "llm_extract", "prompt": "Extract: urgency, category, description"},
            {"action": "create_ticket", "data": "{extracted}"},
            {"action": "notify_team", "channel": "{urgency}_queue"}
        ]
    },
    "report_generation": {
        "steps": [
            {"action": "query_db", "query": "{report_sql}"},
            {"action": "llm_summarize", "prompt": "Summarize data: {results}"},
            {"action": "send_email", "to": "{recipients}", "body": "{summary}"}
        ]
    }
}

def execute_template(template_name: str, params: dict):
    template = AUTOMATION_TEMPLATES[template_name]
    context = params
    for step in template["steps"]:
        result = execute_step(step, context)
        context.update(result)
    return context
```

### C. Conditional Automation

#### If-Else Logic

Branch the workflow based on LLM output or programmatic conditions.

```python
def route_request(request_data: dict):
    """Route incoming request based on LLM classification."""

    # LLM determines intent
    intent = llm_classify(f"Intent: {request_data['text']}")

    if intent == "refund":
        return refund_workflow(request_data)
    elif intent == "cancel_subscription":
        return cancel_workflow(request_data)
    elif intent == "technical_support":
        return support_workflow(request_data)
    elif intent == "sales_inquiry":
        return sales_workflow(request_data)
    else:
        return fallback_workflow(request_data)
```

#### Classifier-Based Routing

Use a dedicated classifier for deterministic routing, then pass to specialized LLM workflows.

```
All incoming requests
      │
      ▼
  Lightweight classifier (fine-tuned BERT or LLM with temperature 0)
      │
      ├── refund → LLM refund workflow (specialized prompt)
      ├── cancel → LLM cancel workflow (specialized prompt)
      ├── support → LLM support workflow + RAG from knowledge base
      ├── sales → LLM sales workflow + CRM lookup
      └── unknown → Human-in-the-loop
```

**Why classifier routing matters:**
- Each workflow can have a specialized prompt (higher quality)
- Classification is cheap (small model, temperature 0)
- If classification is wrong, only the routing is wrong — the workflow itself is reliable

#### Intent-Based Execution

The LLM determines both the intent and the execution parameters, not just the route.

```python
def intent_based_execution(user_input: str):
    """LLM decides what to do AND how to do it."""

    # Single LLM call: determines intent + extracts parameters
    result = llm_call(f"""
    Analyze this request and determine:
    1. Intent (one of: query_database, send_email, create_report, unknown)
    2. Parameters needed for execution

    Request: {user_input}
    Output JSON format:
    {{"intent": "...", "parameters": {{...}} }}
    """)

    intent = result["intent"]
    params = result["parameters"]

    if intent == "query_database":
        return query_database(params["query"])
    elif intent == "send_email":
        return send_email(params["to"], params["subject"], params["body"])
    elif intent == "create_report":
        return generate_report(params["report_type"], params["date_range"])
    else:
        return "I couldn't determine how to handle this request."
```

### D. Loop & Retry Automation

#### Iterative Improvement Loops

The LLM generates output, evaluates it, and improves it in a loop.

```python
def iterative_email_draft(topic: str, max_iterations: int = 3) -> str:
    """Draft an email with iterative improvement."""

    # Initial draft
    draft = llm_generate(f"Draft a professional email about: {topic}")

    for i in range(max_iterations - 1):
        # Self-critique
        feedback = llm_call(f"""
        Evaluate this email for:
        - Professionalism
        - Clarity
        - Conciseness
        - Correct tone

        Email: {draft}
        Provide specific improvement suggestions.
        """)

        if "no improvements needed" in feedback.lower():
            break

        # Improve
        draft = llm_generate(f"""
        Original email: {draft}
        Feedback: {feedback}
        Write an improved version addressing all feedback.
        """)

    return draft
```

#### Error Handling Retries

When a step fails, retry with progressively more context.

```python
def robust_llm_step(prompt: str, max_retries: int = 3, validators: list = None) -> str:
    """Execute an LLM step with retry and validation."""
    last_error = None
    for attempt in range(max_retries):
        try:
            # Attempt LLM call
            result = llm_call(prompt)

            # Run validators (e.g., JSON parse, schema check)
            if validators:
                for validator in validators:
                    validator(result)

            return result

        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}"
            prompt += f"\n\nPrevious output was not valid JSON. Fix it.\nError: {last_error}"

        except SchemaValidationError as e:
            last_error = f"Schema violation: {e}"
            prompt += f"\n\nSchema validation failed. Fix the output.\nError: {last_error}"

        except RateLimitError:
            wait(60)  # Wait and retry
            continue

        except Exception as e:
            last_error = str(e)
            prompt += f"\n\nAn error occurred: {last_error}. Try again."

    raise MaxRetriesExceeded(f"Step failed after {max_retries} attempts. Last error: {last_error}")
```

| Error Type | Retry Strategy | Max Retries |
|---|---|---|
| **LLM API timeout** | Exponential backoff (1s, 2s, 4s) | 3 |
| **Invalid output format** | Re-prompt with error message | 3 |
| **Schema validation failure** | Re-prompt with schema + error | 2 |
| **Rate limit** | Wait 60 seconds | 3 |
| **Empty response** | Re-prompt with "Output was empty" | 2 |
| **No error but low quality** | Re-prompt with critique | 1-2 |

#### Validation Loops

Loop until output passes validation criteria.

```python
def extract_with_validation(raw_text: str, schema: dict, max_iterations: int = 3) -> dict:
    """Extract structured data with iterative validation."""

    prompt = f"""Extract the following fields from the text and return valid JSON.
    Schema: {schema}
    Text: {raw_text}
    Return ONLY valid JSON."""

    for iteration in range(max_iterations):
        output = llm_call(prompt)

        try:
            parsed = json.loads(output)
            errors = validate_schema(parsed, schema)
            if not errors:
                return parsed  # Success
            prompt += f"\n\nValidation errors: {errors}. Fix them."
        except json.JSONDecodeError as e:
            prompt += f"\n\nInvalid JSON: {e}. Return ONLY valid JSON."

    raise ValidationFailed(f"Failed to extract valid data after {max_iterations} iterations")
```

---

## 3. Tool Integration for Automation

### API Integration

The most common integration pattern — LLM calls external APIs based on its decisions.

```python
import requests

API_TOOLS = {
    "get_weather": {
        "execute": lambda loc: requests.get(f"https://api.weather.com/v1/{loc}").json(),
        "description": "Get current weather for a location"
    },
    "create_slack_message": {
        "execute": lambda channel, text: requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
            json={"channel": channel, "text": text}
        ).json(),
        "description": "Post a message to a Slack channel"
    },
    "search_web": {
        "execute": lambda query: requests.get(
            "https://api.search.com/search",
            params={"q": query}
        ).json(),
        "description": "Search the web for information"
    }
}

def execute_api_tool(tool_name: str, params: dict) -> dict:
    """Execute an API tool by name with parameters."""
    if tool_name not in API_TOOLS:
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        result = API_TOOLS[tool_name]["execute"](**params)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### Function Calling

Use LLM function calling to reliably invoke tools with structured arguments.

```python
tools = [
    {
        "name": "search_knowledge_base",
        "description": "Search internal knowledge base for documentation",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    },
    {
        "name": "create_jira_ticket",
        "description": "Create a new Jira ticket",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]}
            },
            "required": ["title", "description", "priority"]
        }
    }
]

def handle_tool_call(tool_call):
    """Execute a tool call from the LLM."""
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if name == "search_knowledge_base":
        return search_kb(args["query"], args.get("max_results", 5))
    elif name == "create_jira_ticket":
        return create_ticket(args["title"], args["description"], args["priority"])
    else:
        return {"error": f"Unknown function: {name}"}
```

### Database Operations

LLM-driven database queries for reporting, analysis, and record management.

```python
def natural_language_query(user_question: str) -> str:
    """Convert a natural language question to a SQL query, execute, and summarize."""

    # 1. Generate SQL
    sql = llm_call(f"""
    Given this schema:
    Tables: orders(id, customer_id, amount, date, status), customers(id, name, email, tier)
    Convert this question to SQL:
    "{user_question}"

    Return ONLY the SQL query, no explanation.
    """)

    # 2. Validate SQL (basic safety check)
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    for kw in forbidden_keywords:
        if kw in sql.upper():
            return f"I cannot execute this query as it contains {kw}."

    # 3. Execute
    try:
        results = db.execute(sql)
    except Exception as e:
        return f"Query failed: {e}"

    # 4. Summarize results
    summary = llm_generate(f"""
    Question: {user_question}
    Results: {results}
    Summarize the answer in plain language.
    """)

    return summary
```

### File Handling

Process files (PDFs, images, CSVs) as part of automation workflows.

```python
def process_uploaded_file(file_path: str, file_type: str) -> dict:
    """Process an uploaded file and extract structured data."""

    if file_type == "pdf":
        text = extract_text_from_pdf(file_path)
    elif file_type == "image":
        text = extract_text_from_image(file_path)  # OCR
    elif file_type == "csv":
        text = convert_csv_to_text(file_path)
    else:
        return {"error": f"Unsupported file type: {file_type}"}

    # LLM extracts relevant data from the text
    extracted = llm_extract(f"""
    Extract key information from this document:
    {text[:10000]}  # Truncate for token budget
    """)

    # Store extracted data
    record_id = db.insert("processed_documents", {
        "original_file": file_path,
        "extracted_data": extracted,
        "processed_at": datetime.now().isoformat()
    })

    return {"record_id": record_id, "extracted": extracted}
```

### Email & Messaging APIs

Send and process emails, Slack messages, SMS as part of workflows.

```python
def process_incoming_email(email_data: dict) -> dict:
    """Process incoming email and trigger appropriate workflow."""

    # Extract email content
    subject = email_data["subject"]
    body = email_data["body"]
    sender = email_data["from"]

    # LLM classifies and extracts
    result = llm_call(f"""
    Analyze this email:
    Subject: {subject}
    Body: {body}
    Sender: {sender}

    Determine:
    1. Intent (refund_request, support_issue, sales_inquiry, spam)
    2. Urgency (high, medium, low)
    3. Key entities (order numbers, amounts, dates)
    """)

    intent = result["intent"]
    urgency = result["urgency"]

    # Route based on intent
    if intent == "refund_request":
        ticket_id = create_refund_ticket(result["entities"])
        auto_reply = llm_generate(f"Draft a refund acknowledgment email to {sender}")
        send_email(to=sender, subject="Refund Request Received", body=auto_reply)

    elif intent == "support_issue":
        ticket_id = create_support_ticket(result["entities"], urgency)
        if urgency == "high":
            send_slack_alert(f"Urgent ticket from {sender}: {subject}")

    elif intent == "spam":
        mark_as_spam(sender)

    return {"ticket_id": ticket_id, "intent": intent, "urgency": urgency}
```

### Web Scraping

Automated data collection from websites.

```python
def scrape_and_summarize(url: str) -> str:
    """Scrape a URL and summarize its content using LLM."""
    import requests
    from bs4 import BeautifulSoup

    # Fetch page
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract main content (remove navigation, footers, ads)
    for tag in soup(["nav", "footer", "script", "style", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)

    # Truncate to fit LLM context
    text = text[:15000]

    # LLM summarizes
    summary = llm_generate(f"""
    Summarize the key information from this page:
    URL: {url}
    Content: {text}
    Provide a concise summary of the main points.
    """)

    return summary
```

### Third-Party SaaS Integration

Connect to common SaaS platforms via their APIs.

| Platform | Common Automation Tasks | API Type |
|---|---|---|
| **Slack** | Send messages, read channels, manage users | REST + WebSocket |
| **Gmail / Outlook** | Send/read emails, manage labels/folders | REST (Graph API, Gmail API) |
| **Jira / Linear** | Create/update tickets, query projects | REST |
| **Salesforce / HubSpot** | Create/update contacts, log activities | REST |
| **Google Sheets / Airtable** | Read/write rows, create records | REST |
| **Notion / Confluence** | Create/update pages, query databases | REST |
| **Stripe** | Process refunds, query transactions | REST |
| **Zendesk / Intercom** | Manage tickets, send messages | REST |

```python
# Example: HubSpot integration
def sync_lead_to_crm(lead_data: dict):
    """Sync a new lead from an automation workflow to HubSpot."""
    import hubspot

    client = hubspot.Client.create(api_key=HUBSPOT_API_KEY)

    # LLM enriches lead data
    enriched = llm_extract(f"""
    Extract structured lead information:
    {lead_data}

    Normalize:
    - First/last name from full name
    - Company from email domain if not provided
    - Standardize phone number format
    """)

    contact = client.crm.contacts.basic_api.create(
        body={
            "properties": {
                "email": enriched["email"],
                "firstname": enriched["first_name"],
                "lastname": enriched["last_name"],
                "company": enriched.get("company", ""),
                "phone": enriched.get("phone", ""),
                "lead_source": "website_form",
                "notes": enriched.get("notes", "")
            }
        }
    )

    return {"contact_id": contact.id, "status": "created"}
```

---

## 4. Workflow Orchestration Tools

### n8n

n8n is an open-source workflow automation tool with a visual editor and code support.

| Feature | Details |
|---|---|
| **License** | Sustainable Use License (free self-hosted, paid cloud) |
| **Hosting** | Self-hosted (Docker, npm) or cloud (n8n.cloud) |
| **LLM Integration** | Built-in OpenAI, Claude, Hugging Face, Ollama nodes |
| **Triggers** | Webhook, cron, email, RSS, 400+ integrations |
| **Strengths** | Open source, self-hosted, extensive integrations |
| **Weaknesses** | UI can lag with complex workflows, limited advanced LLM patterns |

**Example n8n workflow (visual):**

```
[Webhook Trigger] → [OpenAI Node: Classify Intent]
                         ↓
              [Switch Node: Intent]
                  │        │
           [refund]    [support]
                  │        │
         [OpenAI Node:  [OpenAI Node:
          Extract Data]  Sentiment Analysis]
                  │        │
         [HubSpot Node:  [Slack Node:
          Create Ticket] Post Message]
```

**LLM-specific n8n patterns:**
- **OpenAI Node:** Single LLM call with system/user messages
- **Switch Node:** Route based on LLM output
- **Code Node:** Custom JavaScript for complex logic
- **Webhook Node:** Trigger from any external system
- **Wait Node:** Add delays, implement polling patterns

**When to use n8n:**
- Need self-hosted orchestration (data privacy)
- Want visual workflow editor + code when needed
- Have complex multi-step workflows with many integrations
- Team is comfortable with Node.js ecosystem

### Zapier

Zapier is a cloud-only no-code automation platform.

| Feature | Details |
|---|---|
| **License** | Proprietary (SaaS, free tier limited) |
| **Hosting** | Cloud only |
| **LLM Integration** | OpenAI (ChatGPT) via Zapier's AI actions |
| **Triggers** | 5,000+ integrations, polling-based |
| **Strengths** | Easiest setup, massive integration library |
| **Weaknesses** | Expensive at scale, no local data processing, polling latency |

**Example Zapier workflow:**

```
Trigger: New Gmail email matching label "support"
  → Action: ChatGPT step — "Extract order number and classify urgency"
  → Action: Create row in Google Sheets
  → Action: Send Slack notification to support channel
```

**LLM-specific Zapier patterns:**
- **ChatGPT action:** Place LLM calls anywhere in the workflow
- **Formatter:** Structured data manipulation
- **Filter:** Conditional branching
- **Paths:** Multi-branch workflows
- **Webhooks:** Connect to custom apps

**When to use Zapier:**
- Need quick setup with zero code
- Connecting common SaaS tools (Gmail, Slack, Sheets, CRM)
- Low-volume automation (hundreds, not thousands of runs/day)
- Non-technical team members building automations

### Make (formerly Integromat)

Make is a visual automation platform with a focus on complex data transformations.

| Feature | Details |
|---|---|
| **License** | Proprietary (SaaS, free tier up to 1K ops/month) |
| **Hosting** | Cloud only (no self-hosted option) |
| **LLM Integration** | OpenAI, Anthropic, AI modules |
| **Triggers** | Webhook, polling, 1,500+ integrations |
| **Strengths** | Powerful data transformation, visual mapping, complex scenarios |
| **Weaknesses** | Cloud-only, pricing jumps at scale, learning curve |

**When to use Make:**
- Need complex data transformations (maps, filters, aggregators)
- Visual data mapping is valuable
- Moderate-volume automation
- Data processing pipelines (not just simple triggers → actions)

### Tools Comparison

| Criterion | n8n | Zapier | Make |
|---|---|---|---|
| **License** | Open source (free self-hosted) | Proprietary (SaaS only) | Proprietary (SaaS only) |
| **Hosting** | Self-hosted or cloud | Cloud only | Cloud only |
| **Data privacy** | Full control (self-host) | Data processed on Zapier servers | Data processed on Make servers |
| **LLM support** | Built-in (OpenAI, Claude, Ollama) | ChatGPT action, AI by Zapier | OpenAI, Anthropic modules |
| **Integrations** | 400+ | 5,000+ | 1,500+ |
| **Visual editor** | Yes | Yes (simple) | Yes (complex) |
| **Code support** | JavaScript, Python (code node) | Limited (Code step) | Limited |
| **Free tier** | Yes (self-hosted) | 100 tasks/month | 1,000 ops/month |
| **Pricing (scale)** | Free (self-host) or $20/mo (cloud) | $30–$600/mo | $9–$170/mo |
| **Best for** | Self-hosted, complex, custom | Quick SaaS integrations | Data transformation pipelines |

### Selection Guide

```
Need data privacy / self-hosting?
  ├── Yes → n8n (self-hosted)
  └── No → ↓

Technical team / need custom code?
  ├── Yes → n8n (code nodes, JavaScript/Python)
  └── No → ↓

Complex data transformations?
  ├── Yes → Make
  └── No → ↓

Quick SaaS integration, non-technical?
  └── Yes → Zapier
```

### Custom Orchestration vs Tools

| Approach | When | Pros | Cons |
|---|---|---|---|
| **n8n / Make / Zapier** | Standard connectors, visual workflow, small-medium scale | Fast setup, no infra, visual debugging | Limited LLM patterns, cost at scale, less flexible |
| **Custom code (Python + FastAPI)** | Complex LLM logic, custom integrations, high scale | Full control, optimized cost, advanced patterns | Higher initial effort, need to build everything |
| **LangChain / LlamaIndex** | LLM-heavy workflows, agent patterns | Built for LLM orchestration, advanced patterns | Framework lock-in, abstraction leaks |
| **Temporal / Airflow** | Long-running, fault-tolerant workflows at huge scale | Industrial-grade reliability, replay, retries | Heavy infrastructure, overkill for simple cases |

**Best practice:** Start with n8n for prototyping. As patterns emerge and scale grows, port to custom code or Temporal. Don't start with Temporal — it's too heavy for the first iteration.

### Complete Automation Example: Customer Support Ticket Pipeline

```python
# Full automation pipeline combining all patterns

def customer_support_automation(webhook_data: dict) -> dict:
    """
    End-to-end automated customer support ticket processing.
    Triggered by webhook from support platform.
    """

    # ── Phase 1: Extract & Classify ──
    ticket = llm_extract(f"""
    Extract from this support request:
    - customer_name
    - email
    - order_id (if any)
    - issue_description
    - urgency (high/medium/low)

    Request: {webhook_data['message']}
    """)

    # ── Phase 2: Conditional Routing ──
    if ticket["urgency"] == "high":
        send_slack_alert(f"URGENT: {ticket['customer_name']} - {ticket['issue_description'][:100]}")
    else:
        log_to_queue("standard_queue", ticket)

    # ── Phase 3: Knowledge Retrieval ──
    relevant_docs = search_knowledge_base(ticket["issue_description"])

    if not relevant_docs:
        # No docs found — route to human
        return create_human_ticket(ticket, priority="high")

    # ── Phase 4: Generate Response ──
    response = llm_generate(f"""
    Using these knowledge base articles:
    {relevant_docs}

    Draft a helpful response to this customer issue:
    {ticket['issue_description']}

    Guidelines:
    - Be empathetic and professional
    - Provide specific steps
    - Include relevant links or documentation references
    - If uncertain, offer to escalate
    """)

    # ── Phase 5: Validation ──
    validation = llm_call(f"""
    Check this response for:
    - Accuracy (does it match the knowledge base?)
    - Professionalism (appropriate tone?)
    - Completeness (does it address the issue?)
    - Safety (no harmful content?)

    Response: {response}
    """)

    if validation.get("issues"):
        # Low confidence — human review
        draft_for_review(ticket["email"], response, validation["issues"])
        return {"status": "pending_review"}
    else:
        # Send automatically
        send_email(ticket["email"], "Support Response", response)
        log_resolution(ticket)
        return {"status": "resolved"}
```

### Automation Anti-Patterns

| Anti-Pattern | Why It Fails | Fix |
|---|---|---|
| **No validation on LLM output** | Hallucinations cause wrong actions | Always validate before executing |
| **No human fallback** | Automation fails on edge cases | Always have a human escalation path |
| **Too many LLM calls** | Latency and cost explode | Use rules for simple steps, LLM only for judgment |
| **Polling when webhooks exist** | Wasted resources, delayed responses | Prefer event-driven triggers |
| **Ignoring rate limits** | Workflow fails silently under load | Implement exponential backoff |
| **No idempotency** | Retrying causes duplicate actions | Design for exactly-once execution |
| **Single point of failure** | One service down = everything fails | Use DLQ, fallbacks, circuit breakers |
