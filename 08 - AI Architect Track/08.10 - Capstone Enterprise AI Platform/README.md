# Capstone: Enterprise AI Platform

Design a comprehensive, multi-tenant AI platform that serves 50-200 data scientists and ML engineers across multiple product teams.

## Context

A mid-sized e-commerce company is scaling its AI capabilities. Currently each team uses its own tools and infrastructure, leading to duplication, inconsistent governance, and high costs. You are tasked with designing a unified AI platform.

- **Scale**: 50–200 data scientists and ML engineers across 5–10 product teams
- **Use cases**: real-time recommendations, NLP/LLM-powered search, computer vision quality inspection, demand forecasting
- **Constraints**: multi-cloud strategy, GDPR/HIPAA compliance, sub-100ms inference p99 latency, $500k–$2M annual budget
- **Maturity**: currently ad-hoc, no ML platform, each team using different tools

## Deliverables

### 1. Architecture Decision Record (ADR) Set
- At least **5 ADRs** covering:
  1. Cloud provider strategy (primary/fallback, single vs. multi-cloud)
  2. Model serving architecture (batch/streaming/online patterns)
  3. Feature store technology and topology
  4. MLOps platform component selection (build vs. buy)
  5. LLM strategy (model selection, hosting, RAG architecture)
- Each ADR: context, decision, consequences, rejected alternatives

### 2. High-Level System Diagrams
- **Component view**: data ingestion → feature engineering → training → model registry → serving → monitoring
- **Deployment view**: cloud regions, Kubernetes clusters, network topology
- **Data flow diagram**: online inference request flow step-by-step
- Each component annotated with technology choices

### 3. Component Specifications
Per major component:
- Functional requirements
- Non-functional requirements (latency, throughput, availability, durability)
- API contracts (key interfaces, data formats)
- Scaling characteristics (horizontal/vertical, autoscaling policies)
- Security and compliance controls

### 4. Security & Compliance Checklist
- Data privacy: PII detection, anonymization, retention policies
- Access control: model/data endpoint IAM, just-in-time access
- Model security: prompt injection prevention, output filtering, audit logging
- Regulatory mapping: GDPR / EU AI Act / HIPAA per component

### 5. Cost Model & FinOps Plan
- Compute cost breakdown per workload type (training, batch inference, online inference)
- Storage cost estimates (feature store, model registry, data lake)
- Network cost estimates (inter-region, cross-cloud, egress)
- Optimization opportunities (spot, reserved, model compression)
- Budget alerts and cost allocation strategy

### 6. Migration & Rollout Plan
- Phase 0: Assessment and quick wins (monitoring, model catalog)
- Phase 1: Platform foundation (feature store, model registry, ML pipelines)
- Phase 2: Migration of initial workloads (2–3 high-value use cases)
- Phase 3: Self-serve enablement and onboarding of remaining teams
- Phase 4: Optimization and advanced capabilities (LLMOps, multi-cloud DR)
- Risk assessment and mitigation for each phase

### 7. Team Topology & RACI
- Organizational model recommendation (centralized, federated, hybrid)
- Team structure: platform engineering, ML platform, applied ML, data engineering
- Role definitions and RACI matrix for cross-team collaboration
- Hiring roadmap and skill gap analysis

### 8. Stakeholder Presentation
- Executive summary (problem, solution, business value)
- Architecture overview (diagrams, key decisions)
- Cost and ROI summary
- Timeline and milestones
- Risk and mitigation
- Appendix: detailed ADRs, component specs

## Evaluation Criteria

| Criteria | Weight | Description |
|----------|--------|-------------|
| Architectural soundness | 25% | Decisions justified, tradeoffs acknowledged |
| Completeness | 20% | All components addressed, no blind spots |
| Practical feasibility | 20% | Realistic timeline, cost, and team constraints |
| Communication | 15% | Clear diagrams, compelling narrative |
| Security & compliance | 10% | Controls identified and mapped to regulations |
| Innovation | 10% | Creative solutions, emerging tech consideration |

## Format
- **Written**: PDF or Markdown (30–50 pages)
- **Live review**: 60-minute presentation + 30-minute Q&A
- **Tools**: Draw.io, Excalidraw, Miro, LucidChart
