# AI Governance, Security & Compliance

Establish the policies, processes, and technical controls to deploy AI systems responsibly, securely, and in compliance with evolving regulations.

## Learning Objectives
- Design a model risk management framework aligned with regulatory expectations
- Implement data governance, lineage, and privacy controls for AI pipelines
- Build security guardrails against adversarial attacks and data leakage

## Deep Topics

### 1. Model Risk Management Frameworks
- **NIST AI Risk Management Framework**: Govern, Map, Measure, Manage
- **SR 11-7 / OCC 2011-12**: model risk management standards (validation, independent review)
- **EU AI Act**: risk categories (unacceptable, high, limited, minimal), conformity assessment
- **Model validation lifecycle**: conceptual soundness, outcomes analysis, ongoing monitoring
- **Model inventory**: centralized registry with risk ratings, version history, approval status

### 2. Data Lineage, Provenance & Cataloging
- **Data lineage**: tracking data origin, transformations, and consumption across pipelines
- **Tools**: Apache Atlas, DataHub, Amundsen, Marquez, OpenLineage
- **Data contracts**: schema, semantics, SLAs between data producers and consumers
- **Provenance for ML**: tracking training data sources, feature versions, augmentation steps

### 3. Privacy Techniques
- **Differential privacy**: adding calibrated noise to training (DP-SGD) or inference outputs
- **Data anonymization**: k-anonymity, l-diversity, t-closeness for datasets
- **Synthetic data**: generating privacy-preserving synthetic datasets (GANs, diffusion models)
- **Federated learning**: training across decentralized data without centralizing raw data

### 4. Adversarial Robustness & Testing
- **Adversarial attacks**: evasion (input perturbations), poisoning (training data manipulation), extraction (model stealing), inversion (reconstructing training data)
- **Robustness testing**: adversarial training, certified defenses, red-teaming
- **Tools**: CleverHans, Foolbox, ART (Adversarial Robustness Toolbox)
- **Continuous red-teaming**: LLM-specific attacks (jailbreaking, prompt leaking, role-playing)

### 5. Regulatory Landscape
- **EU AI Act**: classification, obligations for providers/users, penalties
- **GDPR**: right to explanation, data minimization, automated decision-making articles
- **HIPAA**: de-identification standards, business associate agreements for AI services
- **CCPA/CPRA**: consumer rights over data used in AI systems
- **Emerging regulations**: Canada AIDA, Brazil LGPD, India DPDP, US Executive Orders
- **Cross-border data transfer**: adequacy decisions, SCCs, DPF

### 6. Audit Trails, Explainability & Model Cards
- **Audit logging**: who deployed which model version, with what configuration, when
- **Explainability techniques**: SHAP, LIME, Integrated Gradients, attention visualization
- **Model cards**: standardized documentation (training data, performance, biases, intended use)
- **Datasheets for datasets**: motivation, composition, collection, preprocessing, uses
- **Bias detection**: fairness metrics (demographic parity, equal opportunity, equalized odds)

### 7. Access Control, IAM & Secret Management
- **Fine-grained access**: role-based (RBAC) and attribute-based (ABAC) for models and data
- **Model endpoint security**: API keys, OAuth2, mTLS for inference endpoints
- **Secrets management**: model API keys, database credentials, encryption keys (HashiCorp Vault, AWS Secrets Manager)
- **Just-in-time access**: ephemeral permissions for model training and deployment

### 8. Ethical AI Guidelines & Bias Mitigation
- **AI ethics principles**: fairness, accountability, transparency, privacy, robustness
- **Bias mitigation stages**: pre-processing (reweighting, sampling), in-processing (adversarial debiasing), post-processing (threshold adjustment)
- **Inclusive design**: representative datasets, diverse evaluation teams, accessibility
- **Ethics review board**: composition, charter, escalation process

## Deliverables
- Create a model card for an example LLM covering intended use, limitations, and bias evaluations
- Design a governance workflow for model promotion including risk review gates
- Perform a threat model for an LLM-powered application (STRIDE/LINDUNN)
