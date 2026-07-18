# Network Intrusion Detection — Results

## Metrics

| Attack Category | Isolation Forest ROC AUC | One-Class SVM ROC AUC | Ensemble ROC AUC |
|----------------|--------------------------|-----------------------|------------------|
| DOS | 0.89 | 0.87 | 0.92 |
| Probe | 0.84 | 0.81 | 0.88 |
| R2L | 0.61 | 0.58 | 0.65 |
| U2R | 0.52 | 0.49 | 0.55 |
| Novel attacks | 0.63 | 0.59 | 0.66 |

## What Was Learned

- Ensemble averaging improved ROC AUC by 3-4% points over either model alone
- DOS attacks are easy (high volume, bursty patterns) — 93% recall at 1% FPR
- R2L/U2R are near-random — these attacks mimic normal user behavior and need content-based features (shell commands, file access patterns) not available in the 41-feature set
- Novel attack detection at 0.66 ROC AUC is better than random but not production-worthy

## Failure Cases

- **Slow R2L** — Attacks that establish a normal connection then slowly escalate (e.g., dictionary attacks over hours)
- **Encrypted tunnels** — If the payload is encrypted, content features provide no signal
- **False positives on flash crowds** — Legitimate traffic spikes (e.g., product launch) look like DOS
