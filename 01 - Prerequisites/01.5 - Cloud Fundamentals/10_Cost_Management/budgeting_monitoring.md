# Budgeting & Cost Monitoring

## Budgets & Alerts
| Provider | Service | Trigger |
|---|---|---|
| AWS | AWS Budgets | Actual or forecasted spend vs budget |
| GCP | Budget Alerts | % of budget threshold |
| Azure | Cost Management Alerts | Budget, anomaly, or credit |

Best practice: set alerts at 50%, 80%, 90%, and 100% of budget.

## Cost Monitoring Tools
| Tool | Use |
|---|---|
| AWS Cost Explorer | Visualize spend, filter by tag/service/region |
| GCP Cost Table | Detailed reports, breakdowns |
| Azure Cost Management | Multi-cloud cost analysis |
| Third-party | Vantage, CloudHealth, Kubecost (for Kubernetes) |

## Anomaly Detection
- AWS Cost Anomaly Detection: ML-based spend anomaly alerts
- GCP Recommender: cost optimization recommendations
- Azure Advisor: identifies idle resources and right-sizing opportunities

## For AI Engineers
- Set a budget per project (e.g., `$500/month for experiment-X`)
- Monitor GPU spend separately — it's likely your biggest cost
- Set alerts on spot instance termination (losing GPUs mid-training costs money)
- Review cost explorer weekly during active ML projects
