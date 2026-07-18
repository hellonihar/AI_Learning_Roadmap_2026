# Insurance Claim Anomaly — Deployment Notes

## Performance
- Ensemble score computation: ~0.3ms per claim (autoencoder: GPU, others: CPU)
- Daily batch scoring of 200K claims takes ~60s
- Model size (autoencoder weights): 0.5 MB

## Retraining Strategy
- **Monthly retrain** autoencoder on rolling 12 months of clean claims (confirmed non-fraud)
- **Quarterly** re-run injection experiment to validate detection rates on known fraud patterns
- Retrain if reconstruction error distribution shifts significantly (K-S test p < 0.01)

## Monitoring
- Track monthly: % flagged, auditor confirmation rate, average claim amount in flagged set
- Auditor feedback loop — confirmed/rejected flags used to tune ensemble weights (Bayesian update)
- Monitor for concept drift: new procedure codes, regulatory changes, new provider onboarding

## Integration
- Claims management system feeds batch jobs daily via CSV export
- Top 5% flags pushed to auditor work queue with per-feature reconstruction breakdown
- API endpoint for real-time pre-adjudication scoring on high-value claims (>$10K)
