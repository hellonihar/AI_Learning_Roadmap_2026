# Insurance Claim Anomaly — Dataset

## Source
Simulated claims data based on public CMS Part B outpatient claims (synthetic, with known fraud patterns injected at ~3%).

## Size & Shape
- **Total**: 150K claims, 28 features
- **Features**: Patient age, diagnosis codes (ICD-10 top 20 categories), procedure codes (CPT top 50), provider ID, claim amount, place of service, number of line items, modifier codes, date patterns (weekend/holiday), patient-provider distance
- **Target**: `is_fraud` (labelled — only used for evaluation)

## Challenges
- **Noisy labels** — Only ~40% of true fraud is caught in manual audits, so labelled data is itself incomplete
- **Count-based features** — Some patients have 1 claim, others 50+; per-patient baselines are sparse
- **Seasonal patterns** — Flu season drives claim volume up 3x; volume-based rules generate false positives
- **Provider ID sparsity** — 15K providers, most with < 10 claims
