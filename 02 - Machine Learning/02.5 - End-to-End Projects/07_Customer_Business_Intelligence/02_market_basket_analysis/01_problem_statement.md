# Market Basket Analysis — Problem Statement

## Business Context
A grocery chain with 200 stores wants to optimize shelf placement, cross-sell promotions, and bundle pricing. Currently product placement is category-based (all bread together). Need association rules to identify products frequently purchased together to drive impulse buys (+15% basket value target).

## Problem Type
- **Primary**: Association rule mining (Apriori algorithm)
- **Secondary**: Frequent itemset generation

## Success Metrics
- **Lift** (strength of association beyond random)
- Number of actionable rules (confidence > 0.5, lift > 1.5)
- Business adoption: % of recommendations implemented in store layout
- Basket size increase in test stores (A/B)
