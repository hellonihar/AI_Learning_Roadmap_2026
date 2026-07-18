# Decision Tree (Classification)

A tree where each **internal node** tests a feature, each **branch** represents the outcome, and each **leaf** predicts a class.

## How It Works

1. Start with all data at the root
2. Find the feature and threshold that best splits the data (most "pure" child nodes)
3. Create child nodes for each split
4. Repeat recursively until stopping condition is met

## Split Criteria

| Criterion | Intuition |
|---|---|
| **Gini impurity** | "How often would a randomly chosen element be mislabeled?" Lower = better |
| **Entropy / Information gain** | "How much uncertainty is reduced?" Higher gain = better |

Both usually produce similar trees. Gini is slightly faster; entropy is slightly more balanced.

## Overfitting & Pruning

Decision trees are prone to overfitting — they can grow until every training sample is in its own leaf (100% training accuracy).

**Pre-pruning** (stop early): `max_depth`, `min_samples_split`, `min_samples_leaf`
**Post-pruning**: Grow full tree, then remove branches that don't improve validation performance.

## Examples

1. **Loan approval**: First split: `income > $50K?`. If yes, next: `credit_score > 700?`. If no: `debt_ratio < 0.4?`. Each path is a human-readable decision rule — regulators love this.
2. **Medical triage**: `heart_rate > 100?` → `blood_pressure < 90?` → `age > 65?`. The tree mimics a doctor's decision flow. Emergency staff can follow the tree without understanding ML.
3. **Customer support routing**: `is_billing_issue?` → `is_urgent?` → route to billing team. `is_technical?` → `is_app_crash?` → route to engineering. Simple, explainable, fast.
