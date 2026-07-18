# Market Basket Analysis — Implementation

## Steps

1. **Data Preparation** — Filter cancelled invoices, aggregate quantity, pivot to basket format (one list of items per transaction)
2. **Prune Rare Items** — Remove items appearing in < 1% of transactions (~30 items threshold)
3. **Apriori** — Mine frequent itemsets with min_support=0.01
4. **Rule Generation** — Generate association rules with min_confidence=0.3, min_lift=1.2
5. **Product Placement Recommendations** — Map rules to store layout strategy

## Key Code

```python
# Prepare basket format
from mlxtend.preprocessing import TransactionEncoder

# Group items per invoice
baskets = df.groupby('InvoiceNo')['Description'].apply(list).tolist()

# Remove single-item baskets
baskets = [b for b in baskets if len(b) >= 2]

te = TransactionEncoder()
te_ary = te.fit(baskets).transform(baskets)
basket_df = pd.DataFrame(te_ary, columns=te.columns_)

print(f'Baskets: {len(baskets)}, Items: {len(te.columns_)}')
```

```python
# Apriori frequent itemsets
from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(
    basket_df, min_support=0.01, use_colnames=True
)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)

print(f'Frequent itemsets: {len(frequent_itemsets)}')
print(frequent_itemsets.sort_values('support', ascending=False).head(10))
```

```python
# Association rules
from mlxtend.frequent_patterns import association_rules

rules = association_rules(
    frequent_itemsets, metric='lift', min_threshold=1.2
)
rules = rules.sort_values('lift', ascending=False)

# Top cross-sell opportunities
top_rules = rules[
    (rules['confidence'] >= 0.4) &
    (rules['lift'] >= 2.0) &
    (rules['antecedent_len'] == 1)  # Single-item triggers are most actionable
].head(20)

for _, r in top_rules.iterrows():
    a = list(r['antecedents'])[0]
    c = list(r['consequents'])[0]
    print(f'{a:40s} -> {c:40s}  (conf={r.confidence:.2f}, lift={r.lift:.2f})')
```
