# Customer Segmentation — Implementation

## Steps

1. **RFM Calculation** — Aggregate per customer: recency, frequency, monetary; apply log transform to normalize skewed distributions
2. **Feature Scaling** — StandardScaler on log-transformed RFM values
3. **Optimal K** — Elbow method (WCSS) + Silhouette analysis for K=2..10
4. **K-Means Clustering** — Fit with optimal K, analyze segment characteristics
5. **PCA 2D Visualization** — Project to 2 components, color by cluster
6. **Segment Interpretation** — Business labels (e.g., "Loyal High-Value", "At Risk", "New")

## Key Code

```python
# RFM feature engineering
from datetime import datetime

snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',   # Frequency
    'TotalPrice': 'sum'       # Monetary
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary'
})

rfm['LogMonetary'] = np.log1p(rfm['Monetary'])
rfm['LogFrequency'] = np.log1p(rfm['Frequency'])
rfm['LogRecency'] = np.log1p(rfm['Recency'])
```

```python
# Optimal K via silhouette score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sil_scores = []
wcss = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))
    wcss.append(kmeans.inertia_)

best_k = K_range[np.argmax(sil_scores)]
print(f'Optimal K: {best_k} (Silhouette: {max(sil_scores):.3f})')
```

```python
# PCA visualization
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

segments = pd.DataFrame({
    'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1],
    'Segment': kmeans.labels_,
    'Recency': rfm['Recency'].values,
    'Frequency': rfm['Frequency'].values,
    'Monetary': rfm['Monetary'].values
})

# Interpret segments by median RFM values
segment_summary = segments.groupby('Segment').agg({
    'Recency': 'median', 'Frequency': 'median', 'Monetary': 'median'
}).round(1)
print(segment_summary)
```
