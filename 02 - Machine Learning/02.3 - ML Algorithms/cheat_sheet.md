# ML Algorithm Cheat Sheet — When to Use What

## By Problem Type

### Regression

| Data Size | Linear OK? | Recommendation |
|---|---|---|
| Small (<1K) | Yes | Ridge / Lasso with cross-validation |
| Medium (1K-100K) | Mixed | Random Forest or XGBoost |
| Large (100K+) | NO (unless sparse) | LightGBM or Linear with SGD |

**Best default**: Random Forest (no tuning needed, handles non-linearity, feature importance built-in)

### Classification (Binary)

| Data Size | Linear Separable? | Recommendation |
|---|---|---|
| Small (<1K) | Yes | Logistic Regression (interpretable) |
| Small (<1K) | No | SVC with RBF kernel or Random Forest |
| Medium (1K-100K) | Any | XGBoost / LightGBM |
| Large (100K+) | Any | LightGBM or Logistic Regression with SGD |
| Text data | N/A | MultinomialNB or Linear SVC |

**Best default**: Logistic Regression (fast, interpretable, baseline) then XGBoost

### Classification (Multi-Class)

| Scenario | Recommendation |
|---|---|
| Few classes (3-10) | Random Forest or XGBoost |
| Many classes (10+) | Linear SVC or MultinomialNB |
| Text classification | MultinomialNB |

### Clustering

| Cluster Shape | Cluster Count Known? | Recommendation |
|---|---|---|
| Spherical | Yes | K-Means |
| Spherical | No | Hierarchical (dendrogram) or K-Means + elbow |
| Arbitrary shapes | No | DBSCAN |
| Overlapping | Either | GMM (soft assignments) |

### Anomaly Detection

| Training Data | Recommendation |
|---|---|
| Clean normal data only | One-Class SVM |
| Contains anomalies | Isolation Forest or LOF |
| Local anomalies (context matters) | LOF |
| High dimensions (>100) | Isolation Forest |

### Dimensionality Reduction

| Goal | Recommendation |
|---|---|
| Preprocessing / compression | PCA |
| Visualization (2D/3D) | t-SNE (small data) or UMAP (larger) |
| Separate classes | LDA |
| Non-linear structure | Kernel PCA or t-SNE |

### Recommendation

| Cold Start? | Need Interpretability? | Recommendation |
|---|---|---|
| No | No | Matrix Factorization (SVD) |
| Yes (new users) | — | Content-Based + Popularity |
| Yes (new items) | — | Content-Based |
| Both cold | — | Hybrid (CF + Content) |

## By Data Characteristics

| Data Situation | Avoid | Prefer |
|---|---|---|
| **p >> n** (many features, few samples) | Neural networks, kNN | Lasso, Ridge, SVC (linear), Naive Bayes |
| **n >> p** (many samples, few features) | Naive Bayes (may be too simple) | XGBoost, Random Forest, Neural Nets |
| **Mixed categorical + numerical** | Pure distance-based (kNN, SVM) | CatBoost, Random Forest, XGBoost |
| **Missing values** | kNN, SVC (no missing support) | XGBoost (learns from missing), CatBoost |
| **Imbalanced classes** | Accuracy as metric, standard models | Random Forest (class_weight), XGBoost (scale_pos_weight), anomaly detection methods |
| **Time-series data** | Random train-test split | Time-based split, lag features, gradient boosting with time-aware validation |

## Performance vs Interpretability Spectrum

```
High Interpretability                    Low Interpretability
        ┃                                       ┃
Linear Regression ─── Decision Tree ─── Random Forest ─── XGBoost ─── Neural Network
   ┃                      ┃                  ┃               ┃               ┃
 Explain coefficients    Visualize splits    Feature imp    SHAP values   Saliency maps
```

## Speed Comparison (Relative)

| Algorithm | Training Speed | Inference Speed |
|---|---|---|
| Naive Bayes | ⚡⚡⚡⚡⚡ Fastest | ⚡⚡⚡⚡⚡ |
| Linear / Logistic | ⚡⚡⚡⚡ | ⚡⚡⚡⚡⚡ |
| Decision Tree | ⚡⚡⚡⚡ | ⚡⚡⚡⚡⚡ |
| Random Forest | ⚡⚡ | ⚡⚡⚡⚡ |
| XGBoost | ⚡⚡⚡ | ⚡⚡⚡⚡ |
| LightGBM | ⚡⚡⚡⚡ | ⚡⚡⚡⚡ |
| CatBoost | ⚡⚡ | ⚡⚡⚡ |
| kNN | ⚡⚡⚡⚡⚡ (no training) | ⚡ (slow on large data) |
| SVC (RBF) | ⚡ | ⚡ |
| GMM | ⚡⚡ | ⚡⚡⚡⚡ |
| t-SNE | ⚡ (very slow on 10K+) | N/A (visualization only) |
| Isolation Forest | ⚡⚡⚡⚡ | ⚡⚡⚡⚡ |

## Quick Decision Flowchart

```
What kind of target?
├── Continuous number → Regression section
├── Category →
│   ├── 2 classes → Binary Classification section
│   └── 3+ classes → Multi-Class section
└── No target (just data) →
    ├── Find groups → Clustering section
    ├── Reduce dimensions → Dimensionality Reduction
    └── Find unusual points → Anomaly Detection
```
