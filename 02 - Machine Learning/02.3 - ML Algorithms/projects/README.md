# ML Algorithms — Projects

## Project 1: Regression Showdown

Compare linear, ridge, lasso, elastic net, and decision tree regression on a housing dataset.

**Tasks:**
1. Load a dataset (Ames Housing, Boston, California Housing)
2. Train: Linear Regression, Ridge (α=1.0), Lasso (α=0.1), ElasticNet (α=0.1, l1_ratio=0.5), Decision Tree Regressor
3. Compare RMSE, MAE, R² on validation set
4. For linear models: examine coefficients — which features got zeroed out by Lasso?
5. For ridge vs lasso: which handles multicollinearity better? (check coefficient stability)
6. Plot predicted vs actual for all models

**Questions**: Which model performed best? Why? Was the trade-off between bias and variance visible?

---

## Project 2: Classification Comparison

Compare logistic regression, kNN, Naive Bayes, decision tree, and random forest on a text or tabular classification task.

**Tasks:**
1. Dataset options: Spam Collection (text) or UCI Heart Disease (tabular)
2. Train: Logistic Regression, kNN (k=3,5,11), Gaussian/ MultinomialNB, Decision Tree, Random Forest
3. Compare accuracy, precision, recall, F1, confusion matrix
4. For kNN: try k=1, 3, 5, 11, 21. Plot validation accuracy vs k. What's the optimal k?
5. For Naive Bayes: compare GaussianNB vs MultinomialNB on the same data. Why do results differ?
6. Which models handle class imbalance best?

**Questions**: When would you pick a simple model (logistic regression) over a complex one (random forest)?

---

## Project 3: Boosting Benchmark

Compare XGBoost, LightGBM, and CatBoost on a medium-sized tabular dataset with mixed numerical/categorical features.

**Tasks:**
1. Use a dataset with both numerical and categorical features (e.g., Kaggle Insurance, Employee Attrition)
2. Train each model with default parameters first
3. Train each model with tuned parameters (use randomized search or grid search)
4. Compare: validation score, training time, inference time, model size (memory)
5. For CatBoost: use `cat_features` parameter. How does native categorical handling affect performance vs one-hot encoding for XGBoost/LightGBM?
6. Plot feature importance from each model — do they agree on top features?

**Questions**: Which library is fastest? Which gives the best accuracy? When would you sacrifice accuracy for speed?

---

## Project 4: Customer Segmentation

Apply K-Means, DBSCAN, and GMM to a customer dataset and compare results.

**Tasks:**
1. Use a customer dataset (e.g., Mall Customers, Online Retail, or generate synthetic data with make_blobs)
2. K-Means: find optimal k using elbow method + silhouette score. Visualize clusters.
3. DBSCAN: tune eps and min_samples. How many noise points does it label?
4. GMM: compare full vs diag vs spherical covariance. Visualize soft assignments (probability heatmap).
5. Compare cluster assignments between algorithms for a sample of points. Where do they disagree?
6. Visualize all three results side-by-side in 2D (using PCA if needed).

**Questions**: When would you prefer DBSCAN's ability to label noise over K-Means's forced assignments? When is GMM's soft assignment valuable?

---

## Project 5: Anomaly Detection Lab

Compare Isolation Forest, One-Class SVM, and LOF on a fraud detection dataset.

**Tasks:**
1. Use a fraud/ anomaly dataset (e.g., Credit Card Fraud, KDD Cup, or add synthetic outliers to any dataset)
2. Train: Isolation Forest (contamination=0.05), One-Class SVM (nu=0.05), LOF (n_neighbors=20)
3. Compare: precision, recall, F1 at different thresholds
4. Which model catches the most true anomalies? Which has the most false positives?
5. Check if anomalies are "global" (far from all points) or "local" (unusual for their neighborhood). Does LOF's local approach catch different anomalies?
6. Try varying the contamination assumption (0.01, 0.05, 0.1). How sensitive is each model?

**Questions**: For this problem, which cost matters more — false positives (wasting investigation time) or false negatives (missing fraud)?

---

## Project 6: Recommender Mini-Lab

Build a simple movie recommender using collaborative filtering, content-based filtering, and matrix factorization.

**Tasks:**
1. Use MovieLens Small (100K ratings)
2. **Collaborative (item-based)**: Compute cosine similarity between movies. For a given movie, recommend the top-5 most similar movies based on rating patterns.
3. **Content-based**: Use movie genres as features. For a given movie, find the 5 most similar movies by genre vector similarity.
4. **Matrix factorization**: Use TruncatedSVD to decompose the user-movie rating matrix (k=20). Predict ratings. Compare predicted vs actual on test set.
5. Compare the three approaches — same recommendations? Different? Why?

**Questions**: How would you hybridize these approaches for a production system?
