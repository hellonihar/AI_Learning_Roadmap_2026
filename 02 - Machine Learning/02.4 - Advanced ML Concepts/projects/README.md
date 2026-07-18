# Advanced ML Concepts — Projects

## Project 1: Hyperparameter Tuning Showdown

Compare grid search, random search, and Bayesian optimization on a medium-sized dataset.

**Tasks:**
1. Choose a model (XGBoost, Random Forest, or SVC)
2. Define a search space with 5-8 hyperparameters
3. Run grid search (limit to ~50 combos by using coarse ranges)
4. Run random search with 50 iterations
5. Run Bayesian optimization (use Optuna or scikit-optimize) with 50 iterations
6. Compare: best score achieved, time taken, number of near-optimal configurations found
7. Plot the learning curves: performance vs trial number for each method

**Questions**: Which method found the best configuration fastest? Was the grid search winner the same as Bayesian?

---

## Project 2: Imbalanced Data Experiment

Take a highly imbalanced dataset and compare all handling strategies.

**Tasks:**
1. Use a dataset with <5% minority class (Credit Card Fraud, Mammography, or create synthetic)
2. Baseline: train a classifier with no imbalance handling. Report accuracy, precision, recall, F1, confusion matrix
3. Apply each strategy: oversampling, undersampling, SMOTE, class weighting, threshold moving
4. Compare all strategies on the same metrics — which improves recall most? Which improves F1?
5. Check: does oversampling cause overfitting? (compare train vs test performance)
6. Find the optimal threshold via precision-recall curve

**Questions**: Which strategy worked best for your dataset? Why? Did any strategy make the model worse?

---

## Project 3: Model Interpretation Deep-Dive

Train a black-box model (XGBoost or neural network) and explain its predictions using multiple methods.

**Tasks:**
1. Train an XGBoost classifier on any tabular dataset
2. Get built-in feature importance — do the top features make sense?
3. Compute permutation importance — does it agree with built-in importance?
4. Pick 3 individual predictions (one correct, one incorrect, one borderline). Use SHAP to explain each.
5. Repeat with LIME — do the explanations agree? Where do they differ?
6. Create a summary SHAP beeswarm plot for global interpretation
7. Create partial dependence plots for the top 3 features

**Questions**: Which explanation method was most intuitive? Did SHAP and LIME agree on the 3 individual predictions? What did you learn about your model's behavior?

---

## Project 4: Model Selection Playground

Design a model selection experiment with statistical rigor.

**Tasks:**
1. Choose 1 dataset and 4 diverse models (e.g., Logistic Regression, Random Forest, XGBoost, kNN)
2. Use nested cross-validation (outer 5-fold, inner 3-fold) to compare them
3. Report: mean ± std of validation metric for each model
4. Run a statistical test (McNemar's or Wilcoxon) to determine if the best model is significantly better than the second-best
5. Also compare training time and inference time
6. Create a decision matrix: which model would you choose if interpretability is most important? If speed is most important? If accuracy is most important?

**Questions**: Was the best model statistically significantly better? How much accuracy are you willing to trade for interpretability or speed?

---

## Project 5: Online Learning vs Batch

Compare batch retraining vs online learning on a streaming dataset.

**Tasks:**
1. Create a simulated data stream (use a large dataset split into time-ordered chunks)
2. Train a batch model on the first 3 months of data. Evaluate on each subsequent month without updating.
3. Train an online model (SGDClassifier with `partial_fit`). Update it on each month's data. Evaluate on each subsequent month.
4. Plot performance over time for both models
5. Simulate concept drift: reverse the relationship of a feature mid-stream. How does each model cope?
6. Compare: total training time, peak memory usage, final accuracy

**Questions**: Which model was more robust to concept drift? How much more computation did online learning require?
