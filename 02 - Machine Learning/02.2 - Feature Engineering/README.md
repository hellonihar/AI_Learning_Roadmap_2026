# 02.2 — Feature Engineering

Feature engineering is the process of transforming raw data into inputs that ML models can learn from. It is often the highest-leverage activity in applied ML — good features can make simple models perform better than complex models on raw data.

## Sections

| # | Section | Topics |
|---|---|---|
| 01 | [What Is Feature Engineering](01_What_Is_Feature_Engineering/) | Features vs raw data, engineering vs selection, signal extraction, when it matters more than algorithms |
| 02 | [Understanding Data Before Engineering](02_Understanding_Data_Before_Engineering/) | Feature types, cardinality, distributions, target leakage awareness |
| 03 | [Missing Values Handling](03_Missing_Values_Handling/) | MCAR vs MNAR, deletion, mean/median/mode imputation, flagging missingness, when not to impute |
| 04 | [Encoding Categorical Variables](04_Encoding_Categorical_Variables/) | Label, one-hot, ordinal, high-cardinality strategies, dummy trap, encoding leakage |
| 05 | [Feature Scaling & Normalization](05_Feature_Scaling_Normalization/) | Standardization, min-max, robust scaling, when scaling is needed, train-test split order |
| 06 | [Handling Outliers](06_Handling_Outliers/) | Outliers vs rare values, detection, capping, removal vs transformation, when outliers help |
| 07 | [Feature Transformation](07_Feature_Transformation/) | Log, Box-Cox, power transforms, handling skewness, non-linear transforms, interpretability trade-offs |
| 08 | [Feature Creation](08_Feature_Creation/) | Domain-driven features, ratios, aggregations, interactions, binning, polynomial features |
| 09 | [Date & Time Features](09_Date_Time_Feature_Engineering/) | Component extraction, sin/cos encoding, lag features, rolling windows, time-based leakage |
| 10 | [Text Features (Intro)](10_Text_Feature_Engineering/) | Cleaning, BoW, TF-IDF, when to use embeddings, what's deferred to NLP |
| 11 | [Feature Selection](11_Feature_Selection/) | Correlation filtering, variance thresholding, domain-driven selection, vs dimensionality reduction |
| 12 | [Pipelines](12_Feature_Engineering_Pipelines/) | Fit vs transform, order of operations, reproducibility, preventing leakage |
| 13 | [Pitfalls](13_Pitfalls/) | Leakage, over-engineering, feature explosion, ignoring domain knowledge |
| — | [Projects](projects/) | 3 hands-on projects: house prices, fraud detection, comparison experiment |

## Core Principles

1. **Engineer with the real world in mind** — would this feature be available at prediction time?
2. **Start simple** — a few good features beat hundreds of noisy ones
3. **Domain knowledge first** — ask experts "what matters?" before guessing
4. **Never let test data influence training transformations** — fit on train, transform test
5. **Pipeline everything** — reproducibility and leakage prevention
6. **Validate each feature** — does it improve performance on validation data (not training)?

**Total**: 14 files across 13 sections + projects. 2-3 real-world examples per topic.
