# Dataset — Ames Housing

## Source
[Ames Housing Dataset](http://jse.amstat.org/v19n3/decock.pdf) (Dean De Cock, 2011). Alternative: Kaggle "House Prices — Advanced Regression Techniques."

## Size
- **Rows:** 1,460 residential sales
- **Features:** 79 (23 nominal, 23 ordinal, 14 discrete, 20 continuous)
- **Target:** `SalePrice` (integer, \$)

## Key Features
| Group | Examples |
|-------|----------|
| Lot/Zone | `LotArea`, `MSZoning`, `Neighborhood` |
| Structure | `YearBuilt`, `TotalBsmtSF`, `GarageCars` |
| Interior | `KitchenQual`, `Fireplaces`, `OverallQual` |
| Utilities | `CentralAir`, `HeatingQC` |

## Known Challenges
- **Skewed target** — SalePrice is right-skewed; log-transform recommended
- **Missing values** — ~20 columns have sparse NAs (e.g., `PoolQC` = 99% NA, `Alley` = 93% NA). These encode "absence" not error
- **Outliers** — GrLivArea vs SalePrice has extreme high-leverage points (e.g., 4,000+ sqft sold cheap)
- **Ordinal encoding** — Quality columns (Ex, Gd, TA, Fa, Po) need manual mapping, not one-hot
- **Temporal leakage** — `YrSold`/`MoSold` could leak post-sale info if not handled

## Train/Test Split
Original Kaggle provides train (1,460) and test (1,459) already separated. Cross-validation uses 5-fold on the training set.
