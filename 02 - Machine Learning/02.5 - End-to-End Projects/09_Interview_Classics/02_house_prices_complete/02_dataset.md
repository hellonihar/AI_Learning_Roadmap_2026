# Dataset — House Prices

## Source
Kaggle — House Prices: Advanced Regression Techniques.  
[Link](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Size
- Train: 1,460 rows, 81 columns (79 features + Id + SalePrice)
- Test: 1,459 rows

## Features
79 features covering:
- **Lot**: area, shape, contour, utilities
- **Building**: year built/remodeled, style, quality, condition
- **Rooms**: bedrooms, bathrooms, kitchen, fireplace, garage
- **Basement**: area, finish, height
- **Porch/Deck**: multiple porch types
- **Location**: neighborhood, zone
- **Sale**: month, year, condition

## Target
- `SalePrice`: continuous, right-skewed (range $34,900 – $755,000)

## Known Challenges
- **Missing values** in 19 features — most indicate absence (e.g., PoolArea=0 → PoolQC=NaN).
- **Skewed target** — log-transform required.
- **Outliers** — some houses sold well below/above market (e.g., $34,900).
- **High cardinality** — Neighborhood (25), MSSubClass (15).
- **Feature interactions** — overall quality × area, year built × year remodeled.
