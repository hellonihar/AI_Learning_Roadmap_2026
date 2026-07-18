# Dataset — Credit Risk

## Source
Kaggle — Credit Risk Dataset (or UCI German Credit / LendingClub subset).  
[Link](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

## Size
- 1,000 rows, 12 columns (German Credit format)
- Clean — no missing values

## Features

| Feature | Type | Description |
|---------|------|-------------|
| age | numeric | Age in years |
| sex | binary | male / female |
| job | ordinal | Job type (0–3) |
| housing | categorical | own / rent / free |
| saving_accounts | categorical | little / moderate / quite rich / rich |
| checking_account | categorical | little / moderate / rich |
| credit_amount | numeric | Loan amount |
| duration | numeric | Loan duration (months) |
| purpose | categorical | car / furniture / education / etc. |
| risk | binary | Target: good (0) / bad (1) |

## Known Challenges
- **Small dataset** (1,000 rows) — risk of overfitting.
- **Class imbalance** — 70% good, 30% bad.
- **Categorical features** with rare levels need grouping.
- **WoE binning** requires careful handling of monotonicity.
