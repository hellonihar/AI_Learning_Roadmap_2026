# Dataset — Medical Cost (Kaggle)

## Source
Kaggle — Medical Cost Personal Datasets.  
[Link](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## Size
- 1,338 rows, 7 columns
- No missing values

## Features

| Feature | Type | Description |
|---------|------|-------------|
| age | numeric | Age in years |
| sex | binary | male / female |
| bmi | numeric | Body mass index |
| children | numeric | Number of dependents |
| smoker | binary | yes / no |
| region | categorical | southwest, southeast, northwest, northeast |

## Target
- `charges`: Annual medical costs in USD. Range: $1,122 – $63,770.

## Known Challenges
- **Heavy right skew** — most charges < $15k, a long tail up to $64k.
- **Smoker interaction** — smokers have much higher charges, especially with age.
- **Small dataset** — only 1,338 rows, limits complex models.
- **No temporal features** — can't model cost growth over time.
