# Dataset — Titanic Survival

## Source
Kaggle — Titanic: Machine Learning from Disaster.  
[Link](https://www.kaggle.com/c/titanic)

## Size
- Train: 891 rows, 12 columns
- Test: 418 rows (no labels)

## Features

| Feature | Type | Description |
|---------|------|-------------|
| PassengerId | numeric | Unique ID |
| Survived | binary | Target (0 = died, 1 = survived) |
| Pclass | ordinal | Ticket class (1, 2, 3) |
| Name | text | Passenger name |
| Sex | binary | male / female |
| Age | numeric | Age in years (177 missing) |
| SibSp | numeric | # siblings / spouses aboard |
| Parch | numeric | # parents / children aboard |
| Ticket | text | Ticket number |
| Fare | numeric | Passenger fare |
| Cabin | text | Cabin number (687 missing) |
| Embarked | categorical | Port (C = Cherbourg, Q = Queenstown, S = Southampton) |

## Known Challenges
- 177 missing Age — impute by Title median.
- 687 missing Cabin — extract deck letter, fill missing as "U".
- 2 missing Embarked — fill with mode.
- Name and Ticket are high-cardinality raw text — must engineer features.
- Sex + Pclass creates strong interaction (women in 1st class: high survival).
