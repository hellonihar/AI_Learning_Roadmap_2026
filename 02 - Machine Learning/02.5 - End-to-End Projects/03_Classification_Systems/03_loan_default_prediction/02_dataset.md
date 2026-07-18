# Dataset — Loan Default Prediction

## Source
- **LendingClub** public data (2007–2020), filtered to complete loans.
- ~2.2 million loans, ~150 raw columns.

## Features (selected subset)
- `loan_amnt`, `term`, `int_rate`, `installment`
- `grade`, `sub_grade` (LendingClub risk grades A–G)
- `emp_length`, `annual_inc`, `home_ownership`
- `dti` (debt-to-income ratio), `revol_util`
- `delinq_2yrs`, `pub_rec`, `open_acc`, `total_acc`
- `earliest_cr_line` → derived feature: credit history length

## Target
- `loan_status` mapped to binary: `Fully Paid` = 0, `Charged Off` = 1.
- Default definition excludes "Current" and "In Grace Period".

## Class Balance
- Fully paid: ~78%
- Defaulted: ~22%
- Moderately imbalanced; no special sampling needed.

## Known Challenges
- **Survivorship bias** — only approved loans appear in data; no rejection inferences.
- **Time leakage** — model trained on 2015–2018 must be tested on 2019+ to validate temporal stability.
- **Missing income** for ~15% of applicants (self-reported missing).
- **Lookahead bias** — features like `mths_since_last_delinq` can leak future information.
