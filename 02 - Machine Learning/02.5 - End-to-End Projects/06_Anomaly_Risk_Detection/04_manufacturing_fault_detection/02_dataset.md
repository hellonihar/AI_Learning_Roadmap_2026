# Manufacturing Fault Detection — Dataset

## Source
[SECOM Dataset](https://archive.ics.uci.edu/ml/datasets/SECOM) (UCI) — real sensor readings from a semiconductor manufacturing process. 590 features per wafer (sensor measurements + process parameters).

## Size & Shape
- **Total**: 1,567 wafers, 590 sensor features
- **Train/Test split**: 80/20 stratified by fault class
- **Target**: `pass/fail` — ~7% failure rate
- **Timestamp**: Sorted by processing order (time-series)

## Challenges
- **High dimensionality** — 590 features for only 1,567 samples
- **Missing values** — Many sensors have null readings for certain wafer steps (sensors only active during specific phases)
- **Imbalanced** — 7% failures; standard accuracy is misleading
- **Temporal dependence** — Wafers are processed in batches; consecutive wafers share conditions. Standard CV leaks information
- **Noisy sensors** — Some sensors show drift over time unrelated to faults
