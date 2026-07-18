# Problem Statement — Stock Price Movement

## Business Context
A retail trading app wants to notify users of likely *directional moves* in the next trading day. Users do not need exact prices — they need a buy/hold/sell signal to decide position entry.

## Problem Type
Binary classification. Predict whether tomorrow's closing price will be **higher (1)** or **lower (0)** than today's.

## Why Classification, Not Regression
- Price levels are non-stationary (random walk); regression models fail without constant differencing
- Classification on direction is more robust to volatility shifts
- Aligns with trader decision: "should I enter?" not "what will the price be?"

## Success Metrics
| Metric | Target | Rationale |
|--------|--------|-----------|
| **Precision (up-class)** | ≥ 60% | Fewer false buy signals → user trust |
| **F1-score** | ≥ 0.55 | Balance precision/recall |
| **Accuracy** | > 52% | Beats coin-flip baseline; financial ML rarely exceeds 55–58% |

## Constraints
- No future data leakage (look-ahead bias is the #1 pitfall)
- Must work across 10+ tickers without per-ticker tuning
- Features must be computable at market close, before next open
