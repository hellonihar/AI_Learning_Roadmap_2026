# Results — Stock Price Movement

## Benchmark Scores (SPY / AAPL, 2020–2025 test)

| Model | Precision (Up) | Recall (Up) | F1 | Accuracy |
|-------|----------------|-------------|----|----------|
| Random Guess | 0.53 | 0.53 | 0.53 | 53.0% |
| Logistic Regression | 0.55 | 0.58 | 0.56 | 54.2% |
| Random Forest (300) | **0.58** | **0.62** | **0.60** | **56.8%** |
| XGBoost | 0.57 | 0.61 | 0.59 | 55.9% |

## Winning Model — Random Forest

- More robust to outlier days (flash crash, Fed announcements)
- Feature importance interpretable for user-facing signals
- XGBoost overfit slightly on noisy days (high variance)

## Feature Importance (Top 5)
1. `Price_SMA_ratio` — momentum vs. recent average
2. `RSI_14` — overbought/oversold regime
3. `MACD` — trend-following signal
4. `Volume` — spike confirms breakout
5. `SMA_50` — long-term trend filter

## What Was Learned

1. **Direction > Price prediction** — classification avoids unit-root pitfalls
2. **Technical indicators work, barely** — edge of ~3–5% over random; no magic strategy exists
3. **Volatility regimes matter** — model works well in trending markets, fails in choppy ranges
4. **Ensemble of tickers** — training on AAPL+MSFT+GOOGL together improved generalisation by +1% vs per-ticker models

## Limits
- Transaction costs (0.1% per trade) erase most edge below 55% accuracy
- After 2022, performance degraded — market structure shifted (retail algo dominance)
