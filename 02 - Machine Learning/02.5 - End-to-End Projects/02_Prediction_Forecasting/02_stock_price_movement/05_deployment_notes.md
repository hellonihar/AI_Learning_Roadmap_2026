# Deployment Notes — Stock Price Movement

## Inference

```python
def predict_signal(ticker):
    df = engineer_features(ticker)
    row = df.iloc[[-1]][feature_cols]
    prob = model.predict_proba(row)[0, 1]
    signal = "BUY" if prob >= 0.60 else "HOLD" if prob >= 0.50 else "SELL"
    return {"ticker": ticker, "signal": signal, "confidence": round(prob, 3)}
```

- Runs after market close (16:30 ET) — signal ready before next open
- **Latency:** < 50 ms per ticker (batch of 50 tickers in ~2 s)

## Retraining Strategy

- **Rolling window:** retrain every 5 trading days on last 2 years of data
- **Why frequent:** market regimes shift fast (earnings, macro, policy)
- **Cold start:** new ticker gets 250 trading days of warm-up before live signals

## Monitoring

| Signal | Method | Alert |
|--------|--------|-------|
| Precision drop | Daily evaluation vs actual move | Precision < 0.52 over 20-day window |
| Regime shift | Hidden Markov Model on returns | State change detected → retrain |
| Feature drift | K-S test on RSI distribution | p < 0.05 → investigate |
| Sharpe ratio | Trading simulation of signals | Sharpe < 0 → disable signals |

## Key Risks
- **Look-ahead bias** — Never use data not available at prediction time (e.g., next-day news sentiment)
- **Overfitting to ticker** — Model may memorise AAPL's earnings calendar; validate on out-of-sample tickers
- **Regulatory** — Not financial advice; disclaimers required if shown to end users
