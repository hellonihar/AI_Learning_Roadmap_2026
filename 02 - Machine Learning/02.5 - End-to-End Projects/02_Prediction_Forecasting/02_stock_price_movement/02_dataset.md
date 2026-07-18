# Dataset — Stock Price Movement

## Source
Yahoo Finance (`yfinance` library) for daily OHLCV data. Tickers: AAPL, MSFT, GOOGL, AMZN, TSLA, SPY (index).

## Size
- **Rows:** ~5,000 trading days per ticker (2000–2026)
- **Features:** 15 engineered from raw OHLCV
- **Target:** Binary — `1` if close[t+1] > close[t], else `0`

## Raw Features (OHLCV)
| Column | Description |
|--------|-------------|
| Open | Day open price |
| High | Day high price |
| Low | Day low price |
| Close | Day close price |
| Volume | Shares traded |

## Engineered Features
| Feature | Formula | Window |
|---------|---------|--------|
| SMA_10 | Close.rolling(10).mean() | 10 days |
| SMA_50 | Close.rolling(50).mean() | 50 days |
| RSI_14 | Wilder's RSI | 14 days |
| MACD | 12-EMA − 26-EMA | Exp weighted |
| BB_%B | (Close − BB_low) / (BB_high − BB_low) | 20 days |
| Volume_Ratio | Volume / SMA(Volume, 20) | 20 days |

## Known Challenges
- **Class imbalance** — Markets trend up ~53% of days; use class weighting or SMOTE
- **Non-stationarity** — Rolling windows required; never fit on 2010 data to predict 2023
- **Survivorship bias** — yfinance gives current S&P 500 members; backtesting on delisted stocks is missed
- **Market regime shifts** — 2008, 2020 COVID, 2022 rate hikes change volatility structure
