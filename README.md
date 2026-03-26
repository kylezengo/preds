# preds — Stock Direction Prediction

Predicts whether a stock will go up or down tomorrow. Uses walk-forward backtesting.

## Workflow

1. **Download data** — run `download_data.save_data_to_csv()` and `download_gt_data.save_gt_data_to_csv()`
2. **Run a backtest** — open `Predict a Stock.ipynb`, set `TICKER`, run all cells
3. **Batch run** — `Best Prediction.ipynb` runs the best config across many tickers

## Key files

| File | Purpose |
|---|---|
| `config.py` | Shared settings (start date, transaction cost, OOS split point) |
| `download_data.py` | Fetches prices, VIX, yields, weather, Wikipedia pageviews, sector ETFs |
| `download_gt_data.py` | Fetches Google Trends data |
| `prep_data.py` | Builds the feature matrix (~37 features) |
| `strat_defs.py` | Strategy definitions + backtesting engine |
| `analysis.py` | Prints edge report with Sharpe, accuracy, p-value vs base rate |

## Data sources

Stock prices (yfinance), Wikipedia pageviews, Google Trends, NOAA weather, Federal funds rate, VIX term structure, treasury yields, SPDR sector ETFs.