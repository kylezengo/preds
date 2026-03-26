"""This module builds prepd_data"""

import glob
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytz
from astral import LocationInfo
from astral.sun import sun


@dataclass
class MovingAverageConfig:
    """
    Moving average configuration class
    """
    short_window: int = 20
    long_window: int = 50

@dataclass
class BollingerConfig:
    """
    Bollinger configuration class
    """
    window: int = 20
    num_std: float = 2.0

@dataclass
class MACDConfig:
    """
    MACD configuration class
    """
    short_window: int = 12
    long_window: int = 26

@dataclass
class IndicatorConfig:
    """
    Configuration class for technical indicators used in forecasting strategies.
    """
    target: str = 'Adj Close'
    ticker: str = 'SPY'
    rsi_window: int = 30
    vwap_window: int = 60   # rolling window for VWAP (days)
    bko_window: int = 20    # rolling window for breakout high/low range (days)
    # Day-of-week dummies were top features in XGBoost importance (2026-03-26) but likely
    # spurious — model was trading on day-of-week cycles rather than market signal.
    # Disabled by default; re-enable to test if useful for a specific model/context.
    include_day_of_week: bool = False
    moving_average: MovingAverageConfig = field(default_factory=MovingAverageConfig)
    bollinger: BollingerConfig = field(default_factory=BollingerConfig)
    macd: MACDConfig = field(default_factory=MACDConfig)


def load_data():
    """
    Load data from CSV files.
    """
    stocks_df_files = glob.glob('data/stocks_df_*.csv')
    if not stocks_df_files:
        raise FileNotFoundError("No stocks_df files found in data/")
    def _latest(files):
        return max(files, key=lambda f: f.split("_")[-1])

    stocks_df_raw = pd.read_csv(_latest(stocks_df_files), parse_dates=['Date'])

    wiki_pageviews_files = glob.glob('data/wiki_pageviews_*.csv')
    if not wiki_pageviews_files:
        raise FileNotFoundError("No wiki_pageviews files found in data/")
    wiki_pageviews = pd.read_csv(_latest(wiki_pageviews_files), parse_dates=['Date'])

    ffr_files = glob.glob('data/ffr_*.csv')
    if not ffr_files:
        raise FileNotFoundError("No ffr files found in data/")
    ffr = pd.read_csv(_latest(ffr_files), parse_dates=['Date'])

    weather_files = glob.glob('data/weather_*.csv')
    if not weather_files:
        raise FileNotFoundError("No weather files found in data/")
    weather = pd.read_csv(_latest(weather_files), parse_dates=['date'])

    gt_adjusted_files = glob.glob('data/gt_adjusted_*.csv')
    if not gt_adjusted_files:
        raise FileNotFoundError("No gt_adjusted files found in data/")
    gt_adjusted = pd.read_csv(_latest(gt_adjusted_files), parse_dates=['date'])

    vix_yields_files = glob.glob('data/vix_yields_*.csv')
    if not vix_yields_files:
        raise FileNotFoundError("No vix_yields files found in data/")
    vix_yields = pd.read_csv(_latest(vix_yields_files), parse_dates=['Date'])

    sp_df_files = glob.glob('data/sp_df_*.csv')
    if not sp_df_files:
        raise FileNotFoundError("No sp_df files found in data/")
    sp_df = pd.read_csv(_latest(sp_df_files))

    sector_etfs_files = glob.glob('data/sector_etfs_*.csv')
    if not sector_etfs_files:
        raise FileNotFoundError("No sector_etfs files found in data/")
    sector_etfs = pd.read_csv(_latest(sector_etfs_files), parse_dates=['Date'])

    return stocks_df_raw, wiki_pageviews, ffr, weather, gt_adjusted, vix_yields, sp_df, sector_etfs

# Technical indicators
def calculate_rsi(data, target, ticker, window):
    """
    Calculate the Relative Strength Index (RSI).

    Parameters:
        data (DataFrame): Stock data with target prices.
        target (str): column to predict (usually Adj Close)
        ticker (str): Stock ticker
        window (int): Lookback period for RSI.

    Returns:
        Series: RSI values.
    """
    delta = data[target+"_"+ticker].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))

def calculate_vwap(data, target, ticker, window=60):
    """
    Calculate rolling Volume Weighted Average Price.

    Uses a rolling window (default 60 days) rather than a cumulative sum from
    the start of the dataset. A cumulative VWAP over a 10-year dataset becomes
    correlated with the absolute price level (VIF ~154), making it a redundant
    feature. A rolling VWAP captures near-term volume-weighted positioning instead.

    Parameters:
        data (DataFrame): Stock data with required columns.
        target (str): column to predict (usually Adj Close)
        ticker (str): Stock ticker
        window (int): Rolling window in days. Default 60.

    Returns:
        Series: VWAP values.
    """
    price_volume = data[target + "_" + ticker] * data["Volume_" + ticker]
    return (
        price_volume.rolling(window=window, min_periods=1).sum()
        / data["Volume_" + ticker].rolling(window=window, min_periods=1).sum()
    )

def calculate_technical_indicators(data, config: IndicatorConfig):
    """
    Calculate technical indicators for the dataset.

    Parameters:
        data (DataFrame): Stock data with required columns.
        config (IndicatorConfig): Configuration object for technical indicators.

    Returns:
        DataFrame: Original dataframe with technical indicators included 
    """
    data = data.copy()
    target_ticker = config.target+"_"+config.ticker
    price = data[target_ticker]

    data['RSI'] = calculate_rsi(data, config.target, config.ticker, window=config.rsi_window)
    data['MA_S'] = price.rolling(window=config.moving_average.short_window).mean()
    data['MA_L'] = price.rolling(window=config.moving_average.long_window).mean()
    data['MA_B'] = price.rolling(window=config.bollinger.window).mean()
    data['Bollinger_Upper'] = (data['MA_B'] +
                               config.bollinger.num_std *
                               price.rolling(window=config.bollinger.window).std())
    data['Bollinger_Lower'] = (data['MA_B'] -
                               config.bollinger.num_std *
                               price.rolling(window=config.bollinger.window).std())
    data['VWAP'] = calculate_vwap(data, config.target, config.ticker, window=config.vwap_window)
    data['short_ema'] = price.ewm(span=config.macd.short_window, adjust=False).mean()
    data['long_ema'] = price.ewm(span=config.macd.long_window, adjust=False).mean()
    data['macd_line'] = data['short_ema'] - data['long_ema']

    # ── Multicollinearity reduction ───────────────────────────────────────────
    # MA_S, MA_L, MA_B, Bollinger_Upper/Lower, VWAP, short_ema, long_ema are all
    # rolling windows of the same price series → VIF = inf (perfectly collinear).
    # macd_line is an absolute dollar difference that also scales with price → VIF ~3e15.
    # Replace all with ratios that capture relative position without the collinearity.
    # SMA/VWAP/Bollinger strategies in strat_defs.py use the ratio columns directly.
    # See "Multicollinearity Audit" in Exploratory data analysis.ipynb.
    data['price_vs_ma_s']   = price / data['MA_S'] - 1          # above/below short MA
    data['ma_crossover']    = data['MA_S'] / data['MA_L'] - 1   # SMA crossover signal
    data['bollinger_pct_b'] = (                                  # %B: 0=lower, 1=upper band
        (price - data['Bollinger_Lower']) /
        (data['Bollinger_Upper'] - data['Bollinger_Lower'])
    )
    data['price_vs_vwap']   = price / data['VWAP'] - 1          # above/below VWAP
    data['macd_pct']        = data['macd_line'] / data['long_ema']  # MACD / price scale

    data = data.drop(columns=['MA_S', 'MA_L', 'MA_B', 'Bollinger_Upper', 'Bollinger_Lower',
                               'VWAP', 'short_ema', 'long_ema', 'macd_line'])

    # Breakout feature: position within recent high/low range (stochastic-style %K).
    # < 0: price below recent low (bearish breakout); > 1: above recent high (bullish).
    # Precomputed here so the Breakout strategy doesn't need Adj Close_{ticker} at
    # backtest time, allowing the raw price level to be dropped from the feature set.
    # Skipped when High/Low columns are absent (e.g. unit tests).
    high_col = 'High_' + config.ticker
    low_col  = 'Low_'  + config.ticker
    if high_col in data.columns and low_col in data.columns:
        rolling_high = data[high_col].rolling(window=config.bko_window).max()
        rolling_low  = data[low_col].rolling(window=config.bko_window).min()
        data['breakout_pct'] = (price - rolling_low) / (rolling_high - rolling_low)

    return data


# Build dataframes
def gen_stocks_w(ticker, stocks_df, wiki_pageviews, drop_tickers=None):
    """
    Generates stocks_w dataframe

    Parameters:
        ticker (str): Stock ticker
        drop_tickers (bool): Whether to drop other tickers

    Returns:
        DataFrame: stocks_w dataframe
    """

    stocks_df = stocks_df.copy()

    tickers_list = stocks_df['ticker'].unique()
    tickers_list = tickers_list[tickers_list != "SPY"]

    drop_tickers_list = [x for x in tickers_list if x!= ticker]

    # Can drop other ticker columns for testing (so runs quicker)
    if drop_tickers:
        stocks_df = stocks_df.loc[~stocks_df['ticker'].isin(drop_tickers_list)]


    # Set up data frame for testing
    stocks_df['movement'] = stocks_df.groupby('ticker')['Adj Close'].diff() * stocks_df['Volume']

    # wiki_pageviews is expected to already have 'views_prev_day' (lagged by 1 day in prep_data)
    stocks_df = stocks_df.merge(
        wiki_pageviews[['Date','ticker','views_prev_day']],how='left', on=['Date','ticker']
    )

    # Some companies have calss A, B, etc. stock. See sp_df for details. Choosing class A for now.
    # Maybe keep in the future? Trying to reduce multicolinearity for now
    # Alphabet: GOOGL, GOOG; Fox: FOX, FOXA; News Corp: NWS, NWSA
    stocks_df = stocks_df[~stocks_df['ticker'].isin(['GOOG', 'FOX', 'NWS'])].reset_index(drop=True)

    # Pivot
    stocks_w = stocks_df.pivot(
        index='Date',
        columns='ticker',
        values=['Open','High','Low','Close','Adj Close','Volume','movement','views_prev_day']
    )

    stocks_w.columns = ['_'.join(col).strip() for col in stocks_w.columns.values]
    stocks_w = stocks_w.reset_index().rename_axis(None, axis=1)

    return stocks_w

def prep_data(
        stocks_df, wiki_pageviews, ffr_raw, weather, gt_adjusted, vix_yields,
        sp_df, sector_etfs,
        config: IndicatorConfig, drop_tickers=None
):
    """
    Prepare data for forecasting strategies (add some extra features).

    Parameters:
        config (IndicatorConfig): Configuration object for technical indicators.
        drop_tickers (bool): Whether to drop other tickers.

    Returns:
        DataFrame: Prepared data with additional features and technical indicators.
    """
    target_ticker = config.target+"_"+config.ticker

    # Wikipedia pageviews are available with a ~1-day API lag, so we shift the date
    # forward by 1 day before passing to gen_stocks_w. The resulting pivot columns will be
    # named 'views_prev_day_TICKER' rather than 'views_TICKER' to make this explicit:
    # these are prior-day pageviews, not same-day. Using same-day views would introduce
    # a data availability leak in live trading.
    wiki_pageviews_lagged = wiki_pageviews.copy()
    wiki_pageviews_lagged['Date'] = wiki_pageviews_lagged['Date'] + pd.Timedelta(days=1)
    wiki_pageviews_lagged = wiki_pageviews_lagged.rename(columns={'views': 'views_prev_day'})

    prepd_data = gen_stocks_w(config.ticker, stocks_df, wiki_pageviews_lagged, drop_tickers)

    # Sunlight
    nyc = LocationInfo("New York City", "USA", "America/New_York", 40.7128, -74.0060)
    nyc_tz = pytz.timezone("America/New_York")

    def _daylight_seconds(d):
        s = sun(nyc.observer, date=d, tzinfo=nyc_tz)
        return (s['sunset'] - s['sunrise']).total_seconds()

    prepd_data['sunlight_nyc'] = prepd_data['Date'].apply(_daylight_seconds)

    # Federal funds rate
    prepd_data = prepd_data.merge(ffr_raw,on='Date',how='left')

    # Sector ETF relative strength
    sector_etf_map = {
        'Information Technology': 'XLK', 'Health Care': 'XLV', 'Financials': 'XLF',
        'Energy': 'XLE', 'Industrials': 'XLI', 'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP', 'Materials': 'XLB', 'Real Estate': 'XLRE',
        'Utilities': 'XLU', 'Communication Services': 'XLC',
    }
    ticker_sector = sp_df.loc[sp_df['Symbol'] == config.ticker, 'GICS Sector']
    if not ticker_sector.empty and ticker_sector.iloc[0] in sector_etf_map:
        etf = sector_etf_map[ticker_sector.iloc[0]]
        etf_col = f'close_{etf}'
        etf_returns = sector_etfs[['Date', etf_col]].copy()
        etf_returns['sector_etf_return_prev_day'] = etf_returns[etf_col].pct_change().shift(1)
        prepd_data = prepd_data.merge(
            etf_returns[['Date', 'sector_etf_return_prev_day']], on='Date', how='left'
        )
        # sector vs SPY: use Adj Close_SPY already in prepd_data as the broad market benchmark
        spy_return = prepd_data['Adj Close_SPY'].pct_change().shift(1)
        prepd_data['sector_vs_spy_return'] = prepd_data['sector_etf_return_prev_day'] - spy_return

    # VIX and treasury yields
    prepd_data = prepd_data.merge(vix_yields, on='Date', how='left')
    prepd_data['vix_change_prev_day'] = prepd_data['vix_close'].pct_change().shift(1)
    prepd_data['vix_pct_rank_252'] = (
        prepd_data['vix_close'].rolling(252).rank(pct=True)
    )
    # Drop raw VIX levels (VIF: vix_close=5327, vix3m_close=4977, vix6m_close=3424,
    # vix9d_close=1264) — term structure ratios (vix9d_to_vix, vix_to_vix3m) and
    # vix_pct_rank_252 capture the same information without the collinearity.
    # Also drop vix_to_vix3m (VIF=357) — it is correlated 0.72 with vix9d_to_vix
    # and 0.77 with vix_pct_rank_252; vix9d_to_vix is the more near-term signal.
    # See "Multicollinearity Audit" in Exploratory data analysis.ipynb.
    prepd_data = prepd_data.drop(
        columns=['vix_close', 'vix9d_close', 'vix3m_close', 'vix6m_close', 'vix_to_vix3m']
    )

    # NYC weather (high and low temperature and precipitation)
    weather = weather.rename(columns={'date': 'Date'})
    prepd_data = prepd_data.merge(weather, on='Date', how='left')
    # Drop redundant weather features (VIF: high_temp_nyc=56, low_temp_nyc=28) —
    # all three weather columns are seasonal proxies; sunlight_nyc is the primary signal.
    # See "Multicollinearity Audit" in Exploratory data analysis.ipynb.
    prepd_data = prepd_data.drop(columns=['high_temp_nyc', 'low_temp_nyc'])

    # Google Trends — same availability lag as Wikipedia pageviews. Daily/weekly GT data
    # is not available in real-time, so we shift the date forward by 1 day before merging.
    # Columns are renamed from 'index_<term>' to 'gt_prev_day_<term>' to make this explicit.
    gt_adjusted_pivot = gt_adjusted.pivot(index='date', columns='search_term',values=['index'])

    gt_adjusted_pivot.columns = ['_'.join(col).strip() for col in gt_adjusted_pivot.columns.values]
    gt_adjusted_pivot = gt_adjusted_pivot.reset_index().rename_axis(None, axis=1)
    gt_adjusted_pivot = gt_adjusted_pivot.rename(columns={'date': 'Date'})
    gt_adjusted_pivot.columns = [
        col.replace('index_', 'gt_prev_day_', 1) if col.startswith('index_') else col
        for col in gt_adjusted_pivot.columns
    ]
    gt_adjusted_pivot['Date'] = gt_adjusted_pivot['Date'] + pd.Timedelta(days=1)

    prepd_data = prepd_data.merge(gt_adjusted_pivot,on='Date',how='left')

    # Check for missing or duplicate dates after merging
    if prepd_data['Date'].isna().any():
        logging.warning("Missing dates in prepd_data after merging.")
    if prepd_data['Date'].duplicated().any():
        logging.warning("Duplicate dates in prepd_data after merging.")

    # Streaks
    prepd_data['yesterday_to_today'] = np.where(
        (prepd_data[target_ticker] - prepd_data[target_ticker].shift(1)) < 0, 0, 1
    )

    # Calculate the length of consecutive streaks of up or down days
    prepd_data['streak'] = prepd_data.groupby(
        (prepd_data['yesterday_to_today'] != prepd_data['yesterday_to_today'].shift(1)).cumsum()
    ).cumcount()+1

    # streak0 = consecutive down-day count; streak1 = consecutive up-day count
    prepd_data['streak0'] = np.where(prepd_data['yesterday_to_today']==1,0,prepd_data['streak'])
    prepd_data['streak1'] = np.where(prepd_data['yesterday_to_today']==0,0,prepd_data['streak'])

    prepd_data = prepd_data.drop(columns=['yesterday_to_today','streak'])

    prepd_data = calculate_technical_indicators(prepd_data, config)

    prepd_data['Daily_Return'] = prepd_data[target_ticker].pct_change()

    if target_ticker != 'Adj Close_SPY':
        prepd_data['Daily_Return_SPY'] = prepd_data['Adj Close_SPY'].pct_change()

    # Day of week (disabled by default — see IndicatorConfig.include_day_of_week)
    if config.include_day_of_week:
        prepd_data['day_of_week_name'] = prepd_data['Date'].dt.day_name()
        prepd_data = pd.get_dummies(prepd_data, columns=['day_of_week_name'],
                                    drop_first=True, dtype=int)

    # Calculate Target column
    prepd_data = prepd_data.sort_values(by='Date').reset_index(drop=True)
    prepd_data['Target'] = np.where(
        (prepd_data[target_ticker].shift(-1) - prepd_data[target_ticker]) < 0, 0, 1
    )

    # Drop raw price levels (VIF: target ~260, Adj Close_SPY ~168).
    # Relative position is captured by price_vs_ma_s, price_vs_vwap, bollinger_pct_b,
    # breakout_pct. Daily_Return and Target (computed above) are the only downstream uses.
    # See "Multicollinearity Audit" in Exploratory data analysis.ipynb.
    adj_close_cols = [c for c in prepd_data.columns if c.startswith('Adj Close_')]
    prepd_data = prepd_data.drop(columns=adj_close_cols)

    return prepd_data


def resample_to_weekly(daily_df: pd.DataFrame, return_col: str = 'Daily_Return') -> pd.DataFrame:
    """
    Resample a daily prep_data() output to weekly bars (last trading day of each
    calendar week, typically Friday).

    Most features take their end-of-week value. Daily_Return / Daily_Return_SPY
    are replaced with the cumulative return for the week. streak0/streak1 and
    Target are recomputed on weekly bars.

    The input daily_df is not modified.

    Parameters:
        daily_df: Output of prep_data().
        return_col: Column used to determine weekly direction (default 'Daily_Return').

    Returns:
        DataFrame with the same columns as daily_df, resampled to weekly frequency.
    """
    df = daily_df.copy().set_index('Date').sort_index()

    return_cols = [c for c in ['Daily_Return', 'Daily_Return_SPY'] if c in df.columns]
    skip_cols = return_cols + ['streak0', 'streak1', 'Target']
    other_cols = [c for c in df.columns if c not in skip_cols]

    # End-of-week value for all non-return features
    weekly = df[other_cols].resample('W-FRI').last()

    # Cumulative return for the week: (1+r1)(1+r2)...(1+rN) - 1
    # fillna(0) treats the first row's NaN return as 0% (no change)
    for col in return_cols:
        weekly[col] = (1 + df[col].fillna(0)).resample('W-FRI').prod() - 1

    # Drop empty weeks (e.g. full-week market holidays)
    weekly = weekly.dropna(how='all')

    # Recompute streaks on weekly bars (same logic as daily streaks in prep_data)
    direction = (weekly[return_col] >= 0).astype(int)
    streak = weekly.groupby(
        (direction != direction.shift(1)).cumsum()
    ).cumcount() + 1
    weekly['streak0'] = np.where(direction == 1, 0, streak)
    weekly['streak1'] = np.where(direction == 0, 0, streak)

    # Recompute Target: 1 if next week's return > 0
    weekly['Target'] = (weekly[return_col].shift(-1) > 0).astype(int)
    weekly.loc[weekly.index[-1], 'Target'] = pd.NA
    weekly = weekly.dropna(subset=['Target'])
    weekly['Target'] = weekly['Target'].astype(int)

    return weekly.reset_index()
