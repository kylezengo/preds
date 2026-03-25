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

def calculate_vwap(data, target, ticker):
    """
    Calculate Volume Weighted Average Price

    Parameters:
        data (DataFrame): Stock data with required columns.
        target (str): column to predict (usually Adj Close)
        ticker (str): Stock ticker

    Returns:
        Series: VWAP values.
    """
    cumulative_volume = data["Volume_"+ticker].cumsum()
    cumulative_price_volume = (data[target+"_"+ticker] * data["Volume_"+ticker]).cumsum()
    return cumulative_price_volume / cumulative_volume

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
    data['RSI'] = calculate_rsi(data, config.target, config.ticker, window=config.rsi_window)
    data['MA_S'] = data[target_ticker].rolling(window=config.moving_average.short_window).mean()
    data['MA_L'] = data[target_ticker].rolling(window=config.moving_average.long_window).mean()
    data['MA_B'] = data[target_ticker].rolling(window=config.bollinger.window).mean()
    data['Bollinger_Upper'] = (data['MA_B'] +
                               config.bollinger.num_std *
                               data[target_ticker].rolling(window=config.bollinger.window).std())
    data['Bollinger_Lower'] = (data['MA_B'] -
                               config.bollinger.num_std *
                               data[target_ticker].rolling(window=config.bollinger.window).std())
    data['VWAP'] = calculate_vwap(data, config.target, config.ticker)

    data['short_ema'] = data[target_ticker].ewm(span=config.macd.short_window, adjust=False).mean()
    data['long_ema'] = data[target_ticker].ewm(span=config.macd.long_window, adjust=False).mean()
    data['macd_line'] = data['short_ema'] - data['long_ema']

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
        prepd_data = prepd_data.merge(etf_returns[['Date', 'sector_etf_return_prev_day']], on='Date', how='left')
        # sector vs SPY: use Adj Close_SPY already in prepd_data as the broad market benchmark
        prepd_data['sector_vs_spy_return'] = (
            prepd_data['sector_etf_return_prev_day'] - prepd_data['Adj Close_SPY'].pct_change().shift(1)
        )

    # VIX and treasury yields
    prepd_data = prepd_data.merge(vix_yields, on='Date', how='left')
    prepd_data['vix_change_prev_day'] = prepd_data['vix_close'].pct_change().shift(1)
    prepd_data['vix_pct_rank_252'] = (
        prepd_data['vix_close'].rolling(252).rank(pct=True)
    )

    # NYC weather (high and low temperature and precipitation)
    weather = weather.rename(columns={'date': 'Date'})
    prepd_data = prepd_data.merge(weather,on='Date',how='left')

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

    # Day of week
    prepd_data['day_of_week_name'] = prepd_data['Date'].dt.day_name()

    prepd_data = pd.get_dummies(prepd_data, columns=['day_of_week_name'],
                                drop_first=True, dtype=int)

    # Calculate Target column
    prepd_data = prepd_data.sort_values(by='Date').reset_index(drop=True)
    prepd_data['Target'] = np.where(
        (prepd_data[target_ticker].shift(-1) - prepd_data[target_ticker]) < 0, 0, 1
    )

    return prepd_data
