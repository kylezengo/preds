"""This module builds prepd_data"""

import glob
import os
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
    stocks_df_files = glob.glob('stocks_df_*.csv')
    stocks_df_latest = max(stocks_df_files, key=os.path.getctime)
    stocks_df_raw = pd.read_csv(stocks_df_latest, parse_dates=['Date'])

    wiki_pageviews_files = glob.glob('wiki_pageviews_*.csv')
    wiki_pageviews_latest = max(wiki_pageviews_files, key=os.path.getctime)
    wiki_pageviews = pd.read_csv(wiki_pageviews_latest, parse_dates=['Date'])

    ffr_files = glob.glob('ffr_*.csv')
    ffr_latest = max(ffr_files, key=os.path.getctime)
    ffr = pd.read_csv(ffr_latest, parse_dates=['Date'])

    weather_files = glob.glob('weather_*.csv')
    weather_latest = max(weather_files, key=os.path.getctime)
    weather = pd.read_csv(weather_latest, parse_dates=['date'])

    gt_adjusted_files = glob.glob('gt_adjusted_*.csv')
    gt_adjusted_latest = max(gt_adjusted_files, key=os.path.getctime)
    gt_adjusted = pd.read_csv(gt_adjusted_latest, parse_dates=['date'])

    return stocks_df_raw, wiki_pageviews, ffr, weather, gt_adjusted

# Technical indicators
def calculate_rsi_wide(data, target, ticker, window):
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

def calculate_rsi_long(data, target, window):
    """
    Calculate the Relative Strength Index (RSI).
    
    Parameters:
        data (DataFrame): Stock data with target prices.
        target (str): column to predict (usually Adj Close)
        window (int): Lookback period for RSI.
        
    Returns:
        Series: RSI values.
    """
    delta = data[target].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))

def calculate_vwap_wide(data, target, ticker):
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

def calculate_vwap_long(data, target):
    """
    Calculate Volume Weighted Average Price

    Parameters:
        data (DataFrame): Stock data with required columns.
        target (str): column to predict (usually Adj Close)

    Returns:
       
    """
    cumulative_volume = data['Volume'].cumsum()
    cumulative_price_volume = (data[target] * data['Volume']).cumsum()
    return cumulative_price_volume / cumulative_volume

def calculate_macd(data, short_period=12, long_period=26):
    """
    Calculate the MACD line (moving average convergence/divergence).
    
    Parameters:
        data (DataFrame): Must contain a 'Close' price column.
        short_period (int): Short-term EMA period (default=12).
        long_period (int): Long-term EMA period (default=26).
    
    Returns:
        Series: The MACD line.
    """
    short_ema = data['Close'].ewm(span=short_period, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_period, adjust=False).mean()
    macd_line = short_ema - long_ema
    return macd_line

def calculate_technical_indicators(data, config: IndicatorConfig):
    """
    Calculate technical indicators for the dataset (uses wide functions).

    Parameters:
        data (DataFrame): Stock data with required columns.
        config (IndicatorConfig): Configuration object for technical indicators.

    Returns:
        DataFrame: Original dataframe with technical indicators included 
    """
    target_ticker = config.target+"_"+config.ticker
    data['RSI'] = calculate_rsi_wide(data, config.target, config.ticker, window=config.rsi_window)
    data['MA_S'] = data[target_ticker].rolling(window=config.moving_average.short_window).mean()
    data['MA_L'] = data[target_ticker].rolling(window=config.moving_average.long_window).mean()
    data['MA_B'] = data[target_ticker].rolling(window=config.bollinger.window).mean()
    data['Bollinger_Upper'] = (data['MA_B'] +
                               config.bollinger.num_std *
                               data[target_ticker].rolling(window=config.bollinger.window).std())
    data['Bollinger_Lower'] = (data['MA_B'] -
                               config.bollinger.num_std *
                               data[target_ticker].rolling(window=config.bollinger.window).std())
    data['VWAP'] = calculate_vwap_wide(data, config.target, config.ticker)

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

    stocks_df = stocks_df.merge(
        wiki_pageviews[['Date','ticker','views']],how='left', on=['Date','ticker']
    )

    # Some companies have calss A, B, etc. stock. See sp_df for details. Choosing class A for now.
    # Maybe keep in the future? Trying to reduce multicolinearity for now
    # Alphabet: GOOGL, GOOG; Fox: FOX, FOXA; News Corp: NWS, NWSA
    stocks_df = stocks_df[~stocks_df['ticker'].isin(['GOOG', 'FOX', 'NWS'])].reset_index(drop=True)

    # Pivot
    stocks_w = stocks_df.pivot(
        index='Date',
        columns='ticker',
        values=['Open','High','Low','Close','Adj Close','Volume','movement','views']
    )

    stocks_w.columns = ['_'.join(col).strip() for col in stocks_w.columns.values]
    stocks_w = stocks_w.reset_index().rename_axis(None, axis=1)

    return stocks_w

def prep_data(stocks_df, wiki_pageviews, ffr_raw, weather, gt_adjusted, config: IndicatorConfig,
              drop_tickers=None):
    """
    Prepare data for forecasting strategies (add some extra features).

    Parameters:
        config (IndicatorConfig): Configuration object for technical indicators.
        drop_tickers (bool): Whether to drop other tickers.

    Returns:
        DataFrame: Prepared data with additional features and technical indicators.
    """
    target_ticker = config.target+"_"+config.ticker

    prepd_data = gen_stocks_w(config.ticker, stocks_df, wiki_pageviews, drop_tickers)

    # Sunlight
    nyc = LocationInfo("New York City", "USA", "America/New_York", 40.7128, -74.0060)
    nyc_tz = pytz.timezone("America/New_York")

    prepd_data['sunlight_nyc'] = prepd_data['Date'].apply(
        lambda d: (sun(nyc.observer, date=d, tzinfo=nyc_tz)['sunset'] -
                   sun(nyc.observer, date=d, tzinfo=nyc_tz)['sunrise']).total_seconds()
    )

    # Federal funds rate
    prepd_data = prepd_data.merge(ffr_raw,on='Date',how='left')

    # NYC weather (high and low temperature and precipitation)
    weather = weather.rename(columns={'date': 'Date'})
    prepd_data = prepd_data.merge(weather,on='Date',how='left')

    # Google Trends
    gt_adjusted_pivot = gt_adjusted.pivot(index='date', columns='search_term',values=['index'])

    gt_adjusted_pivot.columns = ['_'.join(col).strip() for col in gt_adjusted_pivot.columns.values]
    gt_adjusted_pivot = gt_adjusted_pivot.reset_index().rename_axis(None, axis=1)
    gt_adjusted_pivot = gt_adjusted_pivot.rename(columns={'date': 'Date'})

    prepd_data = prepd_data.merge(gt_adjusted_pivot,on='Date',how='left')

    # Check for missing or duplicate dates after merging
    if prepd_data['Date'].isna().any():
        print("Warning: Missing dates in prepd_data after merging.")
    if prepd_data['Date'].duplicated().any():
        print("Warning: Duplicate dates in prepd_data after merging.")

    # Streaks
    prepd_data['yesterday_to_today'] = np.where(
        (prepd_data[target_ticker] - prepd_data[target_ticker].shift(1)) < 0, 0, 1
    )

    # Calculate the length of consecutive streaks of up or down days
    prepd_data['streak'] = prepd_data.groupby(
        (prepd_data['yesterday_to_today'] != prepd_data['yesterday_to_today'].shift(1)).cumsum()
    ).cumcount()+1

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
