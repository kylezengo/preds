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
    Calculate the MACD line.
    
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
def gen_stocks_w(ticker, drop_tickers=None):
    """
    Generates stocks_w dataframe

    Parameters:
        ticker (str): Stock ticker
        drop_tickers (bool): Whether to drop other tickers

    Returns:
        DataFrame: stocks_w dataframe
    """
    # Load data
    stocks_df_files = glob.glob('stocks_df_*.csv')
    stocks_df_latest = max(stocks_df_files, key=os.path.getctime)
    stocks_df_raw = pd.read_csv(stocks_df_latest, parse_dates=['Date'])

    wiki_pageviews_files = glob.glob('wiki_pageviews_*.csv')
    wiki_pageviews_latest = max(wiki_pageviews_files, key=os.path.getctime)
    wiki_pageviews_raw = pd.read_csv(wiki_pageviews_latest, parse_dates=['Date'])

    #
    stocks_df = stocks_df_raw.copy()

    tickers_list = [
        'A','AAPL','ABBV','ABNB','ABT','ACGL','ACN','ADBE','ADI','ADM','ADP','ADSK','AEE','AEP',
        'AES','AFL','AIG','AIZ','AJG','AKAM','ALB','ALGN','ALL','ALLE','AMAT','AMCR','AMD','AME',
        'AMGN','AMP','AMT','AMZN','ANET','ANSS','AON','AOS','APA','APD','APH','APO','APTV','ARE',
        'ATO','AVB','AVGO','AVY','AWK','AXON','AXP','AZO','BA','BAC','BALL','BAX','BBY','BDX',
        'BEN','BG','BIIB','BK','BKNG','BKR','BLDR','BLK','BMY','BR','BRO','BSX','BWA','BX','BXP',
        'C','CAG','CAH','CARR','CAT','CB','CBOE','CBRE','CCI','CCL','CDNS','CDW','CE','CEG','CF',
        'CFG','CHD','CHRW','CHTR','CI','CINF','CL','CLX','CMCSA','CME','CMG','CMI','CMS','CNC',
        'CNP','COF','COO','COP','COR','COST','CPAY','CPB','CPRT','CPT','CRL','CRM','CRWD','CSCO',
        'CSGP','CSX','CTAS','CTRA','CTSH','CTVA','CVS','CVX','CZR','D','DAL','DAY','DD','DE',
        'DECK','DELL','DFS','DG','DGX','DHI','DHR','DIS','DLR','DLTR','DOC','DOV','DOW','DPZ',
        'DRI','DTE','DUK','DVA','DVN','DXCM','EA','EBAY','ECL','ED','EFX','EG','EIX','EL','ELV',
        'EMN','EMR','ENPH','EOG','EPAM','EQIX','EQR','EQT','ERIE','ES','ESS','ETN','ETR','EVRG',
        'EW','EXC','EXPD','EXPE','EXR','F','FANG','FAST','FCX','FDS','FDX','FE','FFIV','FI','FICO',
        'FIS','FITB','FMC',
        'FOX','FOXA',
        'FRT','FSLR','FTNT','FTV','GD','GDDY','GE','GEHC','GEN','GEV','GILD','GIS','GL','GLW','GM',
        'GNRC',
        'GOOG','GOOGL',
        'GPC','GPN','GRMN','GS','GWW','HAL','HAS','HBAN','HCA','HD','HES','HIG','HII','HLT','HOLX',
        'HON','HPE','HPQ','HRL','HSIC','HST','HSY','HUBB','HUM','HWM','IBM','ICE','IDXX','IEX',
        'IFF','INCY','INTC','INTU','INVH','IP','IPG','IQV','IR','IRM','ISRG','IT','ITW','IVZ','J',
        'JBHT','JBL','JCI','JKHY','JNJ','JNPR','JPM','K','KDP','KEY','KEYS','KHC','KIM','KKR',
        'KLAC','KMB','KMI','KMX','KO','KR','KVUE','L','LDOS','LEN','LH','LHX','LII','LIN','LKQ',
        'LLY','LMT','LNT','LOW','LRCX','LULU','LUV','LVS','LW','LYB','LYV','MA','MAA','MAR','MAS',
        'MCD','MCHP','MCK','MCO','MDLZ','MDT','MET','META','MGM','MHK','MKC','MKTX','MLM','MMC',
        'MMM','MNST','MO','MOH','MOS','MPC','MPWR','MRK','MRNA','MS','MSCI','MSFT','MSI','MTB',
        'MTCH','MTD','MU','NCLH','NDAQ','NDSN','NEE','NEM','NFLX','NI','NKE','NOC','NOW','NRG',
        'NSC','NTAP','NTRS','NUE','NVDA','NVR',
        'NWS','NWSA',
        'NXPI','O','ODFL','OKE','OMC','ON','ORCL','ORLY','OTIS','OXY','PANW','PARA','PAYC','PAYX',
        'PCAR','PCG','PEG','PEP','PFE','PFG','PG','PGR','PH','PHM','PKG','PLD','PLTR','PM','PNC',
        'PNR','PNW','PODD','POOL','PPG','PPL','PRU','PSA','PSX','PTC','PWR','PYPL','QCOM','RCL',
        'REG','REGN','RF','RJF','RL','RMD','ROK','ROL','ROP','ROST','RSG','RTX','RVTY','SBAC',
        'SBUX','SCHW','SHW','SJM','SLB','SMCI','SNA','SNPS','SO','SOLV','SPG','SPGI',
        # 'SPY',
        'SRE','STE','STLD','STT','STX','STZ','SW','SWK','SWKS','SYF','SYK','SYY','T','TAP','TDG',
        'TDY','TECH','TEL','TER','TFC','TFX','TGT','TJX','TMO','TMUS','TPL','TPR','TRGP','TRMB',
        'TROW','TRV','TSCO','TSLA','TSN','TT','TTWO','TXN','TXT','TYL','UAL','UBER','UDR','UHS',
        'ULTA','UNH','UNP','UPS','URI','USB','V','VICI','VLO','VLTO','VMC','VRSK','VRSN','VRTX',
        'VST','VTR','VTRS','VZ','WAB','WAT','WBA','WBD','WDAY','WDC','WEC','WELL','WFC','WM','WMB',
        'WMT','WRB','WST','WTW','WY','WYNN','XEL','XOM','XYL','YUM','ZBH','ZBRA','ZTS'
    ]

    drop_tickers_list = [x for x in tickers_list if x!= ticker]

    # Can drop other ticker columns for testing (so runs quicker)
    if drop_tickers:
        stocks_df = stocks_df.loc[~stocks_df['ticker'].isin(drop_tickers_list)]


    # Set up data frame for testing
    stocks_df['movement'] = stocks_df.groupby('ticker')['Adj Close'].diff() * stocks_df['Volume']

    stocks_df = stocks_df.merge(
        wiki_pageviews_raw[['Date','ticker','views']],how='left', on=['Date','ticker']
    )

    # Some companies have calss A, B, etc. stock. See sp_df for details. Choosing class A for now.
    # Maybe keep in the future? Trying to reduce multicolinearity for now
    # Alphabet: GOOGL, GOOG; Fox: FOX, FOXA; News Corp: NWS, NWSA
    stocks_df = stocks_df[~stocks_df['ticker'].isin(['GOOG', 'FOX', 'NWS'])].reset_index(drop=True)

    # Pivot
    stocks_w = stocks_df.pivot(
        index='Date', columns='ticker',
        values=['Open','High','Low','Close','Adj Close','Volume','movement','views']
    )

    stocks_w.columns = ['_'.join(col).strip() for col in stocks_w.columns.values]
    stocks_w = stocks_w.reset_index().rename_axis(None, axis=1)

    return stocks_w

def prep_data(config: IndicatorConfig, drop_tickers=None):
    """
    Prepare data for forecasting strategies (add some extra features).

    Parameters:
        config (IndicatorConfig): Configuration object for technical indicators.
        drop_tickers (bool): Whether to drop other tickers.

    Returns:
        DataFrame: Prepared data with additional features and technical indicators.
    """
    ffr_files = glob.glob('ffr_*.csv')
    ffr_latest = max(ffr_files, key=os.path.getctime)
    ffr_raw = pd.read_csv(ffr_latest, parse_dates=['Date'])

    weather_files = glob.glob('weather_*.csv')
    weather_latest = max(weather_files, key=os.path.getctime)
    weather_raw = pd.read_csv(weather_latest, parse_dates=['date'])

    gt_adjusted_files = glob.glob('gt_adjusted_*.csv')
    gt_adjusted_latest = max(gt_adjusted_files, key=os.path.getctime)
    gt_adjusted_raw = pd.read_csv(gt_adjusted_latest, parse_dates=['date'])

    prepd_data = gen_stocks_w(config.ticker, drop_tickers)

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
    weather = weather_raw.rename(columns={'date': 'Date'})
    prepd_data = prepd_data.merge(weather,on='Date',how='left')

    # Google Trends
    gt_adjusted_pivot = gt_adjusted_raw.pivot(index='date', columns='search_term',values=['index'])

    gt_adjusted_pivot.columns = ['_'.join(col).strip() for col in gt_adjusted_pivot.columns.values]
    gt_adjusted_pivot = gt_adjusted_pivot.reset_index().rename_axis(None, axis=1)
    gt_adjusted_pivot = gt_adjusted_pivot.rename(columns={'date': 'Date'})

    prepd_data = prepd_data.merge(gt_adjusted_pivot,on='Date',how='left')

    #
    target_ticker = config.target+"_"+config.ticker

    # Streaks
    prepd_data['yesterday_to_today'] = np.where(
        (prepd_data[target_ticker]-prepd_data[target_ticker].shift(1)) < 0, 0, 1
    )

    # Calculate the length of consecutive streaks of up or down days
    prepd_data['streak'] = prepd_data.groupby(
        (prepd_data['yesterday_to_today'] != prepd_data['yesterday_to_today'].shift(1)).cumsum()
    ).cumcount()+1

    prepd_data['streak0'] = np.where(prepd_data['yesterday_to_today']==1,0,prepd_data['streak'])
    prepd_data['streak1'] = np.where(prepd_data['yesterday_to_today']==0,0,prepd_data['streak'])

    prepd_data = prepd_data.drop(columns=['yesterday_to_today','streak'])

    # Must be inside the strategies ##################################
    # data['next_is_0'] = (data['yesterday_to_today'].shift(-1) == 0).astype(int) # leak
    # data['next_is_1'] = (data['yesterday_to_today'].shift(-1) == 1).astype(int)

    # # prob of a 0 or 1 following a streak length
    # prob_df_0 = data.groupby('streak0')['next_is_0'].mean().to_frame() # leak?
    # prob_df_0 = prob_df_0.rename(columns={'next_is_0': 'prob_next_is_0'})
    # prob_df_1 = data.groupby('streak1')['next_is_1'].mean().to_frame()
    # prob_df_1 = prob_df_1.rename(columns={'next_is_1': 'prob_next_is_1'})

    # # Merge probabilities back into original DataFrame
    # data = data.merge(prob_df_0, how='left', left_on='streak0', right_index=True)
    # data = data.merge(prob_df_1, how='left', left_on='streak1', right_index=True)
    ##################################################################

    prepd_data = calculate_technical_indicators(prepd_data, config)

    prepd_data['Daily_Return'] = prepd_data[target_ticker].pct_change()

    if target_ticker != 'Adj Close_SPY':
        prepd_data['Daily_Return_SPY'] = prepd_data['Adj Close_SPY'].pct_change()

    # Day of week
    prepd_data['day_of_week_name'] = prepd_data['Date'].dt.day_name()

    prepd_data = pd.get_dummies(prepd_data, columns=['day_of_week_name'],
                                drop_first=True, dtype=int)

    # Target
    prepd_data['Target'] = np.where(
        (prepd_data[target_ticker].shift(-1)-prepd_data[target_ticker]) < 0, 0, 1
    )

    return prepd_data
