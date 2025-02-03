"""This module builds stocks_w"""

import glob
import os

import pandas as pd
import pytz
from astral import LocationInfo
from astral.sun import sun


def gen_stocks_w(ticker, drop_tickers=None):
    """
    Generates stocks_w dataframe

    Parameters:
        ticker (str): Stock ticker
        drop_tickers (binary): Whether to drop other tickers

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

    ffr_files = glob.glob('ffr_*.csv')
    ffr_latest = max(ffr_files, key=os.path.getctime)
    ffr_raw = pd.read_csv(ffr_latest, parse_dates=['Date'])

    weather_files = glob.glob('weather_*.csv')
    weather_latest = max(weather_files, key=os.path.getctime)
    weather_raw = pd.read_csv(weather_latest, parse_dates=['date'])

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
        'SPY',
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
    stocks_df['movement'] = stocks_df['Adj Close']*stocks_df['Volume'] # should be diff from yesterday * volume

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

    # Add some more features
    # Sunlight
    nyc = LocationInfo("New York City", "USA", "America/New_York", 40.7128, -74.0060)
    nyc_tz = pytz.timezone("America/New_York")

    stocks_w['sunrise_nyc'] = stocks_w['Date'].apply(
        lambda d: sun(nyc.observer,date=d, tzinfo=nyc_tz)['sunrise']
    )
    stocks_w['sunset_nyc'] = stocks_w['Date'].apply(
        lambda d: sun(nyc.observer, date=d, tzinfo=nyc_tz)['sunset']
    )
    stocks_w['sunlight_nyc'] = (stocks_w['sunset_nyc']-stocks_w['sunrise_nyc']).dt.total_seconds()
    stocks_w = stocks_w.drop(columns=['sunrise_nyc','sunset_nyc'])

    # Federal funds rate
    stocks_w = stocks_w.merge(ffr_raw,on='Date',how='left')

    # NYC weather (high and low temperature and precipitation
    weather = weather_raw.rename(columns={'date': 'Date'})
    stocks_w = stocks_w.merge(weather,on='Date',how='left')

    return stocks_w
