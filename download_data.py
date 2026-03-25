"""This module downloads the data necessary to test different strategies"""

import glob
import logging
import os
import time
import urllib.parse
from datetime import datetime, date, timedelta

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Authentication
load_dotenv()
fred_api_key = os.getenv("fred_api_key")
noaa_api_key = os.getenv("noaa_api_key")
wiki_user_agent = os.getenv("wiki_user_agent")

WIKI_SDATE = "20150701" # earliest date is"20150701"
WIKI_BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
START_DATE = "1993-01-29" # SPY launched on 1993-01-22 ... first data is January 29?
TODAY_ISO_STR = datetime.today().strftime('%Y-%m-%d')

NOAA_BASE_URL = 'https://www.ncei.noaa.gov/cdo-web/api/v2/data'
OBSERVATIONS_URL = "https://api.weather.gov/stations/KNYC/observations"
STATIONID = "GHCND:USW00094728" # Central Park Station in NYC


def load_existing_weather_data():
    """
    Load existing weather data from CSV files.
    """
    weather_df_files = glob.glob('data/weather_df_*.csv')
    if not weather_df_files:
        raise FileNotFoundError("No existing weather data files found in data/weather_df_*.csv")
    weather_df_latest = max(weather_df_files, key=lambda f: f.split("_")[-1])
    weather_df_loaded = pd.read_csv(weather_df_latest, parse_dates=['date'])

    return weather_df_loaded


def get_federal_funds_rate():
    """
    Get federal funds rate using the Fred API (not exact match to current numbers?)
    """
    fred = Fred(api_key=fred_api_key)
    build_ffr = fred.get_series('FEDFUNDS').to_frame(name='federal_funds_rate')
    build_ffr.loc[TODAY_ISO_STR, 'federal_funds_rate'] = build_ffr['federal_funds_rate'].iloc[-1]
    build_ffr = build_ffr.resample('D').ffill()
    build_ffr = build_ffr.reset_index(names='Date')
    return build_ffr


SECTOR_ETF_MAP = {
    'Information Technology': 'XLK',
    'Health Care': 'XLV',
    'Financials': 'XLF',
    'Energy': 'XLE',
    'Industrials': 'XLI',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Utilities': 'XLU',
    'Communication Services': 'XLC',
}


def get_sector_etf_data():
    """
    Get daily price data for the 11 SPDR sector ETFs.
    Returns a DataFrame with columns: Date, and one 'close_<ETF>' column per sector.
    Note: XLRE and XLC only go back to ~2015, all others to ~1999.
    """
    etf_frames = []
    for etf in SECTOR_ETF_MAP.values():
        data = yf.download(etf, start=START_DATE, end=TODAY_ISO_STR, auto_adjust=False, progress=False)
        data.columns = data.columns.get_level_values(0)
        etf_frames.append(data[['Close']].rename(columns={'Close': f'close_{etf}'}))

    result = etf_frames[0].join(etf_frames[1:], how='outer')
    result = result.resample('D').ffill()
    result = result.reset_index(names='Date')
    return result


def get_vix_and_yields():
    """
    Get VIX (CBOE Volatility Index), VIX term structure, and Treasury yield data (10Y, 2Y).
    VIX data is fetched via yfinance; yields via FRED.
    Term structure tickers (VIX9D, VIX3M, VIX6M) are available from ~2015-01-01.
    """
    # VIX and term structure
    vix_tickers = {'^VIX': 'vix_close', '^VIX9D': 'vix9d_close', '^VIX3M': 'vix3m_close', '^VIX6M': 'vix6m_close'}
    vix_frames = []
    for ticker, col_name in vix_tickers.items():
        raw = yf.download(ticker, start=START_DATE, end=TODAY_ISO_STR, auto_adjust=False, progress=False)
        raw.columns = raw.columns.get_level_values(0)
        vix_frames.append(raw[['Close']].rename(columns={'Close': col_name}))

    vix = vix_frames[0].join(vix_frames[1:], how='outer').resample('D').ffill()

    # Treasury yields from FRED
    fred = Fred(api_key=fred_api_key)
    dgs10 = fred.get_series('DGS10').to_frame(name='yield_10y')
    dgs2 = fred.get_series('DGS2').to_frame(name='yield_2y')

    yields = dgs10.join(dgs2, how='outer')
    # Pin today's date to the last known value (same pattern as get_federal_funds_rate)
    # so ffill covers up to today without bfill filling early dates from the future.
    yields.loc[TODAY_ISO_STR, 'yield_10y'] = yields['yield_10y'].dropna().iloc[-1]
    yields.loc[TODAY_ISO_STR, 'yield_2y'] = yields['yield_2y'].dropna().iloc[-1]
    yields = yields.resample('D').ffill()

    result = vix.join(yields, how='outer')
    result['yield_spread'] = result['yield_10y'] - result['yield_2y']
    # VIX term structure slope: ratio > 1 = contango (calm), < 1 = backwardation (fear)
    result['vix9d_to_vix'] = result['vix9d_close'] / result['vix_close']
    result['vix_to_vix3m'] = result['vix_close'] / result['vix3m_close']
    result = result.reset_index(names='Date')
    return result


def get_sp500_tickers():
    """
    Get list of S&P 500 tickers from Wikipedia page
    """
    wiki_headers = {
        "User-Agent": wiki_user_agent
    }

    sp_wiki = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=wiki_headers, timeout=20)

    build_sp_df = pd.read_html(sp_wiki.content)[0]

    # Get the wikipedia page for each company (part of url)
    build_sp_df['wiki_page'] = build_sp_df['Security'].apply(urllib.parse.quote_plus)

    wiki_page_replacements = {
        "Alphabet+Inc.+%28Class+A%29": "Alphabet_Inc.",
        "Alphabet+Inc.+%28Class+C%29": "Alphabet_Inc.",
        'Coca-Cola+Company+%28The%29': "The_Coca-Cola_Company",
        'Cooper+Companies+%28The%29': "The_Cooper_Companies",
        'Est%C3%A9e+Lauder+Companies+%28The%29': "The_Estée_Lauder_Companies",
        'Fox+Corporation+%28Class+A%29': "Fox_Corporation",
        'Fox+Corporation+%28Class+B%29': "Fox_Corporation",
        'Hartford+%28The%29': "The_Hartford",
        'Hershey+Company+%28The%29': "The_Hershey_Company",
        'Home+Depot+%28The%29': "Home_Depot",
        'Interpublic+Group+of+Companies+%28The%29': "The_Interpublic_Group_of_Companies",
        'Lilly+%28Eli%29': "Eli_Lilly_and_Company",
        'Mosaic+Company+%28The%29': "The_Mosaic_Company",
        'News+Corp+%28Class+A%29': "News_Corp",
        'News+Corp+%28Class+B%29': "News_Corp",
        'PTC+Inc.': "PTC_(software_company)",
        'J.M.+Smucker+Company+%28The%29': "The_J.M._Smucker_Company",
        'Travelers+Companies+%28The%29': "The_Travelers_Companies",
        'Walt+Disney+Company+%28The%29': "The_Walt_Disney_Company",
        "O%E2%80%99Reilly+Automotive": "O'Reilly_Auto_Parts",
        "Campbell%27s+Company+%28The%29": "Campbell%27s",
        "Trade+Desk+%28The%29": "The_Trade_Desk"
    }

    ticker_replacements = {
        "BRK.B": "BRK-B",
        "BF.B": "BF-B"
    }

    for key, replacement_value in wiki_page_replacements.items():
        build_sp_df.loc[build_sp_df['wiki_page'] == key, 'wiki_page'] = replacement_value

    for key, replacement_value in ticker_replacements.items():
        build_sp_df.loc[build_sp_df['Symbol'] == key, 'Symbol'] = replacement_value

    build_sp_df_spy = pd.DataFrame([{
        'Symbol': "SPY",
        'Security': "S&P 500", # technically the SPDR S&P 500 ETF Trust
        'GICS Sector': None,
        'GICS Sub-Industry': None,
        'Headquarters Location': None,
        'Date added': "1993-01-22",
        'CIK': None,
        'Founded': "1993-01-22",
        'wiki_page': "S%26P_500"
    }])

    build_sp_df = pd.concat([build_sp_df,build_sp_df_spy],ignore_index=True)
    return build_sp_df


def get_wikipedia_pageviews(sp_df):
    """
    Get daily wikipedia pageviews for each company
    """
    wiki_edate=(date.today()-timedelta(days=1)).strftime('%Y%m%d') # yesterday

    wiki_headers = {
        "User-Agent": wiki_user_agent
    }

    dat=[]
    missing=[]
    for page in set(sp_df['wiki_page']):
        url = f"{WIKI_BASE_URL}{page}/daily/{WIKI_SDATE}/{wiki_edate}"

        try:
            page_response = requests.get(url, headers=wiki_headers, timeout=20)
            page_response.raise_for_status()

            json_data = page_response.json()
            if 'items' in json_data:
                items_df = pd.DataFrame(json_data['items'])

                if len(sp_df.loc[sp_df['wiki_page']==page,'Symbol']) > 1:
                    items_df['ticker'] = ','.join(
                        list(sp_df.loc[sp_df['wiki_page']==page,'Symbol'])
                    )
                else:
                    items_df['ticker'] = sp_df.loc[sp_df['wiki_page']==page,'Symbol'].item()
            else:
                logging.warning("'items' key missing in response for page: %s", page)
                missing.append(page)
                continue

        except requests.exceptions.RequestException as e:
            logging.warning("Request error for %s: %s", page, e)
            missing.append(page)
            continue

        except ValueError as e:
            logging.warning("ValueError for page %s: %s", page, e)
            missing.append(page)
            continue

        if len(items_df)==0:
            logging.warning("Wiki pageviews data frame empty: %s", page)
            missing.append(page)
            continue

        dat.append(items_df)

    if missing:
        logging.warning("Wikipedia pageviews missing for %d pages: %s", len(missing), missing)

    build_wiki_pv = pd.concat(dat).reset_index(drop=True)
    build_wiki_pv['Date'] =  pd.to_datetime(build_wiki_pv['timestamp'], format='%Y%m%d%H')
    return build_wiki_pv


def get_stocks_data(sp500_tickers):
    """
    Get daily stock price data for each S&P 500 company using yfinance
    """
    tickers_list = list(sp500_tickers)
    dat_list = []
    for idx, i in enumerate(tickers_list, start=1):
        logging.info("Stocks %d/%d: %s", idx, len(tickers_list), i)
        data = yf.download(
            i,
            start=START_DATE,
            end=TODAY_ISO_STR,
            auto_adjust=False, # ?
            progress=False
        )
        data.columns = data.columns.get_level_values(0)
        data['ticker'] = i
        dat_list.append(data)
        time.sleep(1)

    result = pd.concat(dat_list)
    result = result.reset_index()
    result = result.rename_axis(None, axis=1)
    return result


def get_outstanding_shares(sp500_tickers):
    """
    Get daily outstanding shares for each S&P 500 company using yfinance
    """
    dat_list = []
    for i in sp500_tickers:
        ticker = yf.Ticker(i)
        shares = ticker.get_shares_full(start=START_DATE, end=TODAY_ISO_STR)
        if shares is not None:
            df = pd.DataFrame(shares)
            df['ticker'] = i
            dat_list.append(df)

    os_df = pd.concat(dat_list)
    os_df = os_df.rename_axis('os_report_datetime').reset_index()
    os_df['os_report_date'] = pd.to_datetime(os_df['os_report_datetime'].dt.date)
    os_df = os_df.rename(columns={0: 'outstanding_shares'})
    os_df = os_df.groupby(
        ['os_report_datetime','os_report_date','ticker']
    ).agg(outstanding_shares = ('outstanding_shares','mean')).reset_index()

    # Group to dates
    # is mean correct? or should I take last value? only matters if there are duplicate dates above
    os_df_date_tick = os_df.groupby(
        ['os_report_date','ticker']
    ).agg(outstanding_shares = ('outstanding_shares','mean')).reset_index()

    # Set outstanding shares for each day using ffill within each ticker
    all_dates = pd.date_range(start=os_df_date_tick['os_report_date'].min(), end=TODAY_ISO_STR, freq='D')
    tickers = os_df_date_tick['ticker'].unique()
    full_index = pd.MultiIndex.from_product([all_dates, tickers], names=['date', 'ticker'])

    result = (
        os_df_date_tick
        .rename(columns={'os_report_date': 'date'})
        .set_index(['date', 'ticker'])
        .reindex(full_index)
        .groupby(level='ticker')['outstanding_shares']
        .ffill()
        .reset_index()
    )
    return result


def fetch_noaa_with_retry(params, headers):
    """
    Fetch a single date range from the NOAA API with up to 3 attempts.
    Returns the response on success, or None if all attempts fail.
    """
    delays = [0, 10, 20]
    for attempt, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        resp = requests.get(NOAA_BASE_URL, headers=headers, params=params, timeout=300)
        if resp.status_code == 200:
            return resp
        logging.warning("Attempt %d failed for %s: %s %s", attempt, params['startdate'], resp.status_code, resp.text)
    logging.error("All %d attempts failed for %s. Skipping.", len(delays), params['startdate'])
    return None


def get_noaa_weather(last_hist_date):
    """
    Get historical weather data for NYC from NOAA.
    Only fetches date ranges after last_hist_date or within the past year.
    """
    end_date_dt = datetime.strptime(TODAY_ISO_STR, "%Y-%m-%d")

    current_start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
    date_ranges = []
    while current_start_date <= end_date_dt:
        current_end_date = min(current_start_date + timedelta(days=29), end_date_dt)
        date_ranges.append({
            'start_date': current_start_date.strftime("%Y-%m-%d"),
            'end_date': current_end_date.strftime("%Y-%m-%d")
        })
        current_start_date = current_end_date + timedelta(days=1)

    date_ranges_df = pd.DataFrame(date_ranges)
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    cutoff = min(one_year_ago, last_hist_date.strftime("%Y-%m-%d"))
    date_ranges_df = date_ranges_df.loc[date_ranges_df['end_date'] >= cutoff].reset_index(drop=True)

    headers = {'token': noaa_api_key}
    weather_data = {}

    for _, row in date_ranges_df.iterrows():
        params = {
            'datasetid': 'GHCND',
            'stationid': STATIONID,
            'startdate': row['start_date'],
            'enddate': row['end_date'],
            'units': 'metric',
            'limit': 1000
        }

        resp = fetch_noaa_with_retry(params, headers)
        if resp is None:
            continue

        logging.info("NOAA %s: %s", row['start_date'], resp.status_code)
        data = resp.json()

        if 'results' not in data:
            logging.warning("No 'results' key in response for start date %s: %s", row['start_date'], data)
            continue

        for item in data['results']:
            item_date = item['date']
            datatype = item['datatype']
            value = item['value']

            if item_date not in weather_data:
                weather_data[item_date] = {
                    'date': item_date,
                    'high_temp_nyc': None,
                    'low_temp_nyc': None,
                    'precipitation_PRCP_nyc': None,
                    'precipitation_SNOW_nyc': None
                }

            if datatype == 'TMAX':
                weather_data[item_date]['high_temp_nyc'] = value
            elif datatype == 'TMIN':
                weather_data[item_date]['low_temp_nyc'] = value
            elif datatype == 'PRCP':
                weather_data[item_date]['precipitation_PRCP_nyc'] = value
            elif datatype == 'SNOW':
                weather_data[item_date]['precipitation_SNOW_nyc'] = value
        time.sleep(1)

    noaa_weather = pd.DataFrame(list(weather_data.values()))
    noaa_weather['date'] = pd.to_datetime(noaa_weather['date'])
    return noaa_weather


def get_recent_weather():
    """
    Get recent weather observations for NYC from weather.gov.
    Returns a DataFrame with daily min/max temperature.
    """
    observations_response = requests.get(OBSERVATIONS_URL, timeout=300)
    observations_response.raise_for_status()
    observations = observations_response.json()

    recent_weather_data = []
    for obs in observations["features"]:
        props = obs["properties"]
        recent_weather_data.append({
            '@id': props["@id"],
            'timestamp': props["timestamp"],
            'temperature': props.get("temperature", {}).get("value"),
            'minTemperatureLast24Hours': props.get("minTemperatureLast24Hours", {}).get("value"),
            'maxTemperatureLast24Hours': props.get("maxTemperatureLast24Hours", {}).get("value"),
            'windSpeed': props.get("windSpeed", {}).get("value"),  # In km_h-1
            'precipitationLastHour': props.get("precipitationLastHour", {}).get("value"),  # In mm
            'precipitationLast3Hours': props.get("precipitationLast3Hours", {}).get("value"),  # In mm
            'precipitationLast6Hours': props.get("precipitationLast6Hours", {}).get("value"),  # In mm
        })

    recent_weather_df = pd.DataFrame(recent_weather_data)
    recent_weather_df['timestamp'] = pd.to_datetime(recent_weather_df['timestamp'])
    recent_weather_df['date'] = pd.to_datetime(recent_weather_df['timestamp'].dt.date)

    weather_simp = recent_weather_df[['date', 'timestamp', 'temperature']]
    weather_simp = weather_simp.groupby('date').agg(
        low_temp_nyc=('temperature', 'min'),
        high_temp_nyc=('temperature', 'max')
    ).reset_index()
    return weather_simp


def get_weather_data():
    """
    Get historical and recent weather data for NYC, merged into a single DataFrame.
    """
    weather_df_hist = load_existing_weather_data()
    noaa_weather = get_noaa_weather(last_hist_date=max(weather_df_hist['date']))
    recent_weather = get_recent_weather()

    just_new = recent_weather.loc[recent_weather['date'] > max(noaa_weather['date'])]

    # Putting it all together
    weather_df_hist = weather_df_hist.loc[weather_df_hist['date'] < min(noaa_weather['date'])]
    result = pd.concat([weather_df_hist, noaa_weather, just_new]).reset_index(drop=True)
    return result


def save_data_to_csv(dic_of_dfs):
    """Save data to CSV files"""
    today_str = datetime.today().strftime("%Y%m%d")
    for df_name, final_df in dic_of_dfs.items():
        final_df.to_csv(f"data/{df_name}_{today_str}.csv", index=False)


if __name__ == "__main__":
    logging.info("Starting data download")

    logging.info("Fetching S&P 500 ticker list...")
    sp500_dataframe = get_sp500_tickers()
    logging.info("Got %d tickers", len(sp500_dataframe))

    logging.info("Downloading stock price data (this takes a while)...")
    stocks_df = get_stocks_data(sp500_dataframe['Symbol'])
    logging.info("Stock data done: %d rows", len(stocks_df))

    logging.info("Downloading outstanding shares...")
    os_df_days = get_outstanding_shares(sp500_dataframe['Symbol'])
    logging.info("Outstanding shares done: %d rows", len(os_df_days))

    logging.info("Downloading weather data...")
    weather_df = get_weather_data()
    logging.info("Weather data done: %d rows", len(weather_df))

    logging.info("Downloading federal funds rate...")
    ffr = get_federal_funds_rate()
    logging.info("FFR done: %d rows", len(ffr))

    logging.info("Downloading VIX and treasury yields...")
    vix_yields = get_vix_and_yields()
    logging.info("VIX/yields done: %d rows", len(vix_yields))

    logging.info("Downloading sector ETF data...")
    sector_etfs = get_sector_etf_data()
    logging.info("Sector ETFs done: %d rows", len(sector_etfs))

    logging.info("Downloading Wikipedia pageviews...")
    wiki_pageviews = get_wikipedia_pageviews(sp500_dataframe)
    logging.info("Wikipedia pageviews done: %d rows", len(wiki_pageviews))

    logging.info("Saving to CSV...")
    save_data_to_csv({
        'sp_df': sp500_dataframe,
        'wiki_pageviews': wiki_pageviews,
        'stocks_df': stocks_df,
        'os_df_days': os_df_days,
        'ffr': ffr,
        'weather_df': weather_df,
        'vix_yields': vix_yields,
        'sector_etfs': sector_etfs,
    })
