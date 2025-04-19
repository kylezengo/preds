"""This module downloads the data necessary to test different strategies"""

import glob
import os
import time
import urllib.parse
from datetime import datetime, date, timedelta

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

# Authentication
load_dotenv()
fred_api_key = os.getenv("fred_api_key")
noaa_api_key = os.getenv("noaa_api_key")
wiki_user_agent = os.getenv("wiki_user_agent")

WIKI_SDATE = "20150701" # earliest date is"20150701"
WIKI_BASE_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
START_DATE = "1993-01-29" # SPY launched on 1993-01-22 ... first data is January 29?
end_date = datetime.today().strftime('%Y-%m-%d')

def load_existing_data():
    """
    Load existing weather data from CSV files.
    """
    weather_df_files = glob.glob('weather_df_*.csv')
    weather_df_latest = max(weather_df_files, key=os.path.getctime)
    weather_df_loaded = pd.read_csv(weather_df_latest, parse_dates=['date'])

    return weather_df_loaded

def get_federal_funds_rate():
    """
    Get federal funds rate using the Fred API (not exact match to current numbers?)
    """
    fred = Fred(api_key=fred_api_key)
    build_ffr = fred.get_series('FEDFUNDS').to_frame(name='federal_funds_rate')
    build_ffr.loc[datetime.today().strftime('%Y-%m-%d'), 'federal_funds_rate'] = build_ffr['federal_funds_rate'][-1]
    build_ffr = build_ffr.resample('D').ffill()
    build_ffr = build_ffr.reset_index(names='Date')
    return build_ffr

def get_sp500_tickers():
    """
    Get list of S&P 500 tickers from Wikipedia page
    """
    sp_wiki = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=20)

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
        "Campbell%27s+Company+%28The%29": "Campbell%27s"
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
    wiki_edate=(date.today()-pd.Timedelta(days=1)).strftime('%Y%m%d') # yesterday

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
                print(f"'items' key missing in response for page: {page}")
                missing.append(page)

        except requests.exceptions.RequestException as e:
            print(f"Request error for {page}: {e}")
            missing.append(page)

        except ValueError as e:
            print(f"ValueError for page {page}: {e}")
            missing.append(page)

        if len(items_df)==0:
            print(f"wiki pageviews data frame empty: {page}")
            missing.append(page)

        dat.append(items_df)

    build_wiki_pv = pd.concat(dat).reset_index(drop=True)
    build_wiki_pv['Date'] =  pd.to_datetime(build_wiki_pv['timestamp'], format='%Y%m%d%H')
    return build_wiki_pv

def save_data_to_csv(dic_of_dfs):
    """Save data to CSV files"""
    today_str = datetime.today().strftime("%Y%m%d")
    for df_name, final_df in dic_of_dfs.items():
        final_df.to_csv(f"{df_name}_{today_str}.csv", index=False)


sp500_dataframe = get_sp500_tickers()

####################################################################################################
# Get daily stocks data for each company wih yfinance ##############################################
####################################################################################################
sp500_tickers = sp500_dataframe['Symbol']

selected_tickers = list(sp500_tickers)

dat_list = []
for i in selected_tickers:
    data = yf.download(
        i,
        start=START_DATE,
        end=end_date,
        auto_adjust=False, # ?
        progress=False
    )
    data.columns = data.columns.get_level_values(0)
    data['ticker'] = i
    dat_list.append(data)
    time.sleep(1)

stocks_df = pd.concat(dat_list)
stocks_df = stocks_df.reset_index()
stocks_df = stocks_df.rename_axis(None, axis=1)

# Get outstanding shares data
dat_list = []
for i in sp500_tickers:
    if yf.Ticker(i).get_shares_full() is not None:
        df = yf.Ticker(i).get_shares_full(start=START_DATE, end=end_date)
        df = pd.DataFrame(df)
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

# Set outstanding shares for each day
# can this be updated to use .ffill?()?
dat_list = []
prev_date = None
prev_ticker = None
prev_value = None
for index, row in os_df_date_tick.iterrows():
    current_date = row['os_report_date']
    current_ticker = row['ticker']
    current_value = row['outstanding_shares']

    if prev_date is not None:
        # Generate missing dates and values
        missing_dates = pd.date_range(start=prev_date, end=current_date, inclusive='neither')

        # Append missing dates and values to dat_list
        dat_list.append(pd.DataFrame({'os_report_date': prev_date
                                    ,'date': missing_dates
                                    ,'ticker': prev_ticker
                                    ,'outstanding_shares': prev_value}))

    # Add the current row to dat_list
    dat_list.append(pd.DataFrame({'os_report_date': [current_date]
                                ,'date': [current_date]
                                ,'ticker': [current_ticker]
                                ,'outstanding_shares': [current_value]}))

    # Update previous date and value
    prev_date = current_date
    prev_ticker = current_ticker
    prev_value = current_value

os_df_days = pd.concat(dat_list, ignore_index=True)

####################################################################################################
# Get historical weaher data for NYC from NOAA and weather.gov #####################################
####################################################################################################
NOAA_BASE_URL = 'https://www.ncei.noaa.gov/cdo-web/api/v2/data'
OBSERVATIONS_URL = "https://api.weather.gov/stations/KNYC/observations"
STATIONID = "GHCND:USW00094728" # Central Park Station in NYC

weather_df_hist = load_existing_data()

# Set up date_ranges - the data we are going to get download in the current run
# date_ranges is a list of dictionaries with start and end dates for 29 (30?) day periods
# filter to end_date >= min(one_year_ago,last_hist_data_date)
end_date_dt = datetime.strptime(end_date, "%Y-%m-%d") # to datetime

current_start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
date_ranges = []
while current_start_date <= end_date_dt:
    current_end_date = current_start_date + timedelta(days=29)
    current_end_date = min(current_end_date, end_date_dt)

    date_ranges.append({'start_date': current_start_date.strftime("%Y-%m-%d")
                        ,'end_date': current_end_date.strftime("%Y-%m-%d")})

    current_start_date = current_end_date + timedelta(days=1)

date_ranges_df = pd.DataFrame(date_ranges)

one_year_ago = (datetime.now()-timedelta(days=365)).strftime("%Y-%m-%d") # to string
last_hist_data_date = max(weather_df_hist['date']).strftime("%Y-%m-%d") # to string

date_ranges_df = date_ranges_df.loc[
    date_ranges_df['end_date'] >= min(one_year_ago,last_hist_data_date)
]
date_ranges_df = date_ranges_df.reset_index(drop=True)

headers = {
    'token': noaa_api_key
}

weather_data = {}
for index, row in date_ranges_df.iterrows():
    params = {
        'datasetid': 'GHCND',  # Daily Summaries dataset
        'stationid': STATIONID,
        'startdate': row['start_date'],
        'enddate': row['end_date'],
        'units': 'metric',  # Use metric units (Celsius for temperatures, mm for precipitation)
        'limit': 1000  # Maximum number of records to fetch
    }

    resp = requests.get(NOAA_BASE_URL, headers=headers, params=params, timeout=300)

    if resp.status_code != 200:
        print(f"Error at start date {row['start_date']}: {resp.status_code}, {resp.text}")
        print(f"Trying start date {row['start_date']} again")
        time.sleep(10)
        resp = requests.get(NOAA_BASE_URL, headers=headers, params=params, timeout=300)
        if resp.status_code != 200:
            print(f"Error #2 at start date {row['start_date']}: {resp.status_code}, {resp.text}")
            print(f"Failed twice. Trying start date {row['start_date']} one last time")
            time.sleep(20)
            resp = requests.get(NOAA_BASE_URL, headers=headers, params=params, timeout=300)
            if resp.status_code != 200:
                print(f"Error at start date {row['start_date']}: {resp.status_code}, {resp.text}")
                print("Failed three times. Not trying again")
                break

    print(resp.status_code, row['start_date'])
    data = resp.json()

    # Check if 'results' key exists
    if 'results' not in data:
        print(f"No 'results' key in response for start date {row['start_date']} data: {data}")
        continue  # Skip to the next iteration if 'results' is missing

    for item in data['results']:
        item_date = item['date']
        datatype = item['datatype']
        value = item['value']

        # Initialize date entry if not already present
        if item_date not in weather_data:
            weather_data[item_date] = {'date': item_date
                                    ,'high_temp_nyc': None
                                    ,'low_temp_nyc': None
                                    ,'precipitation_PRCP_nyc': None
                                    ,'precipitation_SNOW_nyc': None}

        # Update the weather data dictionary based on the datatype
        if datatype == 'TMAX':
            weather_data[item_date]['high_temp_nyc'] = value
        elif datatype == 'TMIN':
            weather_data[item_date]['low_temp_nyc'] = value
        elif datatype == 'PRCP':
            weather_data[item_date]['precipitation_PRCP_nyc'] = value
        elif datatype == 'SNOW':
            weather_data[item_date]['precipitation_SNOW_nyc'] = value
    time.sleep(1)

weather_records = list(weather_data.values())

noaa_weather = pd.DataFrame(weather_records)
noaa_weather['date'] = pd.to_datetime(noaa_weather['date'])

# Recent weather
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
        'windSpeed': props.get("windSpeed", {}).get("value"), # In km_h-1
        'precipitationLastHour': props.get("precipitationLastHour", {}).get("value"),  # In mm
        'precipitationLast3Hours': props.get("precipitationLast3Hours", {}).get("value"),  # In mm
        'precipitationLast6Hours': props.get("precipitationLast6Hours", {}).get("value"),  # In mm
    })

recent_weather_df = pd.DataFrame(recent_weather_data)
recent_weather_df['timestamp'] = pd.to_datetime(recent_weather_df['timestamp'])
recent_weather_df['date'] = pd.to_datetime(recent_weather_df['timestamp'].dt.date)

#
simp_cols = ['date','timestamp','temperature']
weather_simp = recent_weather_df[simp_cols]

weather_simp = weather_simp.groupby('date').agg(low_temp_nyc=('temperature', 'min')
                                                ,high_temp_nyc=('temperature', 'max')).reset_index()

just_new = weather_simp.loc[weather_simp['date']>max(noaa_weather['date'])]

# Putting it all togeather
weather_df_hist = weather_df_hist.loc[weather_df_hist['date']<min(noaa_weather['date'])]

weather_df = pd.concat([weather_df_hist, noaa_weather, just_new]).reset_index(drop=True)

####################################################################################################
# main #############################################################################################
####################################################################################################
ffr = get_federal_funds_rate()
wiki_pageviews = get_wikipedia_pageviews(sp500_dataframe)

save_data_to_csv({
    'sp_df':sp500_dataframe,
    'wiki_pageviews':wiki_pageviews,
    'stocks_df':stocks_df,
    'os_df_days':os_df_days,
    'ffr':ffr,
    'weather_df':weather_df
})
