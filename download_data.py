"""This module downloads the data necessary to test different strategies"""

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

START_DATE = "1993-01-29" # SPY launched on 1993-01-22 ... first data is January 29?
end_date = datetime.today().strftime('%Y-%m-%d')

# Get list of S&P 500 tickers from Wikipedia page
sp_wiki = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", timeout=20)

sp_df = pd.read_html(sp_wiki.content)[0]

# Get the wikipedia page for each company (part of url)
sp_df['wiki_page'] = sp_df['Security'].apply(urllib.parse.quote_plus)

sp_df.loc[sp_df['wiki_page']=="Alphabet+Inc.+%28Class+A%29",'wiki_page'] = "Alphabet_Inc."
sp_df.loc[sp_df['wiki_page']=="Alphabet+Inc.+%28Class+C%29",'wiki_page'] = "Alphabet_Inc."
sp_df.loc[sp_df['wiki_page']=='Coca-Cola+Company+%28The%29','wiki_page'] = "The_Coca-Cola_Company"
sp_df.loc[sp_df['wiki_page']=='Cooper+Companies+%28The%29','wiki_page'] = "The_Cooper_Companies"
sp_df.loc[sp_df['wiki_page']=='Est%C3%A9e+Lauder+Companies+%28The%29','wiki_page'] = "The_Estée_Lauder_Companies"
sp_df.loc[sp_df['wiki_page']=='Fox+Corporation+%28Class+A%29','wiki_page'] = "Fox_Corporation"
sp_df.loc[sp_df['wiki_page']=='Fox+Corporation+%28Class+B%29','wiki_page'] = "Fox_Corporation"
sp_df.loc[sp_df['wiki_page']=='Hartford+%28The%29','wiki_page'] = "The_Hartford"
sp_df.loc[sp_df['wiki_page']=='Hershey+Company+%28The%29','wiki_page'] = "The_Hershey_Company"
sp_df.loc[sp_df['wiki_page']=='Home+Depot+%28The%29','wiki_page'] = "Home_Depot"
sp_df.loc[sp_df['wiki_page']=='Interpublic+Group+of+Companies+%28The%29','wiki_page'] = "The_Interpublic_Group_of_Companies"
sp_df.loc[sp_df['wiki_page']=='Lilly+%28Eli%29','wiki_page'] = "Eli_Lilly_and_Company"
sp_df.loc[sp_df['wiki_page']=='Mosaic+Company+%28The%29','wiki_page'] = "The_Mosaic_Company"
sp_df.loc[sp_df['wiki_page']=='News+Corp+%28Class+A%29','wiki_page'] = "News_Corp"
sp_df.loc[sp_df['wiki_page']=='News+Corp+%28Class+B%29','wiki_page'] = "News_Corp"
sp_df.loc[sp_df['wiki_page']=='PTC+Inc.','wiki_page'] = "PTC_(software_company)"
sp_df.loc[sp_df['wiki_page']=='J.M.+Smucker+Company+%28The%29','wiki_page'] = "The_J.M._Smucker_Company"
sp_df.loc[sp_df['wiki_page']=='Travelers+Companies+%28The%29','wiki_page'] = "The_Travelers_Companies"
sp_df.loc[sp_df['wiki_page']=='Walt+Disney+Company+%28The%29','wiki_page'] = "The_Walt_Disney_Company"

# Get daily wikipedia pageviews for each company
WIKI_SDATE="20150701" # earliest date is"20150701"
wiki_edate=(date.today()-pd.Timedelta(days=1)).strftime('%Y%m%d') # yesterday

headers = {
    "User-Agent": wiki_user_agent
}

dat=[]
missing=[]
for page in set(sp_df['wiki_page']):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{page}/daily/{WIKI_SDATE}/{wiki_edate}"

    try:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        json_data = response.json()
        if 'items' in json_data:
            df = pd.DataFrame(json_data['items'])
            df['ticker'] = sp_df.loc[sp_df['wiki_page']==page,'Symbol'].item()
        else:
            print(f"'items' key missing in response for page: {page}")
            missing.append(page)

    except requests.exceptions.RequestException as e:
        print(f"Request error for {page}: {e}")
        missing.append(page)

    except ValueError as e:
        print(f"ValueError for page {page}: {e}")
        missing.append(page)

    if len(df)==0:
        print(f"wiki pageviews df empty: {page}")
        missing.append(page)

    dat.append(df)

wiki_pageviews = pd.concat(dat).reset_index(drop=True)
wiki_pageviews['Date'] =  pd.to_datetime(wiki_pageviews['timestamp'], format='%Y%m%d%H')


# Get daily stocks data
sp500_tickers = sp_df['Symbol']

selected_tickers = list(sp500_tickers)+["SPY"]

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
os_df = os_df.groupby(['os_report_datetime','os_report_date','ticker']).agg(outstanding_shares = ('outstanding_shares','mean')).reset_index()

# Group to dates
# is mean correct? or should I take last value? only matters if there are duplicate dates above
os_df_date_tick = os_df.groupby(['os_report_date','ticker']).agg(outstanding_shares = ('outstanding_shares','mean')).reset_index()

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


# Federal funds rate (not exact match to current numbers?)
fred = Fred(api_key=fred_api_key)

ffr = fred.get_series('FEDFUNDS').to_frame(name='federal_funds_rate')

ffr.loc[datetime.today().strftime('%Y-%m-%d'), 'federal_funds_rate'] = ffr['federal_funds_rate'][-1]

ffr = ffr.resample('D').ffill()
ffr = ffr.reset_index(names='Date')


# Get historical weaher data for NYC from NOAA
current_start_date = datetime.strptime(START_DATE, "%Y-%m-%d")
end_date = datetime.strptime(end_date, "%Y-%m-%d")

date_ranges = []
while current_start_date <= end_date:
    current_end_date = current_start_date + timedelta(days=29)

    current_end_date = min(current_end_date, end_date)

    date_ranges.append({'start_date': current_start_date.strftime("%Y-%m-%d")
                        ,'end_date': current_end_date.strftime("%Y-%m-%d")})

    current_start_date = current_end_date + timedelta(days=1)

date_ranges_df = pd.DataFrame(date_ranges)

NOAA_BASE_URL = 'https://www.ncei.noaa.gov/cdo-web/api/v2/data'

headers = {
    'token': noaa_api_key
}

weather_data = {}
for index, row in date_ranges_df.iterrows():
    params = {
        'datasetid': 'GHCND',  # Daily Summaries dataset
        'stationid': 'GHCND:USW00094728',  # Central Park Station in NYC
        'startdate': row['start_date'],
        'enddate': row['end_date'],
        'units': 'metric',  # Use metric units (Celsius for temperatures, mm for precipitation)
        'limit': 1000  # Maximum number of records to fetch
    }

    response = requests.get(NOAA_BASE_URL, headers=headers, params=params, timeout=300)

    if response.status_code != 200:
        print(f'Error at start date {row['start_date']}: {response.status_code}, {response.text}')
        print(f'Trying start date {row['start_date']} again')
        time.sleep(10)
        response = requests.get(NOAA_BASE_URL, headers=headers, params=params, timeout=300)
        if response.status_code != 200:
            print(f'Second error at start date {row['start_date']}: {response.status_code}, {response.text}')
            print(f'Failed twice. Trying start date {row['start_date']} one last time')
            time.sleep(20)
            response = requests.get(NOAA_BASE_URL, headers=headers, params=params, timeout=300)
            if response.status_code != 200:
                print(f'Error at start date {row['start_date']}: {response.status_code}, {response.text}')
                print("Failed three times. Not trying again")
                break

    print(response.status_code, row['start_date'])
    data = response.json()

    for item in data['results']:
        date = item['date']
        datatype = item['datatype']
        value = item['value']

        # Initialize date entry if not already present
        if date not in weather_data:
            weather_data[date] = {'date': date
                                    ,'high_temp_nyc': None
                                    ,'low_temp_nyc': None
                                    ,'precipitation_PRCP_nyc': None
                                    ,'precipitation_SNOW_nyc': None}

        # Update the weather data dictionary based on the datatype
        if datatype == 'TMAX':
            weather_data[date]['high_temp_nyc'] = value
        elif datatype == 'TMIN':
            weather_data[date]['low_temp_nyc'] = value
        elif datatype == 'PRCP':
            weather_data[date]['precipitation_PRCP_nyc'] = value
        elif datatype == 'SNOW':
            weather_data[date]['precipitation_SNOW_nyc'] = value
    time.sleep(1)

weather_records = list(weather_data.values())

weather_df = pd.DataFrame(weather_records)
weather_df['date'] = pd.to_datetime(weather_df['date'])

# Recent weather
OBSERVATIONS_URL = "https://api.weather.gov/stations/KNYC/observations"

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

# Putting it all togeather
just_new = weather_simp.loc[weather_simp['date']>max(weather_df['date'])]

weather_df = pd.concat([weather_df,just_new]).reset_index(drop=True)


# Save everything to csv files
sp_df.to_csv(f'sp_df_{datetime.today().strftime("%Y%m%d")}.csv', index=False)
wiki_pageviews.to_csv(f'wiki_pageviews_{datetime.today().strftime("%Y%m%d")}.csv', index=False)
stocks_df.to_csv(f'stocks_df_{datetime.today().strftime("%Y%m%d")}.csv', index=False)
os_df_days.to_csv(f'os_df_days_{datetime.today().strftime("%Y%m%d")}.csv', index=False)
ffr.to_csv(f'ffr_{datetime.today().strftime("%Y%m%d")}.csv', index=False)
weather_df.to_csv(f'weather_df_{datetime.today().strftime("%Y%m%d")}.csv', index=False)
