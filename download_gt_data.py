"""Download Google Trends data (finicky so separate from download_data.py for now)"""

import argparse
import glob
import logging
import os
import re
import time
from datetime import datetime, timedelta

import pandas as pd
from pandas.tseries.offsets import MonthBegin, MonthEnd
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError
import requests

# Argument parsing
parser = argparse.ArgumentParser(description='Download Google Trends data.')
parser.add_argument('--keyword', type=str, help='Keyword to add for Google Trends data')
args = parser.parse_args()

new_keyword = args.keyword

# Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data
def load_existing_data() -> tuple:
    """
    Load existing Google Trends data from CSV files.

    Returns:
        tuple: DataFrames for monthly, weekly, and daily data.
    """
    gt_monthly_files = glob.glob('data/gt_monthly_*.csv')
    if gt_monthly_files:
        gt_monthly_latest = max(gt_monthly_files, key=lambda f: f.split("_")[2])
        gt_monthly_loaded = pd.read_csv(gt_monthly_latest, parse_dates=['start_date', 'end_date'])
    else:
        gt_monthly_loaded = pd.DataFrame(columns=['start_date','index','isPartial','end_date','search_term','pytrends_params'])

    gt_weekly_files = glob.glob('data/gt_weekly_*.csv')
    if gt_weekly_files:
        gt_weekly_latest = max(gt_weekly_files, key=lambda f: f.split("_")[2])
        gt_weekly_loaded = pd.read_csv(gt_weekly_latest, parse_dates=['start_date','end_date'])
    else:
        gt_weekly_loaded = pd.DataFrame(columns=['start_date','index','isPartial','end_date','search_term','pytrends_params'])

    gt_daily_files = glob.glob('data/gt_daily_*.csv')
    if gt_daily_files:
        gt_daily_latest = max(gt_daily_files, key=lambda f: f.split("_")[2])
        gt_daily_loaded = pd.read_csv(gt_daily_latest, parse_dates=['date'])
    else:
        gt_daily_loaded = pd.DataFrame(columns=['date','index','isPartial','search_term','pytrends_params'])

    params_return_empty_df_files = glob.glob('params_return_empty_df_*.txt')
    if params_return_empty_df_files:
        params_return_empty_df_files_latest = max(params_return_empty_df_files, key=os.path.getctime)
        with open(params_return_empty_df_files_latest, "r", encoding="utf-8") as f:
            params_return_empty_df_raw = [line.strip() for line in f]
    else:
        params_return_empty_df_raw = []

    return gt_monthly_loaded, gt_weekly_loaded, gt_daily_loaded, params_return_empty_df_raw

def clean_up(gt_monthly, gt_weekly, gt_daily) -> pd.DataFrame:
    """
    Clean up Google Trends data by merging and adjusting indices.

    Parameters:
        gt_monthly (DataFrame): Monthly Google Trends data.
        gt_weekly (DataFrame): Weekly Google Trends data.
        gt_daily (DataFrame): Daily Google Trends data.
    Returns:
        DataFrame: gt_adjusted DataFrame with adjusted indices.
    """
    idx_of_month = gt_monthly.copy()
    idx_of_month['params_date_range'] = idx_of_month['pytrends_params'].str.extract(
        r'"(\d{4}-\d{2}-\d{2} \d{4}-\d{2}-\d{2})"'
    )[0]

    idx_of_month = idx_of_month.loc[
        idx_of_month['params_date_range']==max(idx_of_month['params_date_range'])
    ]
    idx_of_month = idx_of_month.rename(columns={'start_date':'month_start','index':'idx_of_month'})
    idx_of_month = idx_of_month[['month_start','idx_of_month','search_term']]
    idx_of_month = idx_of_month.drop_duplicates(subset=['month_start', 'search_term'], keep='first')

    # there are duplicate start_date/search_term rows because weeks can be spread across different
    # years - smart way to adjust this would be weighted average based on days of week in each year
    # just keeping the first row for now, come back to this later
    idx_of_week = gt_weekly.drop_duplicates(subset=['start_date', 'search_term'], keep='first')
    idx_of_week = idx_of_week[['start_date','index','search_term']]
    idx_of_week = idx_of_week.rename(columns={'start_date':'week_start_sun','index':'idx_of_week'})

    # Daily
    idx_of_day_nas = gt_daily.loc[gt_daily['request_datetime'].isna()]
    idx_of_day_nas = idx_of_day_nas.loc[~idx_of_day_nas['isPartial']]

    idx_of_day = gt_daily.loc[~gt_daily['request_datetime'].isna()]
    idx_of_day = idx_of_day.loc[
        idx_of_day.groupby(['date', 'search_term'])['request_datetime'].idxmax()
    ]
    idx_of_day = pd.concat([idx_of_day_nas, idx_of_day], ignore_index=True)
    idx_of_day = idx_of_day.loc[~idx_of_day['isPartial']] # exclude partial data (last day)

    # Drop duplicates where one row has a request_datetime and the other is missing request_datetime
    idx_of_day = idx_of_day.sort_values(by='request_datetime', na_position='last')
    idx_of_day = idx_of_day.drop_duplicates(
        subset=['date', 'isPartial', 'search_term', 'pytrends_params'],
        keep='first'
    )
    idx_of_day['day_of_week'] = idx_of_day['date'].dt.day_name()
    idx_of_day['week_start_sun'] = idx_of_day["date"].dt.to_period("W-SAT").dt.start_time
    idx_of_day['month_start'] = idx_of_day["date"] - MonthBegin(1)

    # Adjusted
    gt_adjusted = idx_of_day.merge(idx_of_week, how='left', on=['week_start_sun','search_term'])
    gt_adjusted = gt_adjusted.merge(idx_of_month, how='left', on=['month_start','search_term'])

    gt_adjusted['index'] = gt_adjusted['index']*gt_adjusted['idx_of_week']/100
    gt_adjusted['index'] = gt_adjusted['index']*gt_adjusted['idx_of_month']/100

    gt_adjusted = gt_adjusted[['date','day_of_week','search_term','index']]

    return gt_adjusted

def custom_retry(kw, pytrends, df_list, no_resp_list, rep_count):
    """
    Custom retry function to handle exceptions and retry the request.

    Parameters::
        kw (str): Keyword to search for.
        pytrends (TrendReq): Pytrends object.
        df_list (list): List to store DataFrames.
        no_resp_list (list): List to store failed requests.
        rep_count (int): Number of retry attempts.
    """
    for attempt in range(rep_count):
        try:
            resp_df = pytrends.interest_over_time()
            if resp_df.empty:
                no_resp_list.append(str(pytrends.token_payload))
            else:
                resp_df = resp_df.reset_index()
                resp_df = resp_df.rename(columns={kw:'index'})
                resp_df['search_term'] = kw
                resp_df['pytrends_params'] = str(pytrends.token_payload)
                resp_df['request_datetime'] = datetime.now()
                df_list.append(resp_df)

            logging.info("Success!")
            break
        except requests.exceptions.RequestException as e:
            logging.error('RequestException: %s',e)
            if attempt<(rep_count-1):
                logging.error('Sleeping for 71s and then trying attempt %d...',attempt+2)
                time.sleep(71)
        except ResponseError as e:
            logging.error("ResponseError: %s", e)
            if attempt<(rep_count-1):
                logging.error('Sleeping for 71s and then trying attempt %d...',attempt+2)
                time.sleep(71)

def review_past_requests(my_kws, params_return_empty_df, gt_weekly, gt_daily) -> tuple:
    """
    Review past requests to determine which data needs to be fetched.

    Builds year ranges to pull weekly data and week ranges to pull daily data. Checks if the
    data for the keyword and year or keyword and week already exists, avoiding redundant
    requests. Google Trends returns weekly data when a full year is requested and daily data
    when one week is requested.

    Parameters:
        my_kws (set): Set of keywords to search for.
        params_return_empty_df (list): List of parameters that returned empty data frames.
        gt_weekly (DataFrame): DataFrame containing raw weekly Google Trends data.
        gt_daily (DataFrame): DataFrame containing raw daily Google Trends data.

    Returns:
        tuple: Contains dictionaries for year ranges to do, week ranges to do, and parameters
            that returned empty data frames.
    """
    # Build year ranges to pull weekly data
    years = list(range(2004, datetime.now().year+1))
    year_ranges = [(f'{year}-01-01 {year}-12-31') for year in years]

    logging.info('Since 2004 there are %d year ranges', len(year_ranges))
    kw_yrc = {}
    kw_yrtd = {}
    for kw in my_kws:
        kw_data = gt_weekly.loc[gt_weekly['search_term']==kw]

        year_ranges_completed = kw_data['pytrends_params'].str.extract(
            r'"(\d{4}-\d{2}-\d{2} \d{4}-\d{2}-\d{2})"'
        )[0]
        year_ranges_completed = list(set(year_ranges_completed))

        # Remove the current year - new weeks in current year may have occurred
        current_year_date_range = f"{datetime.now().year}-01-01 {datetime.now().year}-12-31"
        year_ranges_completed = [x for x in year_ranges_completed if x != current_year_date_range]

        kw_yrc[kw] = year_ranges_completed
        logging.info('Already have %d year ranges for "%s"', len(kw_yrc[kw]), kw)

        year_ranges_to_do = [x for x in year_ranges if x not in kw_yrc[kw]]
        year_ranges_to_do.sort()

        kw_yrtd[kw] = year_ranges_to_do
        logging.info('Need to get %d year ranges for "%s"', len(year_ranges_to_do), kw)

    print('-----------------------------------------------------------------------------')
    # Build week ranges to pull daily data
    week_ranges = []
    current = datetime.strptime("2003-12-28", "%Y-%m-%d")
    while current <= datetime.today():
        week_start = current
        week_end = current + timedelta(days=6)
        week_ranges.append(f"{week_start.strftime('%Y-%m-%d')} {week_end.strftime('%Y-%m-%d')}")
        current += timedelta(weeks=1)

    params_return_empty_df_dict ={}
    for kw in my_kws:
        extracted_dates = [
            re.search(r'"(\d{4}-\d{2}-\d{2} \d{4}-\d{2}-\d{2})"', s)
            for s in params_return_empty_df
            if kw in s
        ]
        params_return_empty_df_dict[kw] = [match.group(1) for match in extracted_dates]

    logging.info('Since 2004 there are %d week ranges', len(week_ranges))
    kw_wrc = {}
    kw_wrtd = {}
    for kw in my_kws:
        kw_data = gt_daily.loc[gt_daily['search_term']==kw]

        # Exclude partial data (we want to request again to finish incomplete weeks)

        # Can't just exclude the isPartial==True rows,
        # the week range is still in another row where isPartial==False

        # So, lets get a list of pytrends_params where isPartial==True
        ptp_partial_t = kw_data.loc[kw_data['isPartial'],'pytrends_params'].unique()

        # Now lets go back and exclude rows where pytrends_params is one of these values
        kw_data = kw_data.loc[~kw_data['pytrends_params'].isin(ptp_partial_t)]

        week_ranges_completed = kw_data['pytrends_params'].str.extract(
            r'"(\d{4}-\d{2}-\d{2} \d{4}-\d{2}-\d{2})"'
        )[0]
        week_ranges_completed = list(set(week_ranges_completed))

        kw_wrc[kw] = week_ranges_completed
        logging.info('Already have %d week ranges for "%s"', len(kw_wrc[kw]), kw)

        week_ranges_to_do = [x for x in week_ranges if x not in kw_wrc[kw]]
        week_ranges_to_do.sort()

        kw_wrtd[kw] = week_ranges_to_do
        logging.info('Need to get %d week ranges for "%s"', len(week_ranges_to_do), kw)

        if params_return_empty_df_dict[kw]:
            logging.info(
                'But %d params returned empty data frames, so just need %d week ranges for "%s"',
                len(params_return_empty_df_dict[kw]),
                len(week_ranges_to_do)-len(params_return_empty_df_dict[kw]),
                kw
            )
    return kw_yrtd, kw_wrtd, params_return_empty_df_dict


def main():
    """
    Main function to download Google Trends data
    """
    today_yyyymmdd = datetime.today().strftime("%Y%m%d")

    gt_monthly_raw, gt_weekly_raw, gt_daily_raw, params_return_empty_df_raw = load_existing_data()

    pytrends = TrendReq(retries=8, backoff_factor=2)

    past_weekly_requests = set(gt_weekly_raw['pytrends_params'])

    # Exclude partial data (we want to request again to finish incomplete weeks)

    # Can't just exclude the isPartial==True rows,
    # the week range is still in another row where isPartial==False

    # So, lets get a list of pytrends_params where isPartial==True
    ptp_partial_t = gt_daily_raw.loc[gt_daily_raw['isPartial'],'pytrends_params'].unique()

    # Now lets go back and exclude rows where pytrends_params is one of these values
    gt_daily_raw_adj = gt_daily_raw.loc[~gt_daily_raw['pytrends_params'].isin(ptp_partial_t)]

    past_daily_requests = set(list(gt_daily_raw_adj['pytrends_params'])+params_return_empty_df_raw) # params_return_empty_df_raw only in daily for now

    if new_keyword:
        my_kws = set(list(gt_daily_raw['search_term'].unique())+[new_keyword])
    else:
        my_kws = set(gt_daily_raw['search_term'])

        if not my_kws:
            logging.error(
                "No keywords found in existing data. Provide a keyword using --keyword argument."
            )
            return

    # Maybe I can remove duplicate logic if replace gt_daily_raw with gt_daily_raw_adj here
    kw_yrtd, kw_wrtd, params_return_empty_df_dict = review_past_requests(
        my_kws, params_return_empty_df_raw, gt_weekly_raw, gt_daily_raw
    )

    # Get the interest index by month since 2004
    dat = []
    for kw in my_kws:
        try:
            pytrends.build_payload(
                [kw],
                cat=0,
                timeframe=f'2004-01-01 {datetime.now().strftime("%Y-%m-%d")}',
                geo="US"
            )
            df = pytrends.interest_over_time()
            df = df.reset_index()
            df = df.rename(columns={'date':'start_date', kw:'index'})
            df['end_date'] = df['start_date'] + MonthEnd(0)
            df['search_term'] = kw
            df['pytrends_params'] = str(pytrends.token_payload)
            df['request_datetime'] = datetime.now()

            dat.append(df)

        except requests.exceptions.RequestException as e:
            logging.error("RequestException: %s", e)
        except ResponseError as e:
            logging.error("ResponseError: %s", e)

    if dat:
        gt_monthly_new = pd.concat(dat)

        gt_monthly = pd.concat([gt_monthly_raw,gt_monthly_new])
        gt_monthly = gt_monthly.drop_duplicates()

        gt_monthly.to_csv(f'data/gt_monthly_{today_yyyymmdd}.csv', index=False)

    # Get the interest index by week for each year for the selected keywords
    # past_weekly_requests_ncy (not current year)
    current_year_date_range = f"{datetime.now().year}-01-01 {datetime.now().year}-12-31"
    past_weekly_requests_ncy = [x for x in past_weekly_requests if current_year_date_range not in x]
    dat = []
    for kw in my_kws:
        for one_year_timeframe in kw_yrtd[kw]:
            try:
                pytrends.build_payload([kw], cat=0, timeframe=one_year_timeframe, geo="US")

                if str(pytrends.token_payload) not in past_weekly_requests_ncy:
                    weekly_us = pytrends.interest_over_time()
                    weekly_us = weekly_us.reset_index()
                    weekly_us = weekly_us.rename(columns={'date':'start_date', kw:'index'})
                    weekly_us['end_date'] = weekly_us['start_date'] + pd.Timedelta(days=6)
                    weekly_us['search_term'] = kw
                    weekly_us['pytrends_params'] = str(pytrends.token_payload)
                    weekly_us['request_datetime'] = datetime.now()

                    dat.append(weekly_us)
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                logging.error("RequestException: %s", e)
                break # if it fails for one year, it will continue to fail
            except ResponseError as e:
                logging.error("ResponseError: %s", e)
                break
    if dat:
        gt_weekly_new = pd.concat(dat)

        gt_weekly = pd.concat([gt_weekly_raw,gt_weekly_new])
        gt_weekly = gt_weekly.drop_duplicates()

        gt_weekly.to_csv(f'data/gt_weekly_{today_yyyymmdd}.csv', index=False)

    # Get the interest index by day for each week for the selected keyword
    params_return_empty_df_new = []
    dat = []
    for kw in my_kws:
        for one_week_timeframe in kw_wrtd[kw]:
            try:
                if one_week_timeframe in params_return_empty_df_dict[kw]:
                    continue

                pytrends.build_payload([kw], cat=0, timeframe=one_week_timeframe, geo="US")
                logging.info('%s "%s" Payload built successfully', one_week_timeframe, kw)

                if str(pytrends.token_payload) not in past_daily_requests:
                    logging.info("This week is new, gathering interest_over_time...")

                    custom_retry(kw, pytrends, dat, params_return_empty_df_new, 3)

                time.sleep(2)

            except requests.exceptions.RequestException as e:
                logging.error("RequestException: %s", e) # no break here, worth trying again
            except ResponseError as e:
                logging.error("ResponseError: %s", e)

    if dat:
        gt_daily_new = pd.concat(dat)

        gt_daily = pd.concat([gt_daily_raw,gt_daily_new])
        gt_daily = gt_daily.drop_duplicates()

        gt_daily.to_csv(f'data/gt_daily_{today_yyyymmdd}.csv', index=False)

    if params_return_empty_df_new:
        params_return_empty_df = params_return_empty_df_raw+params_return_empty_df_new
        with open(
            f"data/params_return_empty_df_{today_yyyymmdd}.txt",
            "w",
            encoding="utf-8"
        ) as f:
            f.writelines(f"{item}\n" for item in params_return_empty_df)

    # Clean up (refresh "raw" files first) to ensure the latest data is used for the cleanup process
    gt_monthly_refreshed, gt_weekly_refreshed, gt_daily_refreshed, _ = load_existing_data()

    gt_adjusted_raw = clean_up(gt_monthly_refreshed, gt_weekly_refreshed, gt_daily_refreshed)

    gt_adjusted_raw.to_csv(f'data/gt_adjusted_{today_yyyymmdd}.csv', index=False)

if __name__ == "__main__":
    main()
