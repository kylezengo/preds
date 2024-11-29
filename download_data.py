import requests
import pandas as pd
import yfinance as yf

from datetime import date

def download_stocks_data(start_date, end_date):
    # Get list of S&P 500 tickers from Wikipedia page
    sp_wiki = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

    sp_df = pd.read_html(sp_wiki.content)[0]

    stocks_list = sp_df['Symbol']

    dat_list = []
    for i in stocks_list:
        data = yf.download(i, start=start_date, end=end_date, progress=False)
        data['ticker'] = i
        dat_list.append(data)
        
    stocks_df = pd.concat(dat_list)
    stocks_df = stocks_df.reset_index()

    # Download SPY data
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
    spy = spy.reset_index()

    # Download outstanding shares data
    dat_list = []
    for i in stocks_list:
        if yf.Ticker(i).get_shares_full() is not None:
            os = yf.Ticker(i).get_shares_full(start=start_date, end=end_date)
            os = pd.DataFrame(os)
            os['ticker'] = i
            dat_list.append(os)
            
    os_df = pd.concat(dat_list)

    os_df = os_df.rename_axis('os_report_datetime').reset_index()
    os_df['os_report_date'] = pd.to_datetime(os_df['os_report_datetime'].dt.date)
    os_df = os_df.rename(columns={0: 'outstanding_shares'})
    os_df = os_df.groupby(['os_report_datetime','os_report_date','ticker']).agg(outstanding_shares = ('outstanding_shares','mean')).reset_index()

    # Group to dates
    # is mean correct? or should we take last value? (would only matter if there are duplicate dates above)
    os_df_date_tick = os_df.groupby(['os_report_date','ticker']).agg(outstanding_shares = ('outstanding_shares','mean')).reset_index()

    # Set outstanding shares for each day
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

    return [sp_df,stocks_df,spy,os_df_days]