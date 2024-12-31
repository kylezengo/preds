import pandas as pd
import requests
import urllib.parse
import yfinance as yf
from datetime import datetime, date

# def download_stocks_data(start_date, end_date):
#     # Get list of S&P 500 tickers from Wikipedia page
#     sp_wiki = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

#     sp_df = pd.read_html(sp_wiki.content)[0]

#     sp500_tickers = sp_df['Symbol']
#     
#     selected_tickers = list(sp500_tickers)+["SPY"]
#     
#     dat_list = []
#     for i in selected_tickers:
#         data = yf.download(i, start=start_date, end=end_date, progress=False)
#         data['ticker'] = i
#         dat_list.append(data)
        
#     stocks_df = pd.concat(dat_list)
#     stocks_df = stocks_df.reset_index()

#     # Download outstanding shares data
#     dat_list = []
#     for i in sp500_tickers:
#         if yf.Ticker(i).get_shares_full() is not None:
#             os = yf.Ticker(i).get_shares_full(start=start_date, end=end_date)
#             os = pd.DataFrame(os)
#             os['ticker'] = i
#             dat_list.append(os)
            
#     os_df = pd.concat(dat_list)

#     os_df = os_df.rename_axis('os_report_datetime').reset_index()
#     os_df['os_report_date'] = pd.to_datetime(os_df['os_report_datetime'].dt.date)
#     os_df = os_df.rename(columns={0: 'outstanding_shares'})
#     os_df = os_df.groupby(['os_report_datetime','os_report_date','ticker']).agg(outstanding_shares = ('outstanding_shares','mean')).reset_index()

#     # Group to dates
#     # is mean correct? or should we take last value? (would only matter if there are duplicate dates above)
#     os_df_date_tick = os_df.groupby(['os_report_date','ticker']).agg(outstanding_shares = ('outstanding_shares','mean')).reset_index()

#     # Set outstanding shares for each day
#     dat_list = []
#     prev_date = None
#     prev_ticker = None
#     prev_value = None
#     for index, row in os_df_date_tick.iterrows():
#         current_date = row['os_report_date']
#         current_ticker = row['ticker']
#         current_value = row['outstanding_shares']
        
#         if prev_date is not None:
#             # Generate missing dates and values
#             missing_dates = pd.date_range(start=prev_date, end=current_date, inclusive='neither')
            
#             # Append missing dates and values to dat_list
#             dat_list.append(pd.DataFrame({'os_report_date': prev_date
#                                         ,'date': missing_dates
#                                         ,'ticker': prev_ticker
#                                         ,'outstanding_shares': prev_value}))
        
#         # Add the current row to dat_list
#         dat_list.append(pd.DataFrame({'os_report_date': [current_date]
#                                     ,'date': [current_date]
#                                     ,'ticker': [current_ticker]
#                                     ,'outstanding_shares': [current_value]}))
        
#         # Update previous date and value
#         prev_date = current_date
#         prev_ticker = current_ticker
#         prev_value = current_value

#     os_df_days = pd.concat(dat_list, ignore_index=True)

#     return [sp_df,stocks_df,os_df_days]


start_date = "1993-01-29" # SPY launched on 1993-01-22 ... first data is January 29?
end_date = datetime.today().strftime('%Y-%m-%d')

# Get list of S&P 500 tickers from Wikipedia page
sp_wiki = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

sp_df = pd.read_html(sp_wiki.content)[0]

# Get the wikipedia page for each company (part of url)
sp_df['wiki_page'] = sp_df['Security'].apply(lambda x: urllib.parse.quote_plus(x))

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
sdate="20150701" # earliest date is"20150701"
edate=(date.today()-pd.Timedelta(days=1)).strftime('%Y%m%d') # yesterday

headers = {
    "User-Agent": "zengokp@gmail.com"
}

dat=[]
missing=[]
for page in sp_df['wiki_page']:
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{page}/daily/{sdate}/{edate}"

    response = requests.get(url,headers=headers)
    try:
        df = pd.DataFrame(response.json()['items'])
        df['ticker'] = sp_df.loc[sp_df['wiki_page']==page,'Symbol'].item()
    except:
        print(f"wiki pageviews error: {page}")
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
    data = yf.download(i, start=start_date, end=end_date, progress=False)
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

sp_df.to_csv(f'sp_df_{datetime.today().strftime("%Y%m%d")}.csv', index=False)
wiki_pageviews.to_csv(f'wiki_pageviews_{datetime.today().strftime("%Y%m%d")}.csv', index=False)
stocks_df.to_csv(f'stocks_df_{datetime.today().strftime("%Y%m%d")}.csv', index=False)
os_df_days.to_csv(f'os_df_days_{datetime.today().strftime("%Y%m%d")}.csv', index=False)