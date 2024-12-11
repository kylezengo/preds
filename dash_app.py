
import pandas as pd
import plotly.graph_objects as go
import glob
import os

from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Output, Input

import strat_defs # custom functions from github 

app = Dash()

# Load data
stocks_df_files = glob.glob('/Users/kylezengo/Desktop/DS/Stocks/stocks_df/*.csv')
stocks_df_latest = max(stocks_df_files, key=os.path.getctime)
stocks_df = pd.read_csv(stocks_df_latest, parse_dates=['Date'])

spy_full_files = glob.glob('/Users/kylezengo/Desktop/DS/Stocks/spy_full/*.csv')
spy_full_latest = max(spy_full_files, key=os.path.getctime)
spy_full = pd.read_csv(spy_full_latest, parse_dates=['Date'])


# App layout
app.layout = html.Div(     
    [
        html.Label("Ticker"),
        dcc.Input(
            id='ticker-input',
            value='SPY'
        ),
        html.Label("Short Window"),
        dcc.Input(
            id='short_window-input',
            type='number',
            value=10
        ),
        html.Label(" Long Window"),
        dcc.Input(
            id='long_window-input',
            type='number',
            value=50
        ),
        html.Label(" Oversold"),
        dcc.Input(
            id='oversold-input',
            type='number',
            value=30
        ),
        html.Label(" Overbought"),
        dcc.Input(
            id='overbought-input',
            type='number',
            value=70
        ),
        html.Label(" RSI Window"),
        dcc.Input(
            id='rsi_window-input',
            type='number',
            value=14
        ),
        dcc.Graph(
            id='my_fig',
            style={'height': '100%'}
        )
    ],
    style={'height': '100vh'},
)

@app.callback(
    Output('my_fig', 'figure'),
    [Input('ticker-input', 'value'),
     Input('short_window-input', 'value'),
     Input('long_window-input', 'value'),
     Input('oversold-input', 'value'),
     Input('overbought-input', 'value'),
     Input('rsi_window-input', 'value')]
)

def update_graph(ticker, short_window, long_window, oversold, overbought, rsi_window):
    ticker=ticker
    short_window = short_window
    long_window = long_window
    oversold = oversold
    overbought = overbought
    rsi_window = rsi_window

    if ticker == "SPY":
        df_for_chart = spy_full.copy()
    else:
        df_for_chart = stocks_df.loc[stocks_df['ticker']==ticker].reset_index(drop=True)

    # Daily prices with moving averages and RSI
    df_for_chart['SMA_Short'] = df_for_chart['Adj Close'].rolling(window=short_window).mean()
    df_for_chart['SMA_Long'] = df_for_chart['Adj Close'].rolling(window=long_window).mean()
    df_for_chart['RSI'] = strat_defs.calculate_rsi(df_for_chart, ticker, 'Adj Close', window=rsi_window)

    fig_sub = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,row_heights=[0.7,0.3])

    fig_sub.add_trace(go.Scatter(x=df_for_chart['Date'], y=df_for_chart['Adj Close'],mode='lines', name=f'{ticker} Adj Close'), row=1, col=1)
    fig_sub.add_trace(go.Scatter(x=df_for_chart['Date'], y=df_for_chart['SMA_Short'],mode='lines', name=f'SMA {short_window}'), row=1, col=1)
    fig_sub.add_trace(go.Scatter(x=df_for_chart['Date'], y=df_for_chart['SMA_Long'],mode='lines', name=f'SMA {long_window}'), row=1, col=1)

    fig_sub.add_trace(go.Scatter(x=df_for_chart['Date'], y=df_for_chart['RSI'],mode='lines', name='RSI'), row=2, col=1)
    fig_sub.add_hline(y=oversold,line_dash="dash", line_color="green",label=dict(text=f'Oversold ({oversold})',textposition="end"), row=2, col=1)
    fig_sub.add_hline(y=overbought, line_dash="dash", line_color="red",label=dict(text=f'Overbought ({overbought})',textposition="end"), row=2, col=1)
    fig_sub.update_layout(title=f'Daily {ticker} Adj Close', legend=dict(yanchor="top",y=0.98, xanchor="left", x=0.01))

    return fig_sub


if __name__ == '__main__':
    app.run(debug=True)
