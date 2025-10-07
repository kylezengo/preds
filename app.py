"""Create an interactive plot in a browser window"""

import glob

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, Output, Input

import prep_data # custom functions

app = Dash()

# Load data
stocks_df_files = glob.glob('data/stocks_df_*.csv')
stocks_df_latest = max(stocks_df_files, key=lambda f: f.split("_")[2])
stocks_df = pd.read_csv(stocks_df_latest, parse_dates=['Date'])


# App layout
app.layout = html.Div([
    html.Div([
        html.Label("Ticker"),
        dcc.Input(
            id='ticker-input',
            value='SPY'
        ),
        html.Label(" Short Window"),
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
        html.Label(" Bollinger moving average"),
        dcc.Input(
            id='bma-input',
            type='number',
            value=20
        ),
        html.Label(" Bollinger num Std"),
        dcc.Input(
            id='bstd-input',
            type='number',
            value=2
        ),
        ], style={'fontFamily': 'Arial', 'width': '200px', 'padding': '20px',
                  'display': 'flex',  'flexDirection': 'column', 'flexShrink': 0}),
        html.Div([
            dcc.Graph(id='my_fig', style={'width': '100%', 'height': '100%'})
        ], style={'flexGrow': 1, 'padding': '20px','height': '100vh'})
], style={'display': 'flex', 'alignItems': 'flex-start','height': '100vh'})

@app.callback(
    Output('my_fig', 'figure'),
    Input('ticker-input', 'value'),
    Input('short_window-input', 'value'),
    Input('long_window-input', 'value'),
    Input('oversold-input', 'value'),
    Input('overbought-input', 'value'),
    Input('rsi_window-input', 'value'),
    Input('bma-input', 'value'),
    Input('bstd-input', 'value')
)

def update_graph(
    ticker, short_window, long_window, oversold, overbought, rsi_window,
    bollinger_window, bollinger_num_std
):
    """
    Update graph
    """
    chart_df = stocks_df.loc[stocks_df['ticker']==ticker].reset_index(drop=True)

    # Daily prices with moving averages and RSI
    chart_df['SMA_Short'] = chart_df['Adj Close'].rolling(window=short_window).mean()
    chart_df['SMA_Long'] = chart_df['Adj Close'].rolling(window=long_window).mean()
    chart_df['RSI'] = prep_data.calculate_rsi_long(chart_df, 'Adj Close', window=rsi_window)

    chart_df['MA_B'] = chart_df['Adj Close'].rolling(window=bollinger_window).mean()
    chart_df['Bollinger_Upper'] = (chart_df['MA_B'] +
                                   bollinger_num_std *
                                   chart_df['Adj Close'].rolling(window=bollinger_window).std())
    chart_df['Bollinger_Lower'] = (chart_df['MA_B'] -
                                   bollinger_num_std *
                                   chart_df['Adj Close'].rolling(window=bollinger_window).std())

    fig_sub = make_subplots(rows=2, cols=1,
                            shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.7,0.3])

    fig_sub.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['Bollinger_Upper'],mode='lines',
                                 line_color='rgba(177, 208, 252, 0.9)',
                                 name='Bollinger Upper'), row=1, col=1)
    fig_sub.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['Bollinger_Lower'],mode='lines',
                                 line_color='rgba(177, 208, 252, 0.9)', fill='tonexty',
                                 name='Bollinger Lower'), row=1, col=1)
    fig_sub.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['MA_B'],mode='lines',
                                 line_color='rgba(177, 208, 252, 0.9)',
                                 name='Bollinger moving average'), row=1, col=1)

    fig_sub.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['Adj Close'],mode='lines',
                                 name=f'{ticker} Adj Close'), row=1, col=1)
    fig_sub.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['SMA_Short'],mode='lines',
                                 name=f'SMA {short_window}'), row=1, col=1)
    fig_sub.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['SMA_Long'],mode='lines',
                                 name=f'SMA {long_window}'), row=1, col=1)

    fig_sub.add_trace(go.Scatter(x=chart_df['Date'], y=chart_df['RSI'],mode='lines',name='RSI'),
                      row=2, col=1)
    fig_sub.add_hline(y=oversold,line_dash="dash", line_color="green",
                      label={'text':f'Oversold ({oversold})','textposition':"end"},
                      row=2, col=1)
    fig_sub.add_hline(y=overbought, line_dash="dash", line_color="red",
                      label={'text':f'Overbought ({overbought})','textposition':"end"},
                      row=2, col=1)

    # fig_sub.update_layout(title=f'Daily {ticker} Adj Close',
    #                       legend={'yanchor':"top",'y': 0.98,'xanchor':"left",'x':0.01})

    fig_sub.update_layout(title=f'Daily {ticker} Adj Close')

    return fig_sub


if __name__ == '__main__':
    app.run(debug=True)
