import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


def calculate_rsiW(data, ticker, target, window):
    """
    Calculate the Relative Strength Index (RSI).
    
    Parameters:
        data (DataFrame): Stock data with target prices.
        ticker: Stock ticker
        target (str): column to predict (usually Adj Close)
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

def calculate_rsiL(data, target, window):
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

def calculate_vwapW(data, ticker, target):
    cumulative_volume = data["Volume_"+ticker].cumsum()
    cumulative_price_volume = (data[target+"_"+ticker] * data["Volume_"+ticker]).cumsum()
    return cumulative_price_volume / cumulative_volume

def calculate_vwapL(data, target):
    cumulative_volume = data['Volume'].cumsum()
    cumulative_price_volume = (data[target] * data['Volume']).cumsum()
    return cumulative_price_volume / cumulative_volume

def calculate_technical_indicators(data, ticker, target, short_window, long_window, rsi_window, bollinger_window, bollinger_num_std):
    """
    Calculate technical indicators for the dataset.

    Parameters:
        data (DataFrame): Stock data with required columns.
        ticker: Stock ticker
        target (str): column to predict (usually Adj Close)
        short_window (int):
        long_window (int):
        rsi_window (int):
        bollinger_window (int):
        bollinger_num_std (flot):

    Returns:
        DataFrame: Data with strategy signals and portfolio value.
    """
    data['RSI'] = calculate_rsiW(data, ticker, target, window=rsi_window)
    data['MA_S'] = data[target+"_"+ticker].rolling(window=short_window).mean()
    data['MA_L'] = data[target+"_"+ticker].rolling(window=long_window).mean()
    data['MA_B'] = data[target+"_"+ticker].rolling(window=bollinger_window).mean()
    data['Bollinger_Upper'] = data['MA_B'] + bollinger_num_std * data[target+"_"+ticker].rolling(window=bollinger_window).std()
    data['Bollinger_Lower'] = data['MA_B'] - bollinger_num_std * data[target+"_"+ticker].rolling(window=bollinger_window).std()
    data['VWAP'] = calculate_vwapW(data, ticker, target)
    return data

def backtest_strategy(data, ticker, initial_capital, strategy, target, short_window, long_window, rsi_window, bollinger_window, bollinger_num_std,
                      random_state=None, **kwargs):
    """
    Backtest various trading strategies.

    Parameters:
        data (DataFrame): Stock data with required columns.
        ticker: Stock ticker
        initial_capital (float): initial investment / starting capital
        strategy (str): The strategy name ('RSI', 'VWAP', 'Bollinger', etc.)
        target (str): column to predict (usually Adj Close)
        short_window (int):
        long_window (int):
        rsi_window (int):
        bollinger_window (int):
        bollinger_num_std (flot):
        random_state (int): 
        **kwargs: Additional parameters for some strategies

    Returns:
        DataFrame: Data with strategy signals and portfolio value.
        model: forcasting model, if available
    """
    data_raw = data.copy() 
    data = data.copy() # Prevent modifying the original DataFrame
    
    og_min_date = min(data_raw['Date'])

    selected_features = ([x for x in list(data_raw) if x not in ['Date']] + ['RSI','MA_S','MA_L','Bollinger_Upper','Bollinger_Lower','VWAP'])

    model = None

    data = calculate_technical_indicators(data, ticker, target, short_window, long_window, rsi_window, bollinger_window, bollinger_num_std)
    data['Target'] = np.sign(data[target+"_"+ticker].shift(-1) - data[target+"_"+ticker]) # price direction (1 = up, -1 = down, 0 = stable)

    # Strategies
    if strategy == "Hold":
        data['Signal'] = 1

    elif strategy == "SMA":
        # Generate signals: 1 = Buy, -1 = Sell, 0 = Hold
        data['Signal'] = 0
        data.loc[data['MA_S'] > data['MA_L'], 'Signal'] = 1
        data.loc[data['MA_S'] <= data['MA_L'], 'Signal'] = -1

    elif strategy == 'RSI':
        rsi_oversold = kwargs.get('rsi_oversold')
        rsi_overbought = kwargs.get('rsi_overbought')

        data['Signal'] = 0
        data.loc[data['RSI'] < rsi_oversold, 'Signal'] = 1
        data.loc[data['RSI'] > rsi_overbought, 'Signal'] = -1

    elif strategy == 'VWAP':
        data['Signal'] = 0
        data.loc[data[target+"_"+ticker] < data['VWAP'], 'Signal'] = 1  # Buy below VWAP
        data.loc[data[target+"_"+ticker] > data['VWAP'], 'Signal'] = -1  # Sell above VWAP

    elif strategy == 'Bollinger':
        data['Signal'] = 1
        data.loc[data[target+"_"+ticker] > data['Bollinger_Upper'], 'Signal'] = -1  # Sell
 
    elif strategy == 'Breakout':
        breakout_window = kwargs.get('breakout_window')

        data['High_Max'] = data['High_'+ticker].rolling(window=breakout_window).max().shift(1)
        data['Low_Min'] = data['Low_'+ticker].rolling(window=breakout_window).min().shift(1)
        data['Signal'] = 0
        data.loc[data[target+"_"+ticker] > data['High_Max'], 'Signal'] = 1  # Breakout above
        data.loc[data[target+"_"+ticker] < data['Low_Min'], 'Signal'] = -1  # Breakout below

    elif strategy == 'Prophet':
        initial_training_period = kwargs.get('initial_training_period')

        data_simp = data[['Date',target+"_"+ticker]]
        data_simp = data_simp.rename(columns={'Date': 'ds',target+"_"+ticker:'y'})
          
        # Prepare columns
        data['Signal'] = 0
        
        for i in range(initial_training_period, len(data)):
            data_simp_cut = data_simp.iloc[:i]

            # Initialize and fit Prophet
            model = Prophet(daily_seasonality=True, yearly_seasonality=True)
            model.fit(data_simp_cut)

            future = model.make_future_dataframe(periods=1, include_history=False)
            forecast = model.predict(future)

            predicted_price = forecast['yhat'].iloc[0]
            current_price = data.loc[data.index[i - 1], target+"_"+ticker]

            data.loc[data.index[i], 'Signal'] = 1 if predicted_price > current_price else -1

    elif strategy == "Logit":
        initial_training_period = kwargs.get('initial_training_period')
        retrain_interval = kwargs.get('retrain_interval')
        logit_max_iter = kwargs.get('logit_max_iter')
        logit_proba = kwargs.get('logit_proba')

        # Drop rows with missing values due to rolling calculations
        data = data.dropna().copy()
                
        model = LogisticRegression(max_iter=logit_max_iter)
        scaler = StandardScaler()
        le = LabelEncoder()

        # Prepare columns
        data['Signal'] = 1

        for i in range(initial_training_period, len(data), retrain_interval):
            # Train only on past data up to the current point
            train_data = data.iloc[:i]
            X_train = train_data[selected_features]
            y_train = le.fit_transform( train_data['Target'] )

            # Fit the scaler on the training data, scale training data and fit model
            scaler.fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            model.fit(X_train_scaled, y_train)

            # Predict for the next retrain_interval days
            prediction_end = min(i + retrain_interval, len(data))
            test_data = data.iloc[i:prediction_end]
            X_test = test_data[selected_features]

            # Scale test data using already fitted scaler
            X_test_scaled = scaler.transform(X_test)

            # store the probabilities for each class in separate columns
            predicted_probabilities = model.predict_proba(X_test_scaled)
            for class_index, class_name in enumerate(le.classes_):
                probability_column = f"proba_logit_{class_name}"
                data.loc[data.index[i:prediction_end], probability_column] = predicted_probabilities[:, class_index]
            
        data['Signal'] = np.where(data['proba_logit_-1.0'] > logit_proba, -1, 1)

    elif strategy =="RandomForest":
        initial_training_period = kwargs.get('initial_training_period')
        retrain_interval = kwargs.get('retrain_interval')
        
        # Drop rows with missing values due to rolling calculations
        data = data.dropna().copy()
        
        # Define features and target
        X = data[selected_features]
        y = data['Target']

        model = RandomForestClassifier(random_state=random_state)
        
        # Prepare columns
        data['Signal'] = 1

        for i in range(initial_training_period, len(data), retrain_interval):
            # Train only on past data up to the current point
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]

            # Train the model
            model.fit(X_train, y_train)

            # Predict for the next retrain_interval days
            prediction_end = min(i + retrain_interval, len(data))
            
            X_test = X.iloc[i:prediction_end]
            data.loc[data.index[i:prediction_end], 'Signal'] = model.predict(X_test)

    elif strategy == "XGBoost_scaled":
        initial_training_period = kwargs.get('initial_training_period')
        retrain_interval = kwargs.get('retrain_interval')
        
        # Drop rows with missing values due to rolling calculations
        data = data.dropna().copy()
        
        model = XGBClassifier(random_state=random_state)
        le = LabelEncoder()
        scaler = StandardScaler()
        
        # Prepare columns
        data['Signal'] = 1

        for i in range(initial_training_period, len(data), retrain_interval):
            # Train only on past data up to the current point
            train_data = data.iloc[:i]
            X_train = train_data[selected_features]
            y_train = le.fit_transform( train_data['Target'] )

            # Fit the scaler on the training data
            scaler.fit(X_train)

            # Scale training data and fit model
            X_train_scaled = scaler.transform(X_train)

            model.fit(X_train_scaled, y_train)

            # Predict for the next retrain_interval days
            prediction_end = min(i + retrain_interval, len(data))
            test_data = data.iloc[i:prediction_end]
            X_test = test_data[selected_features]

            # Scale test data using already fitted scaler
            X_test_scaled = scaler.transform(X_test)

            data.loc[data.index[i:prediction_end], 'Signal'] = le.inverse_transform(model.predict(X_test_scaled))

    elif strategy == "XGBoost":
        initial_training_period = kwargs.get('initial_training_period')
        retrain_interval = kwargs.get('retrain_interval')
        xgboost_proba = kwargs.get('xgboost_proba')
        
        # Drop rows with missing values due to rolling calculations
        data = data.dropna().copy()
        
        model = XGBClassifier(random_state=random_state)
        le = LabelEncoder()

        for i in range(initial_training_period, len(data), retrain_interval):
            # Train only on past data up to the current point
            train_data = data.iloc[:i]
            X_train = train_data[selected_features]
            y_train = le.fit_transform( train_data['Target'] )

            # Train the model
            model.fit(X_train, y_train)

            # Predict for the next retrain_interval days
            prediction_end = min(i + retrain_interval, len(data))
            test_data = data.iloc[i:prediction_end]
            X_test = test_data[selected_features]

            # store the probabilities for each class in separate columns
            predicted_probabilities = model.predict_proba(X_test)
            for class_index, class_name in enumerate(le.classes_):
                probability_column = f"proba_xgboost_{class_name}"
                data.loc[data.index[i:prediction_end], probability_column] = predicted_probabilities[:, class_index]
            
        data['Signal'] = np.where(data['proba_xgboost_-1.0'] > xgboost_proba, -1, 1)

    elif strategy == "MLP":
        initial_training_period = kwargs.get('initial_training_period')
        retrain_interval = kwargs.get('retrain_interval')
        mlp_proba = kwargs.get('mlp_proba')
        mlp_max_iter = kwargs.get('mlp_max_iter')
        
        # Drop rows with missing values due to rolling calculations
        data = data.dropna().copy()
        
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=random_state, max_iter=mlp_max_iter)
        scaler = StandardScaler()
        le = LabelEncoder()

        for i in range(initial_training_period, len(data), retrain_interval):
            # Train only on past data up to the current point
            train_data = data.iloc[:i]
            X_train = train_data[selected_features]
            y_train = le.fit_transform( train_data['Target'] )

            # Fit the scaler on the training data
            scaler.fit(X_train)

            # Scale training data and fit model
            X_train_scaled = scaler.transform(X_train)

            model.fit(X_train_scaled, y_train)

            # Predict for the next retrain_interval days
            prediction_end = min(i + retrain_interval, len(data))
            test_data = data.iloc[i:prediction_end]
            X_test = test_data[selected_features]

            # Scale test data using already fitted scaler
            X_test_scaled = scaler.transform(X_test)

            # store the probabilities for each class in separate columns
            predicted_probabilities = model.predict_proba(X_test_scaled)
            for class_index, class_name in enumerate(le.classes_):
                probability_column = f"proba_mlp_{class_name}"
                data.loc[data.index[i:prediction_end], probability_column] = predicted_probabilities[:, class_index]
            
        data['Signal'] = np.where(data['proba_mlp_-1.0'] > mlp_proba, -1, 1)

    elif strategy == 'Model of Models':
        # WIP
        pass

    elif strategy == 'Perfection':
        data['Signal'] = 1
        data.loc[data['Target']==-1, 'Signal'] = -1

    else:
        raise ValueError(f"Strategy '{strategy}' is not implemented.")
    
    # Stack on older data where had a training period, assume held stock during that time
    if min(data['Date']) != og_min_date:
        data_training_period = data_raw.loc[data_raw['Date']<min(data['Date'])].reset_index(drop=True)
        data_training_period['Signal']=1
        data = pd.concat([data_training_period,data])

    # Backtest logic: Calculate portfolio value
    signal_adj = []
    prev = 1
    for i in data['Signal']:
        if i==1:
            signal_adj.append(1)
            prev=1
        elif i==-1:
            signal_adj.append(0)
            prev=0
        else:
            signal_adj.append(prev)
            
    data['signal_adj'] = signal_adj
    
    data['Daily_Return'] = data[target+"_"+ticker].pct_change()
    data['Strategy_Return'] = data['signal_adj'].shift(1) * data['Daily_Return']
    data['Portfolio_Value'] = (1 + data['Strategy_Return']).cumprod() * initial_capital

    return data, model