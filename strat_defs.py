"""Define forecasting strategies"""

import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


 # Technical indicators
def calculate_rsi_wide(data, ticker, target, window):
    """
    Calculate the Relative Strength Index (RSI).
    
    Parameters:
        data (DataFrame): Stock data with target prices.
        ticker (str): Stock ticker
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

def calculate_rsi_long(data, target, window):
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

def calculate_vwap_wide(data, ticker, target):
    """
    Calculate Volume Weighted Average Price

    Parameters:
        data (DataFrame): Stock data with required columns.
        ticker (str): Stock ticker
        target (str): column to predict (usually Adj Close)

    Returns:
       
    """
    cumulative_volume = data["Volume_"+ticker].cumsum()
    cumulative_price_volume = (data[target+"_"+ticker] * data["Volume_"+ticker]).cumsum()
    return cumulative_price_volume / cumulative_volume

def calculate_vwap_long(data, target):
    """
    Calculate Volume Weighted Average Price

    Parameters:
        data (DataFrame): Stock data with required columns.
        target (str): column to predict (usually Adj Close)

    Returns:
       
    """
    cumulative_volume = data['Volume'].cumsum()
    cumulative_price_volume = (data[target] * data['Volume']).cumsum()
    return cumulative_price_volume / cumulative_volume

def calculate_technical_indicators(data, ticker, target, short_window, long_window,
                                   rsi_window, bollinger_window, bollinger_num_std):
    """
    Calculate technical indicators for the dataset.

    Parameters:
        data (DataFrame): Stock data with required columns.
        ticker (str): Stock ticker
        target (str): column to predict (usually Adj Close)
        short_window (int):
        long_window (int):
        rsi_window (int):
        bollinger_window (int):
        bollinger_num_std (flot):

    Returns:
        DataFrame: Data with strategy signals and portfolio value.
    """
    data['RSI'] = calculate_rsi_wide(data, ticker, target, window=rsi_window)
    data['MA_S'] = data[target+"_"+ticker].rolling(window=short_window).mean()
    data['MA_L'] = data[target+"_"+ticker].rolling(window=long_window).mean()
    data['MA_B'] = data[target+"_"+ticker].rolling(window=bollinger_window).mean()
    data['Bollinger_Upper'] = data['MA_B'] + bollinger_num_std * data[target+"_"+ticker].rolling(window=bollinger_window).std()
    data['Bollinger_Lower'] = data['MA_B'] - bollinger_num_std * data[target+"_"+ticker].rolling(window=bollinger_window).std()
    data['VWAP'] = calculate_vwap_wide(data, ticker, target)
    return data


# ML models
def strategy_prophet(data, initial_training_period, ticker, target):
    """
    Calculate forecast with Facebook Prophet  
    """
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)

    data_simp = data[['Date',target+"_"+ticker]]
    data_simp = data_simp.rename(columns={'Date': 'ds',target+"_"+ticker:'y'})

    for i in range(initial_training_period, len(data)):
        data_simp_cut = data_simp.iloc[:i]

        # Prophet object can only be fit once - must instantiate a new object every time
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(data_simp_cut)

        future = model.make_future_dataframe(periods=1, include_history=False)
        forecast = model.predict(future)

        predicted_price_tomorrow = forecast['yhat'].iloc[0]
        current_price = data.loc[data.index[i - 1], target+"_"+ticker]

        data.loc[data.index[i-1], 'predicted_price_tomorrow'] = predicted_price_tomorrow
        data.loc[data.index[i-1], 'Signal'] = 1 if predicted_price_tomorrow >= current_price else -1

    return data, model

def strategy_logit(data, initial_training_period, logit_proba, logit_max_iter, logit_c, n_jobs=None):
    """
    Calculate forecast with logistic regression  
    """
    model = LogisticRegression(C=logit_c, max_iter=logit_max_iter, n_jobs=n_jobs)
    scaler = StandardScaler()
    le = LabelEncoder()

    selected_features = [x for x in list(data) if x not in ['Date','Target','MA_B']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_training_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train = train_data[selected_features]
        y_train = le.fit_transform( train_data['Target'] )

        # Fit the scaler on the training data, scale training data and fit model
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        model.fit(X_train_scaled, y_train)

        # Predict for the next day
        prediction_end = min(i + 1, len(data))
        test_data = data.iloc[i:prediction_end]
        X_test = test_data[selected_features]

        # Scale test data using already fitted scaler
        X_test_scaled = scaler.transform(X_test)

        # store the probabilities for each class in separate columns
        pred_probs = model.predict_proba(X_test_scaled)
        for class_index, class_name in enumerate(le.classes_):
            probability_column = f"proba_logit_{class_name}"
            data.loc[data.index[i:prediction_end], probability_column] = pred_probs[:, class_index]

    data['Signal'] = np.where(data['proba_logit_-1.0'] > logit_proba, -1, 1)

    score = model.score(X_train_scaled, y_train)

    return data, model, score

def strategy_logit_pca(data, initial_training_period, logit_proba, logit_max_iter, logit_c, logit_pca_n_components, n_jobs=None):
    """
    Calculate forecast with logistic regression  
    """
    model = LogisticRegression(C=logit_c, max_iter=logit_max_iter, n_jobs=n_jobs)
    scaler = StandardScaler()
    le = LabelEncoder()
    pca = PCA(n_components=logit_pca_n_components)

    selected_features = [x for x in list(data) if x not in ['Date','Target','MA_B']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_training_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train = train_data[selected_features]
        y_train = le.fit_transform( train_data['Target'] )

        # Fit the scaler on the training data, scale training data and fit model
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)

        pca.fit(X_train_scaled)
        X_train_pca = pca.transform(X_train_scaled)
        model.fit(X_train_pca, y_train)

        # Predict for the next day
        prediction_end = min(i + 1, len(data))
        test_data = data.iloc[i:prediction_end]
        X_test = test_data[selected_features]

        # Scale test data using already fitted scaler
        X_test_scaled = scaler.transform(X_test)
        X_test_pca = pca.transform(X_test_scaled)

        # store the probabilities for each class in separate columns
        pred_probs = model.predict_proba(X_test_pca)
        for class_index, class_name in enumerate(le.classes_):
            probability_column = f"proba_logit_pca_{class_name}"
            data.loc[data.index[i:prediction_end], probability_column] = pred_probs[:, class_index]

    data['Signal'] = np.where(data['proba_logit_pca_-1.0'] > logit_proba, -1, 1)

    # score = model.score(X_test_pca, y_train)

    return data, model#, score

def strategy_random_forest(data, initial_training_period, random_state=None, njobs=None):
    """
    Calculate forecast with random forest  
    """
    model = RandomForestClassifier(random_state=random_state, n_jobs=njobs)

    selected_features = [x for x in list(data) if x not in ['Date','Target','MA_B']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    # Define features and target
    X = data[selected_features]
    y = data['Target']

    # Prepare columns
    data['Signal'] = 1

    for i in range(initial_training_period, len(data)):
        # Train only on past data up to the current point
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]

        # Train the model
        model.fit(X_train, y_train)

        # Predict for the next day
        prediction_end = min(i + 1, len(data))

        X_test = X.iloc[i:prediction_end]
        data.loc[data.index[i:prediction_end], 'Signal'] = model.predict(X_test)

    score = model.score(X_train, y_train)

    return data, model, score

def strategy_xgboost(data, initial_training_period, xgboost_proba, random_state=None, n_jobs=None):
    """
    Calculate forecast with XGBoost  
    """
    model = XGBClassifier(random_state=random_state, n_jobs=n_jobs)
    le = LabelEncoder()

    selected_features = [x for x in list(data) if x not in ['Date','Target','MA_B']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_training_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train = train_data[selected_features]
        y_train = le.fit_transform( train_data['Target'] )

        # Train the model
        model.fit(X_train, y_train)

        # Predict for the next day
        prediction_end = min(i + 1, len(data))
        test_data = data.iloc[i:prediction_end]
        X_test = test_data[selected_features]

        # store the probabilities for each class in separate columns
        pred_probs = model.predict_proba(X_test)
        for class_index, class_name in enumerate(le.classes_):
            probability_column = f"proba_xgboost_{class_name}"
            data.loc[data.index[i:prediction_end], probability_column] = pred_probs[:, class_index]

    data['Signal'] = np.where(data['proba_xgboost_-1.0'] > xgboost_proba, -1, 1)

    score = model.score(X_train, y_train)

    return data, model, score

def strategy_xgboost_scaled(data, initial_training_period, xgboost_proba, random_state=None, n_jobs=None):
    """
    Calculate forecast with XGBoost scaled
    """
    model = XGBClassifier(random_state=random_state, n_jobs=n_jobs)
    le = LabelEncoder()
    scaler = StandardScaler()

    selected_features = [x for x in list(data) if x not in ['Date','Target','MA_B']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_training_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train = train_data[selected_features]
        y_train = le.fit_transform( train_data['Target'] )

        # Fit the scaler on the training data
        scaler.fit(X_train)

        # Scale training data and fit model
        X_train_scaled = scaler.transform(X_train)

        model.fit(X_train_scaled, y_train)

        # Predict for the next day
        prediction_end = min(i + 1, len(data))
        test_data = data.iloc[i:prediction_end]
        X_test = test_data[selected_features]

        # Scale test data using already fitted scaler
        X_test_scaled = scaler.transform(X_test)

        # store the probabilities for each class in separate columns
        pred_probs = model.predict_proba(X_test_scaled)
        for class_index, class_name in enumerate(le.classes_):
            probability_column = f"proba_xgboost_scaled_{class_name}"
            data.loc[data.index[i:prediction_end], probability_column] = pred_probs[:, class_index]

    data['Signal'] = np.where(data['proba_xgboost_scaled_-1.0'] > xgboost_proba, -1, 1)

    score = model.score(X_train_scaled, y_train)

    return data, model, score

def strategy_mlp(data, initial_training_period, mlp_proba, mlp_max_iter, random_state=None):
    """
    Calculate forecast with MLP
    """
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=random_state, max_iter=mlp_max_iter)
    scaler = StandardScaler()
    le = LabelEncoder()

    selected_features = [x for x in list(data) if x not in ['Date','Target','MA_B']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_training_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train = train_data[selected_features]
        y_train = le.fit_transform( train_data['Target'] )

        # Fit the scaler on the training data
        scaler.fit(X_train)

        # Scale training data and fit model
        X_train_scaled = scaler.transform(X_train)

        model.fit(X_train_scaled, y_train)

        # Predict for the next day
        prediction_end = min(i + 1, len(data))
        test_data = data.iloc[i:prediction_end]
        X_test = test_data[selected_features]

        # Scale test data using already fitted scaler
        X_test_scaled = scaler.transform(X_test)

        # store the probabilities for each class in separate columns
        pred_probs = model.predict_proba(X_test_scaled)
        for class_index, class_name in enumerate(le.classes_):
            probability_column = f"proba_mlp_{class_name}"
            data.loc[data.index[i:prediction_end], probability_column] = pred_probs[:, class_index]

    data['Signal'] = np.where(data['proba_mlp_-1.0'] > mlp_proba, -1, 1)

    score = model.score(X_train, y_train)

    return data, model, score

def strategy_keras(data, initial_training_period, keras_proba, keras_sequence_length, random_state=None):
    """
    Calculate forecast with Keras
    """
    scaler = StandardScaler()

    selected_features = [x for x in list(data) if x not in ['Date','Target','MA_B']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    # Keras
    tf.random.set_seed(random_state) # seems like the seed is very influential... 
    sequence_length = keras_sequence_length  # Number of time steps (lookback window)

    model = models.Sequential([
        layers.Input(shape=(sequence_length, len(selected_features))),
        layers.LSTM(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    for i in range(initial_training_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train_0 = train_data[selected_features]
        y_train_0 = train_data['Target']

        # Fit the scaler on the training data
        scaler.fit(X_train_0)

        # Scale training data and fit model
        X_train_0_scaled = scaler.transform(X_train_0)

        # Select features and target
        X = []
        y = []
        for j in range(len(X_train_0_scaled) - sequence_length):
            X.append(X_train_0_scaled[j:j + sequence_length, :])
            y.append(y_train_0.iloc[j + sequence_length])

        X = np.array(X)  # Shape: (samples, time_steps, num_features)
        y = np.array(y)  # Shape: (samples,)

        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        X_train_1, y_train_1 = X[:train_size], y[:train_size]
        X_test_1, y_test_1 = X[train_size:], y[train_size:]

        # Step 3: Train the Model
        model.fit(X_train_1, y_train_1, epochs=20, batch_size=16, validation_data=(X_test_1, y_test_1), verbose=0)

        # Step 4: Predict the Next Day
        # Use the last `sequence_length` rows of all features as input
        last_sequence = X_train_0_scaled[-sequence_length:, :].reshape(1, sequence_length, len(selected_features))

        # Make the prediction
        next_day_prediction = model.predict(last_sequence, verbose=0)[0][0]
        sig_check = 0 if next_day_prediction < keras_proba else 1
        print(f"sig: {sig_check}; "
              f"len(X_train_0_scaled): {len(X_train_0_scaled)}; "
              f"max(train_data['Date']): {max(train_data['Date']).strftime('%Y-%m-%d')}; "
              f"next_day_pred: {next_day_prediction}")

        prediction_end = min(i + 1, len(data))

        data.loc[data.index[i:prediction_end], 'next_day_prediction'] = next_day_prediction

    data['Signal'] = np.where(data['next_day_prediction'] < keras_proba, -1, 1)

    return data, model

#
def backtest_strategy(data, ticker, initial_capital, strategy, target, short_window, long_window,
                      rsi_window, bollinger_window, bollinger_num_std, random_state=None, **kwargs):
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
        score: model score, if available
    """
    data_raw = data.copy()
    data = data.copy() # Prevent modifying the original DataFrame

    og_min_date = min(data_raw['Date'])

    model = None
    score = None

    data = calculate_technical_indicators(data, ticker, target, short_window, long_window,
                                          rsi_window, bollinger_window, bollinger_num_std)
    data['Target'] = np.sign(data[target+"_"+ticker].shift(-1) - data[target+"_"+ticker])
    data.loc[data['Target']==0,'Target']=1

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

    elif strategy == "Prophet":
        initial_training_period = kwargs.get('initial_training_period')
        data, model = strategy_prophet(data, initial_training_period, ticker, target)

    elif strategy == "Logit":
        initial_training_period = kwargs.get('initial_training_period')
        logit_proba = kwargs.get('logit_proba')
        logit_max_iter = kwargs.get('logit_max_iter')
        logit_c = kwargs.get('logit_c')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strategy_logit(data, initial_training_period, logit_proba, logit_max_iter, logit_c, n_jobs)

    elif strategy == "Logit_PCA":
        initial_training_period = kwargs.get('initial_training_period')
        logit_proba = kwargs.get('logit_proba')
        logit_max_iter = kwargs.get('logit_max_iter')
        logit_c = kwargs.get('logit_c')
        logit_pca_n_components = kwargs.get('logit_pca_n_components')
        n_jobs = kwargs.get('n_jobs')
        data, model = strategy_logit_pca(data, initial_training_period, logit_proba, logit_max_iter, logit_c, logit_pca_n_components, n_jobs)

    elif strategy == "RandomForest":
        initial_training_period = kwargs.get('initial_training_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strategy_random_forest(data, initial_training_period, random_state, n_jobs)

    elif strategy == "XGBoost":
        initial_training_period = kwargs.get('initial_training_period')
        xgboost_proba = kwargs.get('xgboost_proba')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strategy_xgboost(data, initial_training_period, xgboost_proba, random_state, n_jobs)

    elif strategy == "XGBoost_scaled":
        initial_training_period = kwargs.get('initial_training_period')
        xgboost_proba = kwargs.get('xgboost_proba')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strategy_xgboost_scaled(data, initial_training_period, xgboost_proba, random_state, n_jobs)

    elif strategy == "MLP":
        initial_training_period = kwargs.get('initial_training_period')
        mlp_proba = kwargs.get('mlp_proba')
        mlp_max_iter = kwargs.get('mlp_max_iter')
        data, model, score = strategy_mlp(data, initial_training_period, mlp_proba, mlp_max_iter, random_state)

    elif strategy == "Keras":
        initial_training_period = kwargs.get('initial_training_period')
        keras_proba = kwargs.get('keras_proba')
        keras_sequence_length = kwargs.get('keras_sequence_length')
        data, model = strategy_keras(data, initial_training_period, keras_proba, keras_sequence_length, random_state)

    elif strategy == 'Model of Models':
        # WIP
        pass

    elif strategy == "Perfection":
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

    return data, model, score
