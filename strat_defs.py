"""Define forecasting strategies"""

from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier


@dataclass
class LogitConfig:
    """
    Logistic Regression configuration class
    """
    max_iter: int = 1000
    proba: float = 0.5
    c: float = 0.01
    pca_n_components: float = 0.95

@dataclass
class MLPConfig:
    """
    MLP configuration class
    """
    max_iter: int = 1000
    proba: float = 0.5
    alpha: float = 1e-5
    hidden_layer_sizes: tuple = (32, 16)

@dataclass
class KerasConfig:
    """
    Keras configuration class
    """
    proba: float = 0.5
    sequence_length: int = 30
    epochs: int = 20

@dataclass
class BacktestConfig:
    """
    Backtest configuration class
    """
    overbought: int = 70
    xgboost_proba: float = 0.5
    svc_proba: float = 0.5
    logit: LogitConfig = field(default_factory=LogitConfig)
    mlp: MLPConfig = field(default_factory=MLPConfig)
    keras: KerasConfig = field(default_factory=KerasConfig)

# Models
def strat_prophet(data, initial_train_period, target, ticker):
    """
    Calculate forecast with Facebook Prophet  
    """
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)

    target_ticker = target+"_"+ticker

    data_simp = data[['Date',target_ticker]]
    data_simp = data_simp.rename(columns={'Date': 'ds',target_ticker:'y'})

    for i in range(initial_train_period, len(data)):
        data_simp_cut = data_simp.iloc[:i]

        # Prophet object can only be fit once - must instantiate a new object every time
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(data_simp_cut)

        future = model.make_future_dataframe(periods=1, include_history=False)
        forecast = model.predict(future)

        predicted_price_tomorrow = forecast['yhat'].iloc[0]
        current_price = data.loc[data.index[i - 1], target_ticker]

        data.loc[data.index[i-1], 'predicted_price_tomorrow'] = predicted_price_tomorrow
        data.loc[data.index[i-1], 'Signal'] = 1 if predicted_price_tomorrow >= current_price else 0
        # maybe also try if predicted_price_tomorrow > predicted_price_today

    return data, model

def strat_logit(data, initial_train_period, config: LogitConfig, n_jobs=None):
    """
    Calculate forecast with logistic regression
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained logistic regression model.
        score: Model accuracy score.
    """
    model = LogisticRegression(C=config.c, max_iter=config.max_iter, n_jobs=n_jobs)
    scaler = StandardScaler()
    le = LabelEncoder()

    selected_features = [x for x in list(data) if x not in ['Date','Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_train_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train = train_data[selected_features]
        y_train = le.fit_transform(train_data['Target'])

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
            probability_column = f"proba_{class_name}"
            data.loc[data.index[i:prediction_end], probability_column] = pred_probs[:, class_index]

    data['Signal'] = np.where(data['proba_1'].fillna(1) > config.proba, 1, 0)

    score = model.score(X_train_scaled, y_train)

    return data, model, score

def strat_logit_pca(data, initial_train_period, config: LogitConfig, n_jobs=None):
    """
    Calculate forecast with logistic regression  
    """
    model = LogisticRegression(C=config.c, max_iter=config.max_iter, n_jobs=n_jobs)
    scaler = StandardScaler()
    le = LabelEncoder()
    pca = PCA(n_components=config.pca_n_components)

    selected_features = [x for x in list(data) if x not in ['Date','Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_train_period, len(data)):
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
            probability_column = f"proba_{class_name}"
            data.loc[data.index[i:prediction_end], probability_column] = pred_probs[:, class_index]

    data['Signal'] = np.where(data['proba_1'].fillna(1) > config.proba, 1, 0)

    # ValueError: Found input variables with inconsistent numbers of samples: [2194, 1]
    # score = model.score(X_test_pca, y_train)

    return data, model#, score

def strat_random_forest(data, initial_train_period, random_state=None, njobs=None):
    """
    Calculate forecast with random forest  
    """
    model = RandomForestClassifier(random_state=random_state, n_jobs=njobs)

    selected_features = [x for x in list(data) if x not in ['Date','Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    # Define features and target
    X = data[selected_features]
    y = data['Target']

    data['Signal'] = 1
    for i in range(initial_train_period, len(data)):
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

def strat_gradient_boost(data, initial_train_period, random_state=None):
    """
    Calculate forecast with sklearn's GradientBoostingClassifier 
    Probably better to use XGBoost instead (much faster)
    """
    model = GradientBoostingClassifier(random_state=random_state)

    selected_features = [x for x in list(data) if x not in ['Date','Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    # Define features and target
    X = data[selected_features]
    y = data['Target']

    data['Signal'] = 1
    for i in range(initial_train_period, len(data)):
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

def strat_xgboost(data, initial_train_period, xgboost_proba, random_state=None, n_jobs=None):
    """
    Calculate forecast with XGBoost
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        xgboost_proba (float): Probability threshold for Signal = 1.
        random_state (int, optional): Random state for reproducibility.
        n_jobs (int, optional): Number of parallel jobs for XGBoost.
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained XGBoost model.
        score: Model accuracy score.
    """
    model = XGBClassifier(random_state=random_state, n_jobs=n_jobs)
    le = LabelEncoder()

    selected_features = [x for x in list(data) if x not in ['Date','Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_train_period, len(data)):
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
            probability_column = f"proba_{class_name}"
            data.loc[data.index[i:prediction_end], probability_column] = pred_probs[:, class_index]

    data['Signal'] = np.where(data['proba_1'].fillna(1) > xgboost_proba, 1, 0)

    score = model.score(X_train, y_train)

    return data, model, score

def strat_xgboost_scaled(data, initial_train_period, xgboost_proba, random_state=None, n_jobs=None):
    """
    Calculate forecast with XGBoost scaled
    """
    model = XGBClassifier(random_state=random_state, n_jobs=n_jobs)
    le = LabelEncoder()
    scaler = StandardScaler()

    selected_features = [x for x in list(data) if x not in ['Date','Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_train_period, len(data)):
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
            probability_column = f"proba_{class_name}"
            data.loc[data.index[i:prediction_end], probability_column] = pred_probs[:, class_index]

    data['Signal'] = np.where(data['proba_1'].fillna(1) > xgboost_proba, 1, 0)

    score = model.score(X_train_scaled, y_train)

    return data, model, score

def strat_svc(data, initial_train_period, random_state=None):
    """
    Calculate forecast with SVC
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained SVC model.
        score: Model accuracy score.
    """
    model = SVC(random_state=random_state)
    scaler = StandardScaler()
    le = LabelEncoder()

    selected_features = [x for x in list(data) if x not in ['Date','Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_train_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train = train_data[selected_features]
        y_train = le.fit_transform(train_data['Target'])

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

        # predict
        data.loc[data.index[i:prediction_end], 'Signal'] = model.predict(X_test_scaled)

    data['Signal'] = data['Signal'].fillna(1)

    score = model.score(X_train_scaled, y_train)
    score = model.score(X_train_scaled, y_train)

    return data, model, score

def strat_svc_proba(data, initial_train_period, svc_proba, random_state=None):
    """
    Calculate forecast with SVC
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained SVC model.
        score: Model accuracy score.
    """
    model = SVC(probability=True, random_state=random_state)
    scaler = StandardScaler()
    le = LabelEncoder()

    selected_features = [x for x in list(data) if x not in ['Date','Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_train_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train = train_data[selected_features]
        y_train = le.fit_transform(train_data['Target'])

        # Fit the scaler on the training data, scale training data and fit model
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        model.fit(X_train_scaled, y_train)

        # clf = make_pipeline(StandardScaler(), SVC(gamma='auto')) # switch to this?
        # clf.fit(X_train, y_train)

        # Predict for the next day
        prediction_end = min(i + 1, len(data))
        test_data = data.iloc[i:prediction_end]
        X_test = test_data[selected_features]

        # Scale test data using already fitted scaler
        X_test_scaled = scaler.transform(X_test)

        # store the probabilities for each class in separate columns
        pred_probs = model.predict_proba(X_test_scaled)
        for class_index, class_name in enumerate(le.classes_):
            probability_column = f"proba_{class_name}"
            data.loc[data.index[i:prediction_end], probability_column] = pred_probs[:, class_index]

    data['Signal'] = np.where(data['proba_1'].fillna(1) > svc_proba, 1, 0)

    score = model.score(X_train_scaled, y_train)

    return data, model, score

def strat_linear_svc(data, initial_train_period, random_state=None):
    """
    Calculate forecast with SVC
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained Linear SVC model.
        score: Model accuracy score.
    """
    model = LinearSVC(random_state=random_state)
    scaler = StandardScaler()
    le = LabelEncoder()

    selected_features = [x for x in list(data) if x not in ['Date','Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_train_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train = train_data[selected_features]
        y_train = le.fit_transform(train_data['Target'])

        # Fit the scaler on the training data, scale training data and fit model
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        model.fit(X_train_scaled, y_train)

        # clf = make_pipeline(StandardScaler(), SVC(gamma='auto')) # switch to this?
        # clf.fit(X_train, y_train)

        # Predict for the next day
        prediction_end = min(i + 1, len(data))
        test_data = data.iloc[i:prediction_end]
        X_test = test_data[selected_features]

        # Scale test data using already fitted scaler
        X_test_scaled = scaler.transform(X_test)

        # predict
        data.loc[data.index[i:prediction_end], 'Signal'] = model.predict(X_test_scaled)

    data['Signal'] = data['Signal'].fillna(1)

    score = model.score(X_train_scaled, y_train)

    return data, model, score

def strat_mlp(data, initial_train_period, config: MLPConfig, random_state=None):
    """
    Calculate forecast with MLP
    """
    model = MLPClassifier(solver='lbfgs',
                          alpha=config.alpha, hidden_layer_sizes=config.hidden_layer_sizes,
                          random_state=random_state,
                          max_iter=config.max_iter)
    scaler = StandardScaler()
    le = LabelEncoder()

    selected_features = [x for x in list(data) if x not in ['Date','Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    for i in range(initial_train_period, len(data)):
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
            probability_column = f"proba_{class_name}"
            data.loc[data.index[i:prediction_end], probability_column] = pred_probs[:, class_index]

    data['Signal'] = np.where(data['proba_1'].fillna(1) > config.proba, 1, 0)

    score = model.score(X_train, y_train)

    return data, model, score

def strat_keras(data, initial_train_period, config: KerasConfig, random_state=None):
    """
    Calculate forecast with Keras
    """
    scaler = StandardScaler()

    selected_features = [x for x in list(data) if x not in ['Date','Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    tf.random.set_seed(random_state) # seems like the seed is very influential...
    sequence_length = config.sequence_length  # Number of time steps (lookback window)

    model = models.Sequential([
        layers.Input(shape=(sequence_length, len(selected_features))),
        layers.LSTM(32, activation='relu', dropout=0.2, recurrent_dropout=0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    for i in range(initial_train_period, len(data)):
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
            y.append(y_train_0.iloc[j + sequence_length - 1])

        X = np.array(X)  # Shape: (samples, time_steps, num_features)
        y = np.array(y)  # Shape: (samples,)

        # Split into train and test sets
        train_size = int(len(X) * 0.8)
        X_train_1, y_train_1 = X[:train_size], y[:train_size]
        X_test_1, y_test_1 = X[train_size:], y[train_size:]

        # Train the Model
        model.fit(X_train_1, y_train_1, epochs=config.epochs, batch_size=16,
                  validation_data=(X_test_1, y_test_1), verbose=0)

        # Predict the next day - use the last sequence_length rows of all features as input
        last_sequence = X_train_0_scaled[-sequence_length:, :].reshape(1, sequence_length,
                                                                       len(selected_features))

        # Make the prediction
        next_day_prediction = model.predict(last_sequence, verbose=0)[0][0]
        sig_check = 0 if next_day_prediction < config.proba else 1
        print(f"sig: {sig_check}; "
              f"Date: {max(train_data['Date']).strftime('%Y-%m-%d')}; "
              f"next_day_pred: {next_day_prediction} "
              f"Percent 1: {sum(y_train_0[-sequence_length:])/sequence_length:.0%}")

        data.loc[data.index[i], 'next_day_prediction'] = next_day_prediction

    data['Signal'] = np.where(data['next_day_prediction'].fillna(1) > config.proba, 1, 0)

    return data, model


# Backtest
def backtest_strategy(data, initial_capital, strategy, target, ticker,
                      config: BacktestConfig, random_state=None, **kwargs):
    """
    Backtest various trading strategies.

    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_capital (float): initial investment / starting capital
        strategy (str): The strategy name ('RSI', 'VWAP', 'Bollinger', etc.)
        config: config info
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

    target_ticker = target+"_"+ticker

    # Strategies
    if strategy == "Hold":
        data['Signal'] = 1

    elif strategy == "SMA":
        data['Signal'] = 1
        data.loc[data['MA_S'] <= data['MA_L'], 'Signal'] = 0

    elif strategy == 'RSI':
        data['Signal'] = 1
        data.loc[data['RSI'] > config.overbought, 'Signal'] = 0

    elif strategy == 'VWAP':
        data['Signal'] = 1
        data.loc[data[target_ticker] > data['VWAP'], 'Signal'] = 0

    elif strategy == 'Bollinger':
        data['Signal'] = 1
        data.loc[data[target_ticker] > data['Bollinger_Upper'], 'Signal'] = 0

    elif strategy == 'Breakout':
        bko_window = kwargs.get('bko_window')

        data['High_Max'] = data['High_'+config.indicator.ticker].rolling(window=bko_window).max().shift(1)
        data['Low_Min'] = data['Low_'+config.indicator.ticker].rolling(window=bko_window).min().shift(1)
        data['Signal'] = 1
        data.loc[data[target_ticker] < data['Low_Min'], 'Signal'] = 0

    elif strategy == "Prophet":
        initial_train_period = kwargs.get('initial_train_period')
        data, model = strat_prophet(data, initial_train_period, config.indicator.ticker,
                                    config.indicator.target)

    elif strategy == "Logit":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_logit(data, initial_train_period, config=config.logit,
                                         n_jobs=n_jobs)

    elif strategy == "Logit_PCA":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model = strat_logit_pca(data, initial_train_period, config=config.logit,
                                      n_jobs=n_jobs)

    elif strategy == "RandomForest":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_random_forest(data, initial_train_period,
                                                 random_state, n_jobs)

    elif strategy == "GradientBoosting":
        initial_train_period = kwargs.get('initial_train_period')
        data, model, score = strat_gradient_boost(data, initial_train_period,random_state)

    elif strategy == "XGBoost":
        initial_train_period = kwargs.get('initial_train_period')
        xgboost_proba = kwargs.get('xgboost_proba')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_xgboost(data, initial_train_period,
                                           xgboost_proba, random_state, n_jobs)

    elif strategy == "XGBoost_scaled":
        initial_train_period = kwargs.get('initial_train_period')
        xgboost_proba = kwargs.get('xgboost_proba')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_xgboost_scaled(data, initial_train_period,
                                                  xgboost_proba, random_state, n_jobs)

    elif strategy == "SVC":
        initial_train_period = kwargs.get('initial_train_period')
        data, model, score = strat_svc(data, initial_train_period, random_state)

    elif strategy == "SVC_proba":
        initial_train_period = kwargs.get('initial_train_period')
        svc_proba = kwargs.get('svc_proba')
        data, model, score = strat_svc_proba(data, initial_train_period, svc_proba, random_state)

    elif strategy == "LinearSVC":
        initial_train_period = kwargs.get('initial_train_period')
        data, model, score = strat_linear_svc(data, initial_train_period, random_state)

    elif strategy == "MLP":
        initial_train_period = kwargs.get('initial_train_period')
        data, model, score = strat_mlp(data, initial_train_period, config=config.mlp,
                                       random_state=random_state)

    elif strategy == "Keras":
        initial_train_period = kwargs.get('initial_train_period')
        data, model = strat_keras(data, initial_train_period, config=config.keras,
                                  random_state=random_state)

    elif strategy == 'Model of Models':
        pass # WIP

    elif strategy == "Perfection":
        data['Signal'] = 1
        data.loc[data['Target']==-1, 'Signal'] = -1

    else:
        raise ValueError(f"Strategy '{strategy}' is not implemented.")

    # Stack on older data where had a training period, assume held stock during that time
    #   might not this need anymore since rolling calculations applied in prep_data
    if min(data['Date']) != og_min_date:
        data_train_period = data_raw.loc[data_raw['Date']<min(data['Date'])].reset_index(drop=True)
        data_train_period['Signal']=1
        data = pd.concat([data_train_period,data])

    data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']
    data['Portfolio_Value'] = (1 + data['Strategy_Return']).cumprod() * initial_capital

    return data, model, score
