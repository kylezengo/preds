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
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier


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
    knn_proba: float = 0.5
    logit_proba: float = 0.5
    mlp_proba: float = 0.5
    rf_proba: float = 0.5
    svc_proba: float = 0.5
    xgboost_proba: float = 0.5
    keras: KerasConfig = field(default_factory=KerasConfig)

# helper functions
def pred_loop(data, initial_train_period, feats, best_pipeline):
    """
    Loop through the data and make predictions

    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        feats (list): List of features to use.
        best_pipeline: Trained pipeline.
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained model.
        score: Model accuracy score.
    """
    pred_results = []
    for i in range(initial_train_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train = train_data[feats]
        y_train = train_data['Target']

        # Fit the pipeline (scaling + model training)
        best_pipeline.fit(X_train, y_train)

        # Predict for the next day
        test_data = data.loc[[i]]
        X_test = test_data[feats]

        # Predict using the pipeline (scales automatically)
        pred_results.append((i, best_pipeline.predict(X_test)[0]))

    pred_df = pd.DataFrame(pred_results, columns=["index", "Signal"]).set_index("index")
    data.loc[pred_df.index, "Signal"] = pred_df["Signal"]

    data['Signal'] = data['Signal'].fillna(1)

    score = best_pipeline.score(X_train, y_train)
    model = best_pipeline.steps[-1][1]

    return data, model, score

def proba_loop(data, initial_train_period, feats, best_pipeline, proba):
    """
    Loop through the data and predict probabilities

    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        feats (list): List of features to use.
        best_pipeline: Trained pipeline.
        proba (float): Probability threshold for Signal = 1.
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained model.
        score: Model accuracy score.
    """
    proba_results = []
    for i in range(initial_train_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train = train_data[feats]
        y_train = train_data['Target']

        # Fit the pipeline (scaling + model training)
        best_pipeline.fit(X_train, y_train)

        # Predict for the next day
        test_data = data.loc[[i]]
        X_test = test_data[feats]

        # Store predictions with indices
        proba_results.append((i, best_pipeline.predict_proba(X_test)[0]))

    proba_df = pd.DataFrame(proba_results, columns=["index", "proba"]).set_index("index")
    data[["proba_0", "proba_1"]] = pd.DataFrame(proba_df["proba"].to_list(), index=proba_df.index)

    data['Signal'] = np.where(data['proba_1'].fillna(1) > proba, 1, 0)

    score = best_pipeline.score(X_train, y_train)
    model = best_pipeline.steps[-1][1]

    return data, model, score


# sklearn models
def strat_gradient_boost(data, initial_train_period, random_state=None, n_jobs=None):
    """
    Predict with sklearn's GradientBoostingClassifier 
    Probably better to use XGBoost instead (much faster)
    """
    feats = [col for col in data.columns if col not in ['Date', 'Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    train_data = data.iloc[:initial_train_period]
    X_train, y_train = train_data[feats], train_data['Target']

    # Grid search for best parameters
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(svd_solver='full'),
        GradientBoostingClassifier(random_state=random_state)
    )

    param_grid = {
        "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
    }

    search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(), n_jobs=n_jobs)
    search.fit(X_train, y_train)
    # print(search.best_params_)

    return pred_loop(data, initial_train_period, feats, search.best_estimator_)

def strat_knn(data, initial_train_period, knn_proba, n_jobs=None):
    """
    Predict probabilities with K nearest neighbors classifier
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        knn_config:
        n_jobs (int, optional): Number of parallel jobs for GridSearchCV.

    Returns:
        DataFrame: Data with strategy signals.
        model: Trained K nearest neighbors model.
        score: Model accuracy score.
    """
    feats = [col for col in data.columns if col not in ['Date', 'Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    train_data = data.iloc[:initial_train_period]
    X_train, y_train = train_data[feats], train_data['Target']

    # Grid search for best parameters
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(svd_solver='full'),
        KNeighborsClassifier(n_jobs=n_jobs)
    )

    param_grid = {
        "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
    }

    search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(), n_jobs=n_jobs)
    search.fit(X_train, y_train)
    # print(search.best_params_)

    return proba_loop(data, initial_train_period, feats, search.best_estimator_, knn_proba)

def strat_linear_svc(data, initial_train_period, random_state=None, n_jobs=None):
    """
    Predict with Linear SVC
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained Linear SVC model.
        score: Model accuracy score.
    """
    feats = [col for col in data.columns if col not in ['Date', 'Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    train_data = data.iloc[:initial_train_period]
    X_train, y_train = train_data[feats], train_data['Target']

    # Grid search for best parameters
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(svd_solver='full'),
        LinearSVC(random_state=random_state)
    )

    train_data = data.iloc[:initial_train_period]
    X_train = train_data.drop(columns=['Date', 'Target'])
    y_train = train_data['Target']

    param_grid = {
        "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
        "linearsvc__C": np.logspace(-4, 4, 9),
    }

    search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(), n_jobs=n_jobs)
    search.fit(X_train, y_train)
    # print(search.best_params_)

    return pred_loop(data, initial_train_period, feats, search.best_estimator_)

def strat_logit(data, initial_train_period, logit_proba, n_jobs=None):
    """
    Predict probabilities with logistic regression
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        logit_proba (float): Probability threshold for Signal = 1.
        n_jobs (int, optional): Number of parallel jobs for GridSearchCV.

    Returns:
        DataFrame: Data with strategy signals.
        model: Trained logistic regression model.
        score: Model accuracy score.
    """
    feats = [col for col in data.columns if col not in ['Date', 'Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    train_data = data.iloc[:initial_train_period]
    X_train, y_train = train_data[feats], train_data['Target']

    # Grid search for best parameters
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(svd_solver='full'),
        LogisticRegression(n_jobs=n_jobs)
    )

    param_grid = {
        "pca__n_components": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        "logisticregression__C": np.logspace(-4, 4, 9),
        "logisticregression__solver": ["lbfgs", "liblinear", "saga"],
        "logisticregression__max_iter": [100, 500, 1000]
    }

    search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(), n_jobs=n_jobs)
    search.fit(X_train, y_train)
    # print(search.best_params_)

    return proba_loop(data, initial_train_period, feats, search.best_estimator_, logit_proba)

def strat_mlp(data, initial_train_period, mlp_proba, random_state=None, n_jobs=None):
    """
    Predict probabilities with MLP classifier
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        config: LogitConfig
        n_jobs (int, optional): Number of parallel jobs for GridSearchCV.

    Returns:
        DataFrame: Data with strategy signals.
        model: Trained MLP classifier model.
        score: Model accuracy score.
    """
    feats = [col for col in data.columns if col not in ['Date', 'Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    train_data = data.iloc[:initial_train_period]
    X_train, y_train = train_data[feats], train_data['Target']

    # Grid search for best parameters
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(svd_solver='full'),
        MLPClassifier(solver='lbfgs',
                      random_state=random_state,)
    )

    param_grid = {
        "pca__n_components": [0.6,0.7,0.8,0.9],
        "mlpclassifier__alpha": np.logspace(-5, 5, 11),
        "mlpclassifier__hidden_layer_sizes": [(32, 16), (64, 32, 16)],
        "mlpclassifier__max_iter": [100,500,1000,5000]
    }

    search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(), n_jobs=n_jobs)
    search.fit(X_train, y_train)
    # print(search.best_params_)

    return proba_loop(data, initial_train_period, feats, search.best_estimator_, mlp_proba)

def strat_random_forest(data, initial_train_period, rf_proba, random_state=None, n_jobs=None):
    """
    Predict probabilities with Random Forest Classifier
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        random_state (int, optional): Random state for reproducibility.
        njobs (int, optional): Number of parallel jobs for Random Forest.
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained SVC model.
        score: Model accuracy score.
    """
    pipeline = make_pipeline(
        RandomForestClassifier(random_state=random_state,
                               n_jobs=n_jobs)
    )

    feats = [col for col in data.columns if col not in ['Date', 'Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    train_data = data.iloc[:initial_train_period]
    X_train, y_train = train_data[feats], train_data['Target']

    # Grid search for best parameters
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(svd_solver='full'),
        RandomForestClassifier(random_state=random_state,
                               n_jobs=n_jobs)
    )

    param_grid = {
        "pca__n_components": [0.6,0.7,0.8,0.9],
    }

    search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(), n_jobs=n_jobs)
    search.fit(X_train, y_train)
    # print(search.best_params_)

    return proba_loop(data, initial_train_period, feats, search.best_estimator_, rf_proba)

def strat_svc(data, initial_train_period, random_state=None, n_jobs=None):
    """
    Predict with SVC
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained SVC model.
        score: Model accuracy score.
    """
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(random_state=random_state)
    )

    feats = [col for col in data.columns if col not in ['Date', 'Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    train_data = data.iloc[:initial_train_period]
    X_train, y_train = train_data[feats], train_data['Target']

    # Grid search for best parameters
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(svd_solver='full'),
        SVC(random_state=random_state)
    )

    train_data = data.iloc[:initial_train_period]
    X_train = train_data.drop(columns=['Date', 'Target'])
    y_train = train_data['Target']

    param_grid = {
        "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
        "svc__C": np.logspace(-4, 4, 9),
    }

    search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(), n_jobs=n_jobs)
    search.fit(X_train, y_train)
    # print(search.best_params_)

    return pred_loop(data, initial_train_period, feats, search.best_estimator_)

def strat_svc_proba(data, initial_train_period, svc_proba, random_state=None, n_jobs=None):
    """
    Predict probabilities with SVC
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        random_state (int, optional): Random state for reproducibility.
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained SVC probability model.
        score: Model accuracy score.
    """
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(probability=True, random_state=random_state)
    )

    feats = [col for col in data.columns if col not in ['Date', 'Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    train_data = data.iloc[:initial_train_period]
    X_train, y_train = train_data[feats], train_data['Target']

    # Grid search for best parameters
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(svd_solver='full'),
        SVC(probability=True,
            random_state=random_state)
    )

    param_grid = {
        "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
        "svc__C": np.logspace(-4, 4, 9),
        "svc__max_iter": [100,500,1000]
    }

    search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(), n_jobs=n_jobs)
    search.fit(X_train, y_train)
    # print(search.best_params_)

    return proba_loop(data, initial_train_period, feats, search.best_estimator_, svc_proba)

# Other models
def strat_keras(data, initial_train_period, config: KerasConfig, random_state=None):
    """
    Predict with Keras
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

def strat_prophet(data, initial_train_period, target, ticker):
    """
    Predict with Facebook Prophet  
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

def strat_xgboost(data, initial_train_period, xgboost_proba, random_state=None, n_jobs=None):
    """
    Predict probabilities with XGBoost
    
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
    feats = [col for col in data.columns if col not in ['Date', 'Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    train_data = data.iloc[:initial_train_period]
    X_train, y_train = train_data[feats], train_data['Target']

    # Grid search for best parameters
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(svd_solver='full'),
        XGBClassifier(random_state=random_state, n_jobs=n_jobs)
    )

    param_grid = {
        "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
    }

    search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(), n_jobs=n_jobs)
    search.fit(X_train, y_train)
    # print(search.best_params_)

    return proba_loop(data, initial_train_period, feats, search.best_estimator_, xgboost_proba)


# Backtest
def backtest_strategy(data, strategy, target, ticker, config: BacktestConfig, random_state=None, **kwargs):
    """
    Backtest various trading strategies.

    Parameters:
        data (DataFrame): Stock data with required columns.
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
        data, model, score = strat_logit(data, initial_train_period, config.logit_proba,
                                         n_jobs=n_jobs)

    elif strategy == "RandomForest":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_random_forest(data, initial_train_period, config.rf_proba,
                                                 random_state, n_jobs)

    elif strategy == "KNN":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_knn(data, initial_train_period, config.knn_proba, n_jobs)

    elif strategy == "GradientBoosting":
        initial_train_period = kwargs.get('initial_train_period')
        data, model, score = strat_gradient_boost(data, initial_train_period,random_state)

    elif strategy == "XGBoost":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_xgboost(data, initial_train_period,config.xgboost_proba,
                                           random_state, n_jobs)

    elif strategy == "SVC":
        initial_train_period = kwargs.get('initial_train_period')
        data, model, score = strat_svc(data, initial_train_period, random_state)

    elif strategy == "SVC_proba":
        initial_train_period = kwargs.get('initial_train_period')
        data, model, score = strat_svc_proba(data, initial_train_period, config.svc_proba,
                                             random_state)

    elif strategy == "LinearSVC":
        initial_train_period = kwargs.get('initial_train_period')
        data, model, score = strat_linear_svc(data, initial_train_period, random_state)

    elif strategy == "MLP":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_mlp(data, initial_train_period, config.mlp_proba,
                                       random_state=random_state, n_jobs=n_jobs)

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

    if ticker != "SPY":
        initial_train_period = kwargs.get('initial_train_period')
        data.loc[:initial_train_period, 'Strategy_Return'] = data['Daily_Return_SPY']
        data.loc[0, 'Strategy_Return'] = np.nan

    return data, model, score
