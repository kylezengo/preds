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
class ProbaConfig:
    """
    Minimum probabilities for Signal = 1
    """
    # keras: float = 0.5
    gradb: float = 0.5
    knn: float = 0.5
    logit: float = 0.5
    mlp: float = 0.5
    rf: float = 0.5
    svc: float = 0.5
    xgboost: float = 0.5

@dataclass
class BacktestConfig:
    """
    Backtest configuration class
    """
    overbought: int = 70 # RSI overbought threshold
    bko_window: int = 20
    retrain_days: int = 1
    proba: ProbaConfig = field(default_factory=ProbaConfig)
    keras: KerasConfig = field(default_factory=KerasConfig)

# helper functions
def pred_loop(data, initial_train_period, best_pipeline, retrain_days) -> tuple:
    """
    Loop through the data and make predictions

    Parameters:
        data (DataFrame): Stock data with required columns
        initial_train_period (int): Initial training period
        best_pipeline: Trained pipeline
        retrain_days (int): Retrain the model every n days
    
    Returns:
        tuple:
            - data (DataFrame): Data with strategy signals
            - model: Trained model
            - score (float): Model accuracy score
    """
    feats = [col for col in data.columns if col not in ['Date', 'Target']]

    pred_results = []
    for i in range(initial_train_period, len(data)):
        # Retrain only every 'retrain_days' days or on the first iteration
        if (i - initial_train_period) % retrain_days == 0 or i == initial_train_period:
            # Train only on past data up to the current point
            train_data = data.iloc[:i]
            X_train, y_train = train_data[feats], train_data['Target']

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

def proba_loop(data, initial_train_period, best_pipeline, proba, retrain_days) -> tuple:
    """
    Loop through the data and predict probabilities, retraining the model every n days.

    Parameters:
        data (DataFrame): Stock data with required columns
        initial_train_period (int): Initial training period
        best_pipeline: Trained pipeline
        proba (float): Probability threshold for Signal = 1
        retrain_days (int): Retrain the model every n days

    Returns:
        tuple:
            - data (DataFrame): Data with strategy signals
            - model: Trained model
            - score (float): Model accuracy score
    """
    feats = [col for col in data.columns if col not in ['Date', 'Target']]

    proba_results = []
    for i in range(initial_train_period, len(data)):
        # Retrain only every 'retrain_days' days or on the first iteration
        if (i - initial_train_period) % retrain_days == 0 or i == initial_train_period:
            # Train only on past data up to the current point
            train_data = data.iloc[:i]
            X_train, y_train = train_data[feats], train_data['Target']

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

    model = best_pipeline.steps[-1][1]
    score = best_pipeline.score(X_train, y_train)

    return data, model, score

# sklearn models
def generic_sklearn_strategy(
    data, initial_train_period, model_cls, param_grid, retrain_days,
    proba_threshold=0.5, grid_search_n_jobs=None, **model_kwargs
):
    """
    Make predictions using a generic sklearn strategy.

    Parameters:
        data (DataFrame): Stock data with required columns
        initial_train_period (int): Initial training period
        model_cls: Sklearn model class to use (e.g., LogisticRegression, RandomForestClassifier)
        param_grid (dict): Parameter grid for GridSearchCV
        retrain_days (int): Retrain the model every n days
        proba_threshold (float): Probability threshold for Signal = 1
        random_state (int, optional): Random state for reproducibility
        n_jobs (int, optional): Number of parallel jobs for GridSearchCV

    Returns:
        tuple:
            - data (DataFrame): Data with strategy signals.
            - model: Trained logistic regression model.
            - score: Model accuracy score.
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
        model_cls(**model_kwargs)
    )

    search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(), n_jobs=grid_search_n_jobs)
    search.fit(X_train, y_train)

    # Auto-detect proba support if use_proba is None
    estimator = search.best_estimator_
    if hasattr(estimator.steps[-1][1], "predict_proba"):

        return proba_loop(
            data, initial_train_period, estimator, proba_threshold, retrain_days
        )

    return pred_loop(data, initial_train_period, estimator, retrain_days)

# Other models
def strat_keras(data, initial_train_period, config: KerasConfig, random_state=None):
    """
    Predict with Keras

    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        config: KerasConfig
        random_state (int, optional): Random state for reproducibility.

    Returns:
        DataFrame: Data with strategy signals.
        model: Trained Keras model.
    """
    feats = [col for col in data.columns if col not in ['Date', 'Target']]

    # Drop rows with missing values due to rolling calculations
    data = data.dropna().copy()

    scaler = StandardScaler()

    tf.random.set_seed(random_state) # seems like the seed is very influential...
    sequence_length = config.sequence_length  # Number of time steps (lookback window)

    model = models.Sequential([
        layers.Input(shape=(sequence_length, len(feats))),
        layers.LSTM(32, activation='relu', dropout=0.2, recurrent_dropout=0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    proba_results = []
    for i in range(initial_train_period, len(data)):
        # Train only on past data up to the current point
        train_data = data.iloc[:i]
        X_train_0 = train_data[feats]
        y_train_0 = train_data['Target']

        # Fit the scaler on the training data and scale training data
        X_train_0_scaled = scaler.fit_transform(X_train_0)

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
        model.fit(
            X_train_1, y_train_1,
            epochs=config.epochs, batch_size=16, validation_data=(X_test_1, y_test_1), verbose=0
        )

        # Predict the next day - use the last sequence_length rows of all features as input
        last_sequence = X_train_0_scaled[-sequence_length:, :].reshape(1, sequence_length,
                                                                       len(feats))

        # Make the prediction
        next_day_prediction = model.predict(last_sequence, verbose=0)[0][0]

        # Store predictions with indices
        proba_results.append((i, next_day_prediction))

        print(f"Date: {max(train_data['Date']).strftime('%Y-%m-%d')}; "
              f"next_day_pred: {next_day_prediction} "
              f"Sequence Percent 1: {sum(y_train_0[-sequence_length:])/sequence_length:.0%}")

        # data.loc[data.index[i], 'next_day_prediction'] = next_day_prediction

    proba_df = pd.DataFrame(proba_results,
                            columns=["index", "next_day_prediction"]).set_index("index")
    data["next_day_prediction"] = pd.DataFrame(proba_df["next_day_prediction"].to_list(),
                                               index=proba_df.index)

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

def strat_xgboost(
        data, initial_train_period, xgboost_proba, retrain_days, random_state=None, n_jobs=None
):
    """
    Predict probabilities with XGBoost
    
    Parameters:
        data (DataFrame): Stock data with required columns
        initial_train_period (int): Initial training period
        xgboost_proba (float): Probability threshold for Signal = 1
        retrain_days (int): Retrain the model every n days
        random_state (int, optional): Random state for reproducibility
        n_jobs (int, optional): Number of parallel jobs for XGBoost
    
    Returns:
        tuple:
            - data (DataFrame): Data with strategy signals
            - model: Trained XGBoost model
            - score (float): Model accuracy score
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

    return proba_loop(
        data, initial_train_period, search.best_estimator_, xgboost_proba, retrain_days
    )


# Backtest
def backtest_strategy(
        data, strategy, target, ticker, config: BacktestConfig, random_state=None, **kwargs
    ):
    """
    Backtest various trading strategies.

    Parameters:
        data (DataFrame): Stock data with required columns.
        strategy (str): The strategy name ('RSI', 'VWAP', 'Bollinger', etc.)
        target (str): Target variable to predict (e.g., 'Close')
        ticker (str): Ticker symbol for the stock.
        config: config info
        random_state (int): 
        **kwargs: Additional parameters for some strategies

    Returns:
        tuple:
            - date (DataFrame): Data with strategy signals and portfolio value
            - model: Forecasting model object used for predictions, if applicable
            - score (float): Model accuracy score, if applicable
    """
    data = data.copy() # Prevent modifying the original DataFrame

    model = score = None

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
        data['High_Max'] = data['High_'+ticker].rolling(window=config.bko_window).max().shift(1)
        data['Low_Min'] = data['Low_'+ticker].rolling(window=config.bko_window).min().shift(1)
        data['Signal'] = 1
        data.loc[data[target_ticker] < data['Low_Min'], 'Signal'] = 0

    # sklearn strategies
    elif strategy == "GradientBoosting":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        param_grid = {
            "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
        }
        data, model, score = generic_sklearn_strategy(
            data, initial_train_period, GradientBoostingClassifier, param_grid, config.retrain_days,
            proba_threshold=config.proba.gradb, grid_search_n_jobs=n_jobs,
            random_state=random_state # **model_kwargs
        )

    elif strategy == "KNN":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        param_grid = {
            "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
        }
        data, model, score = generic_sklearn_strategy(
            data, initial_train_period, KNeighborsClassifier, param_grid,
            config.retrain_days, proba_threshold=config.proba.knn, grid_search_n_jobs=n_jobs,
            n_jobs=n_jobs # **model_kwargs
        )

    elif strategy == "LinearSVC":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        param_grid = {
            "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
            "linearsvc__C": np.logspace(-4, 4, 9),
        }
        data, model, score = generic_sklearn_strategy(
            data, initial_train_period, LinearSVC, param_grid, config.retrain_days,
            grid_search_n_jobs=n_jobs,
            random_state=random_state # **model_kwargs
        )

    elif strategy == "Logit":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        param_grid = [
            {  # Case where solver is liblinear → NO n_jobs
                "pca__n_components": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                "logisticregression__C": np.logspace(-4, 4, 9),
                "logisticregression__solver": ["liblinear"],
                "logisticregression__max_iter": [100, 500, 1000],
            },
            {  # Case where solver is lbfgs or saga → USE n_jobs
                "pca__n_components": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                "logisticregression__C": np.logspace(-4, 4, 9),
                "logisticregression__solver": ["lbfgs", "saga"],
                "logisticregression__max_iter": [100, 500, 1000],
                "logisticregression__n_jobs": [n_jobs],  # Only set n_jobs for these solvers
            }
        ]
        data, model, score = generic_sklearn_strategy(
            data, initial_train_period, LogisticRegression, param_grid, config.retrain_days,
            proba_threshold=config.proba.logit, grid_search_n_jobs=n_jobs,
            random_state=random_state # **model_kwargs
        )

    elif strategy == "MLP":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        param_grid = {
            "pca__n_components": [0.6,0.7,0.8,0.9],
            "mlpclassifier__alpha": np.logspace(-5, 5, 11),
            "mlpclassifier__hidden_layer_sizes": [(32, 16), (64, 32, 16)],
            "mlpclassifier__max_iter": [100,500,1000,5000]
        }
        data, model, score = generic_sklearn_strategy(
            data, initial_train_period, MLPClassifier, param_grid,
            config.retrain_days, proba_threshold=config.proba.svc, grid_search_n_jobs=n_jobs,
            solver='lbfgs', random_state=random_state # **model_kwargs
        )

    elif strategy == "RandomForest":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        param_grid = {
            "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
        }
        data, model, score = generic_sklearn_strategy(
            data, initial_train_period, RandomForestClassifier, param_grid,
            config.retrain_days, proba_threshold=config.proba.rf, grid_search_n_jobs=n_jobs,
            random_state=random_state, n_jobs=n_jobs # **model_kwargs
        )

    elif strategy == "SVC":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        param_grid = {
            "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
            "svc__C": np.logspace(-4, 4, 9),
        }
        data, model, score = generic_sklearn_strategy(
            data, initial_train_period, SVC, param_grid, config.retrain_days,
            grid_search_n_jobs=n_jobs,
            random_state=random_state # **model_kwargs
        )

    elif strategy == "SVC_proba":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        param_grid = {
            "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95],
            "svc__C": np.logspace(-4, 4, 9),
            "svc__max_iter": [100,500,1000]
        }
        data, model, score = generic_sklearn_strategy(
            data, initial_train_period, SVC, param_grid,
            config.retrain_days, proba_threshold=config.proba.svc, grid_search_n_jobs=n_jobs,
            probability=True, random_state=random_state # **model_kwargs
        )

    #
    elif strategy == "Keras":
        initial_train_period = kwargs.get('initial_train_period')
        data, model = strat_keras(
            data, initial_train_period, config=config.keras, random_state=random_state
        )

    elif strategy == "Prophet":
        initial_train_period = kwargs.get('initial_train_period')
        data, model = strat_prophet(data, initial_train_period, target, ticker)

    elif strategy == "XGBoost":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_xgboost(
            data, initial_train_period, config.proba.xgboost, config.retrain_days,
            random_state, n_jobs
        )

    elif strategy == "Perfection":
        data['Signal'] = 1
        data.loc[data['Target']==-1, 'Signal'] = -1

    else:
        raise ValueError(f"Strategy '{strategy}' is not implemented.")

    data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']

    return data, model, score
