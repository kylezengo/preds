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
    logit_warm_start: bool = False
    proba: ProbaConfig = field(default_factory=ProbaConfig)
    keras: KerasConfig = field(default_factory=KerasConfig)

# helper functions
def pred_loop(data, initial_train_period, feats, best_pipeline, retrain_days) -> tuple:
    """
    Loop through the data and make predictions

    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        feats (list): List of features to use.
        best_pipeline: Trained pipeline.
        retrain_days (int): Retrain the model every n days.
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained model.
        score: Model accuracy score.
    """
    pred_results = []
    for i in range(initial_train_period, len(data)):
        # Retrain only every 'retrain_days' days or on the first iteration
        if (i - initial_train_period) % retrain_days == 0 or i == initial_train_period:
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

def proba_loop(data, initial_train_period, feats, best_pipeline, proba, retrain_days) -> tuple:
    """
    Loop through the data and predict probabilities, retraining the model every n days.

    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        feats (list): List of features to use.
        best_pipeline: Trained pipeline.
        proba (float): Probability threshold for Signal = 1.
        retrain_days (int): Retrain the model every n days.

    Returns:
        DataFrame: Data with strategy signals.
        model: Trained model.
        score: Model accuracy score.
    """
    proba_results = []
    for i in range(initial_train_period, len(data)):
        # Retrain only every 'retrain_days' days or on the first iteration
        if (i - initial_train_period) % retrain_days == 0 or i == initial_train_period:
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

    model = best_pipeline.steps[-1][1]
    score = best_pipeline.score(X_train, y_train)

    return data, model, score

#
def generic_sklearn_strategy(
    data, initial_train_period, model_cls, param_grid, retrain_days,
    use_proba=False, proba_threshold=0.5, random_state=None, n_jobs=None, **model_kwargs
):
    """
    Make predictions using a generic sklearn strategy.
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        model_cls: Sklearn model class to use (e.g., LogisticRegression, RandomForestClassifier).
        param_grid (dict): Parameter grid for GridSearchCV.
        retrain_days (int): Retrain the model every n days.
        use_proba (bool): Whether to use probability predictions.
        proba_threshold (float): Probability threshold for Signal = 1.
        random_state (int, optional): Random state for reproducibility.
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
        model_cls(random_state=random_state, **model_kwargs)
    )

    search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(), n_jobs=n_jobs)
    search.fit(X_train, y_train)

    if use_proba:
        return proba_loop(
            data, initial_train_period, feats, search.best_estimator_, proba_threshold, retrain_days
        )

    return pred_loop(data, initial_train_period, feats, search.best_estimator_, retrain_days)
# sklearn models
def strat_gradient_boost(data, initial_train_period, retrain_days, random_state=None, n_jobs=None):
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

    return pred_loop(data, initial_train_period, feats, search.best_estimator_, retrain_days)

def strat_knn(data, initial_train_period, knn_proba, retrain_days, n_jobs=None):
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

    return proba_loop(
        data, initial_train_period, feats, search.best_estimator_, knn_proba, retrain_days
    )

def strat_linear_svc(data, initial_train_period, retrain_days, random_state=None, n_jobs=None):
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

    return pred_loop(data, initial_train_period, feats, search.best_estimator_, retrain_days)

def strat_logit(data, initial_train_period, logit_proba, logit_warm_start, retrain_days, n_jobs=None):
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
        LogisticRegression(warm_start=logit_warm_start)
    )

    # Parameter grid with conditional n_jobs
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

    search = GridSearchCV(pipeline, param_grid, cv=TimeSeriesSplit(), n_jobs=n_jobs)
    search.fit(X_train, y_train)
    # print(search.best_params_)
    # print(search.best_estimator_.classes_)

    return proba_loop(
        data, initial_train_period, feats, search.best_estimator_, logit_proba, retrain_days
    )

def strat_mlp(data, initial_train_period, mlp_proba, retrain_days, random_state=None, n_jobs=None):
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

    return proba_loop(data, initial_train_period, feats, search.best_estimator_, mlp_proba, retrain_days)

def strat_random_forest(data, initial_train_period, rf_proba, retrain_days, random_state=None, n_jobs=None):
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

    return proba_loop(data, initial_train_period, feats, search.best_estimator_, rf_proba, retrain_days)

def strat_svc(data, initial_train_period, retrain_days, random_state=None, n_jobs=None):
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

    return pred_loop(data, initial_train_period, feats, search.best_estimator_, retrain_days)

def strat_svc_proba(data, initial_train_period, svc_proba, retrain_days, random_state=None, n_jobs=None):
    """
    Predict probabilities with SVC
    
    Parameters:
        data (DataFrame): Stock data with required columns.
        initial_train_period (int): Initial training period.
        svc_proba (float): Probability threshold for Signal = 1.
        random_state (int, optional): Random state for reproducibility.
        n_jobs (int, optional): Number of parallel jobs for GridSearchCV.
    
    Returns:
        DataFrame: Data with strategy signals.
        model: Trained SVC probability model.
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

    return proba_loop(data, initial_train_period, feats, search.best_estimator_, svc_proba, retrain_days)

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
        model.fit(X_train_1, y_train_1, epochs=config.epochs, batch_size=16,
                  validation_data=(X_test_1, y_test_1), verbose=0)

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

def strat_xgboost(data, initial_train_period, xgboost_proba, retrain_days, random_state=None, n_jobs=None):
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

    return proba_loop(
        data, initial_train_period, feats, search.best_estimator_, xgboost_proba, retrain_days
    )


# Backtest
def backtest_strategy(data, strategy, target, ticker, config: BacktestConfig,
                      random_state=None, **kwargs):
    """
    Backtest various trading strategies.

    Parameters:
        data (DataFrame): Stock data with required columns.
        strategy (str): The strategy name ('RSI', 'VWAP', 'Bollinger', etc.)
        config: config info
        random_state (int): 
        **kwargs: Additional parameters for some strategies

    Returns:
        tuple:
            - DataFrame: Data with strategy signals and portfolio value.
            - model: Forecasting model object used for predictions, if applicable.
            - score: Model accuracy score as a float, if applicable.
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
        data['High_Max'] = data['High_'+ticker].rolling(window=config.bko_window).max().shift(1)
        data['Low_Min'] = data['Low_'+ticker].rolling(window=config.bko_window).min().shift(1)
        data['Signal'] = 1
        data.loc[data[target_ticker] < data['Low_Min'], 'Signal'] = 0

    elif strategy == "Prophet":
        initial_train_period = kwargs.get('initial_train_period')
        data, model = strat_prophet(data, initial_train_period, target, ticker)

    elif strategy == "Logit":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_logit(
            data, initial_train_period, config.proba.logit, config.logit_warm_start,
            config.retrain_days, n_jobs=n_jobs
        )

    elif strategy == "RandomForest":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_random_forest(
            data, initial_train_period, config.proba.rf, config.retrain_days, random_state, n_jobs
        )

    elif strategy == "KNN":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_knn(
            data, initial_train_period, config.proba.knn, config.retrain_days, n_jobs
        )

    elif strategy == "GradientBoosting":
        initial_train_period = kwargs.get('initial_train_period')
        # data, model, score = strat_gradient_boost(
        #     data, initial_train_period, config.retrain_days,random_state
        # )

        param_grid = {
            "pca__n_components": [0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
        }
        data, model, score = generic_sklearn_strategy(
                data, initial_train_period, GradientBoostingClassifier, param_grid,
                config.retrain_days, use_proba=False, proba_threshold=0.5, random_state=None,
                n_jobs=None
        )

    elif strategy == "XGBoost":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_xgboost(
            data, initial_train_period, config.proba.xgboost, config.retrain_days,
            random_state, n_jobs
        )

    elif strategy == "SVC":
        initial_train_period = kwargs.get('initial_train_period')
        data, model, score = strat_svc(
            data, initial_train_period, config.retrain_days, random_state
        )

    elif strategy == "SVC_proba":
        initial_train_period = kwargs.get('initial_train_period')
        data, model, score = strat_svc_proba(
            data, initial_train_period, config.proba.svc, config.retrain_days, random_state
        )

    elif strategy == "LinearSVC":
        initial_train_period = kwargs.get('initial_train_period')
        data, model, score = strat_linear_svc(
            data, initial_train_period, config.retrain_days, random_state
        )

    elif strategy == "MLP":
        initial_train_period = kwargs.get('initial_train_period')
        n_jobs = kwargs.get('n_jobs')
        data, model, score = strat_mlp(
            data, initial_train_period, config.proba.mlp, config.retrain_days,
            random_state=random_state, n_jobs=n_jobs
        )

    elif strategy == "Keras":
        initial_train_period = kwargs.get('initial_train_period')
        data, model = strat_keras(
            data, initial_train_period, config=config.keras, random_state=random_state
        )

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
