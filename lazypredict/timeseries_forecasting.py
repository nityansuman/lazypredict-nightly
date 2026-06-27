import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .regressor import LazyRegressor


def make_supervised_windows(series, lookback=5, horizon=1):
    """Convert a raw series into supervised learning windows."""
    values = np.asarray(series)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    features = []
    targets = []
    limit = len(values) - lookback - horizon + 1
    for index in range(limit):
        features.append(values[index : index + lookback].reshape(-1))
        targets.append(values[index + lookback : index + lookback + horizon].reshape(-1))

    X = np.asarray(features)
    y = np.asarray(targets)
    if horizon == 1:
        y = y.reshape(-1)

    return X, y


def evaluate_ts_forecasting(y_true, y_pred):
    """Compute standard forecasting metrics for time-series predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        metrics["MSE"] = mean_squared_error(y_true, y_pred)

    return metrics


class LazyTimeSeriesRegressor:
    """Benchmark regression models on supervised time-series windows."""

    def __init__(self, lookback=5, horizon=1, test_size=0.2, random_state=42, **regressor_kwargs):
        self.lookback = lookback
        self.horizon = horizon
        self.test_size = test_size
        self.random_state = random_state
        self.regressor_kwargs = regressor_kwargs
        self.models = {}

    def fit(self, series):
        """Fit all supported regressors on windowed time-series data."""
        X, y = make_supervised_windows(series, lookback=self.lookback, horizon=self.horizon)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=False
        )

        regressor = LazyRegressor(**self.regressor_kwargs)
        scores, predictions = regressor.fit(X_train, X_test, y_train, y_test)
        self.models = regressor.models
        return scores, predictions


class LazyTimeSeriesForecasting(LazyTimeSeriesRegressor):
    """Alias for LazyTimeSeriesRegressor for a more explicit API name."""
