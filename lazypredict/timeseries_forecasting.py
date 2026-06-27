import numpy as np
import pandas as pd

from .regressor import LazyRegressor
from .utils.time_series_metrics import TimeSeriesMetrics


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


def chronological_holdout_split(X, y, test_size=0.2):
    """Split windows without shuffling so future samples stay in the test set."""
    test_count = max(1, int(len(X) * test_size))
    if test_count >= len(X):
        raise ValueError("test_size leaves no training samples")
    train_end = len(X) - test_count
    return X[:train_end], X[train_end:], y[:train_end], y[train_end:]


def walk_forward_splits(X, y, n_splits=3, test_size=0.2):
    """Generate expanding-window walk-forward splits for backtesting."""
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    window_count = len(X)
    test_count = max(1, int(window_count * test_size))
    if test_count >= window_count:
        raise ValueError("test_size leaves no samples for walk-forward splitting")

    step = max(1, (window_count - test_count) // n_splits)
    for split_index in range(n_splits):
        train_end = max(1, step * (split_index + 1))
        test_end = min(window_count, train_end + test_count)
        if train_end >= test_end:
            continue
        yield X[:train_end], X[train_end:test_end], y[:train_end], y[train_end:test_end]


class LazyTimeSeriesRegressor:
    """Benchmark regression models on supervised time-series windows."""

    def __init__(self, lookback=5, horizon=1, test_size=0.2, random_state=42, split_strategy="holdout", n_splits=3, **regressor_kwargs):
        self.lookback = lookback
        self.horizon = horizon
        self.test_size = test_size
        self.random_state = random_state
        self.split_strategy = split_strategy
        self.n_splits = n_splits
        self.regressor_kwargs = regressor_kwargs
        self.models = {}

    def fit(self, series):
        """Fit all supported regressors on windowed time-series data."""
        X, y = make_supervised_windows(series, lookback=self.lookback, horizon=self.horizon)
        X_train, X_test, y_train, y_test = chronological_holdout_split(X, y, test_size=self.test_size)

        regressor = LazyRegressor(**self.regressor_kwargs)
        scores, predictions = regressor.fit(X_train, X_test, y_train, y_test)
        self.models = regressor.models
        return scores, predictions

    def backtest(self, series):
        """Run walk-forward backtesting and average the benchmark scores across splits."""
        X, y = make_supervised_windows(series, lookback=self.lookback, horizon=self.horizon)
        split_scores = []
        for X_train, X_test, y_train, y_test in walk_forward_splits(
            X, y, n_splits=self.n_splits, test_size=self.test_size
        ):
            regressor = LazyRegressor(**self.regressor_kwargs)
            scores, _ = regressor.fit(X_train, X_test, y_train, y_test)
            split_scores.append(scores)
            self.models.update(regressor.models)

        if not split_scores:
            raise ValueError("No valid walk-forward splits were generated")

        combined = pd.concat(split_scores).reset_index()
        grouped = combined.groupby("Model", as_index=False).mean(numeric_only=True).set_index("Model")
        return grouped.sort_values(by="Adjusted R-Squared", ascending=False)


class LazyTimeSeriesForecasting(LazyTimeSeriesRegressor):
    """Alias for LazyTimeSeriesRegressor for a more explicit API name."""
