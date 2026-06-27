import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from .supervised import LazyClassifier, LazyRegressor


def make_supervised_windows(series, lookback=5, horizon=1):
    """Convert a raw series into supervised learning windows.

    Parameters
    ----------
    series : array-like
        Univariate or multivariate time series.
    lookback : int, optional
        Number of past steps to use as features.
    horizon : int, optional
        Number of future steps to predict.

    Returns
    -------
    X : numpy.ndarray
        Windowed feature matrix.
    y : numpy.ndarray
        Windowed target array.
    """
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


def make_classification_windows(series, labels, lookback=5, horizon=1):
    """Create aligned feature and label windows for time-series classification.

    Parameters
    ----------
    series : array-like
        Time-series input used to build feature windows.
    labels : array-like
        Classification targets aligned to the series.
    lookback : int, optional
        Number of past steps to use as features.
    horizon : int, optional
        Number of future steps per target window.

    Returns
    -------
    features : numpy.ndarray
        Windowed feature matrix.
    y : numpy.ndarray
        Windowed classification targets.
    """
    features, _ = make_supervised_windows(series, lookback=lookback, horizon=horizon)

    label_values = np.asarray(labels)
    if len(label_values) < lookback + horizon:
        raise ValueError("labels must be long enough for the requested lookback and horizon")

    label_windows = []
    limit = len(label_values) - lookback - horizon + 1
    for index in range(limit):
        label_windows.append(label_values[index + lookback : index + lookback + horizon])

    y = np.asarray(label_windows)
    if horizon == 1:
        y = y.reshape(-1)

    return features, y


def evaluate_ts_classification(y_true, y_pred):
    """Compute standard classification metrics for time-series predictions.

    Returns accuracy, balanced accuracy, and weighted F1 score.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
    }


def evaluate_ts_forecasting(y_true, y_pred):
    """Compute standard forecasting metrics for time-series predictions.

    Returns MAE, RMSE, and R2. For multi-step outputs, MSE is also included.
    """
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


class LazyTimeSeriesClassifier:
    """Benchmark classification models on supervised time-series windows."""

    def __init__(
        self,
        lookback=5,
        horizon=1,
        test_size=0.2,
        random_state=42,
        **classifier_kwargs,
    ):
        """Store the windowing configuration and classifier options.

        Parameters
        ----------
        lookback : int, optional
            Number of historical steps used per sample.
        horizon : int, optional
            Number of future steps per target window.
        test_size : float, optional
            Fraction of samples reserved for evaluation.
        random_state : int, optional
            Random seed used by the internal train/test split.
        **classifier_kwargs : dict
            Additional keyword arguments forwarded to LazyClassifier.
        """
        self.lookback = lookback
        self.horizon = horizon
        self.test_size = test_size
        self.random_state = random_state
        self.classifier_kwargs = classifier_kwargs
        self.models = {}

    def fit(self, series, labels):
        """Fit all supported classifiers on windowed time-series data.

        Parameters
        ----------
        series : array-like
            Time-series values used to create feature windows.
        labels : array-like
            Classification targets aligned to the input series.

        Returns
        -------
        scores : pandas.DataFrame
            Model comparison table returned by LazyClassifier.
        predictions : pandas.DataFrame
            Optional predictions table returned by LazyClassifier.
        """
        X, y = make_classification_windows(
            series,
            labels,
            lookback=self.lookback,
            horizon=self.horizon,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=False,
        )

        classifier = LazyClassifier(**self.classifier_kwargs)
        scores, predictions = classifier.fit(X_train, X_test, y_train, y_test)
        self.models = classifier.models
        return scores, predictions


class LazyTimeSeriesRegressor:
    """Benchmark regression models on supervised time-series windows."""

    def __init__(
        self,
        lookback=5,
        horizon=1,
        test_size=0.2,
        random_state=42,
        **regressor_kwargs,
    ):
        """Store the windowing configuration and regressor options.

        Parameters
        ----------
        lookback : int, optional
            Number of historical steps used per sample.
        horizon : int, optional
            Number of future steps per target window.
        test_size : float, optional
            Fraction of samples reserved for evaluation.
        random_state : int, optional
            Random seed used by the internal train/test split.
        **regressor_kwargs : dict
            Additional keyword arguments forwarded to LazyRegressor.
        """
        self.lookback = lookback
        self.horizon = horizon
        self.test_size = test_size
        self.random_state = random_state
        self.regressor_kwargs = regressor_kwargs
        self.models = {}

    def fit(self, series):
        """Fit all supported regressors on windowed time-series data.

        Parameters
        ----------
        series : array-like
            Time-series values used to create feature and target windows.

        Returns
        -------
        scores : pandas.DataFrame
            Model comparison table returned by LazyRegressor.
        predictions : pandas.DataFrame
            Optional predictions table returned by LazyRegressor.
        """
        X, y = make_supervised_windows(series, lookback=self.lookback, horizon=self.horizon)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=False
        )

        regressor = LazyRegressor(**self.regressor_kwargs)
        scores, predictions = regressor.fit(X_train, X_test, y_train, y_test)
        self.models = regressor.models
        return scores, predictions


class LazyTimeSeriesForecasting(LazyTimeSeriesRegressor):
    """Alias for LazyTimeSeriesRegressor for forecasting use cases."""


class LazyTimeSeriesClassification(LazyTimeSeriesClassifier):
    """Alias for LazyTimeSeriesClassifier for classification use cases."""