import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from lazypredict import (
    LazyTimeSeriesClassification,
    LazyTimeSeriesForecasting,
    TimeSeriesMetrics,
    make_supervised_windows,
)


def test_time_series_forecasting_smoke():
    series = np.arange(80).reshape(-1, 1)
    X, y = make_supervised_windows(series, lookback=5, horizon=2)

    assert X.shape[0] == y.shape[0]
    assert y.shape[1] == 2

    model = LazyTimeSeriesForecasting(
        lookback=5,
        horizon=2,
        test_size=0.2,
        regressors=[DecisionTreeRegressor],
    )
    scores, predictions = model.fit(series)

    assert not scores.empty
    assert predictions is None or not predictions.empty


def test_time_series_forecasting_backtest_smoke():
    series = np.arange(80).reshape(-1, 1)
    model = LazyTimeSeriesForecasting(
        lookback=5,
        horizon=2,
        test_size=0.2,
        n_splits=3,
        regressors=[DecisionTreeRegressor],
    )
    backtest_scores = model.backtest(series)

    assert not backtest_scores.empty


def test_time_series_forecasting_handles_tiny_series():
    series = np.arange(20).reshape(-1, 1)
    model = LazyTimeSeriesForecasting(
        lookback=3,
        horizon=1,
        test_size=0.25,
        n_splits=2,
        regressors=[DecisionTreeRegressor],
    )
    with pytest.raises(ValueError, match="No valid backtest scores were produced"):
        model.backtest(series)


def test_time_series_classification_smoke():
    series = np.column_stack([np.arange(80), np.arange(80) * 2])
    labels = np.array([0, 1] * 40)

    model = LazyTimeSeriesClassification(
        lookback=5,
        horizon=1,
        test_size=0.2,
        classifiers=[DecisionTreeClassifier],
    )
    scores, predictions = model.fit(series, labels)

    assert not scores.empty
    assert predictions is None or not predictions.empty


def test_time_series_classification_backtest_smoke():
    series = np.column_stack([np.arange(80), np.arange(80) * 2])
    labels = np.array([0, 1] * 40)
    model = LazyTimeSeriesClassification(
        lookback=5,
        horizon=1,
        test_size=0.2,
        n_splits=3,
        classifiers=[DecisionTreeClassifier],
    )
    backtest_scores = model.backtest(series, labels)

    assert not backtest_scores.empty


def test_time_series_classification_handles_short_window():
    series = np.column_stack([np.arange(30), np.arange(30) * 2])
    labels = np.array([0, 1] * 15)
    model = LazyTimeSeriesClassification(
        lookback=4,
        horizon=1,
        test_size=0.25,
        n_splits=2,
        classifiers=[DecisionTreeClassifier],
    )
    scores = model.backtest(series, labels)

    assert not scores.empty


def test_time_series_metrics_helpers():
    classification_metrics = TimeSeriesMetrics.classification([0, 1, 1], [0, 1, 0])
    forecasting_metrics = TimeSeriesMetrics.forecasting(np.array([1.0, 2.0]), np.array([1.1, 1.9]))

    assert set(classification_metrics) == {"Accuracy", "Balanced Accuracy", "F1 Score"}
    assert set(forecasting_metrics) >= {"MAE", "RMSE", "R2"}