"""End-to-end smoke runner for the full LazyPredict package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from lazypredict import (
    AutoLazyClassifier,
    AutoLazyRegressor,
    AutoLazyTimeSeriesClassifier,
    AutoLazyTimeSeriesForecasting,
    LazyClassifier,
    LazyRegressor,
    LazyTimeSeriesClassification,
    LazyTimeSeriesForecasting,
)


def run_classification():
    X, y = make_classification(n_samples=80, n_features=6, n_informative=4, random_state=42)
    split = 60
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        classifiers=[LogisticRegression, DecisionTreeClassifier],
    )
    scores, _ = clf.fit(X_train, X_test, y_train, y_test)
    assert not scores.empty

    auto_clf = AutoLazyClassifier(
        metric="Balanced Accuracy",
        classifiers=[LogisticRegression, DecisionTreeClassifier],
    )
    best_name, best_model, scores = auto_clf.fit(X_train, X_test, y_train, y_test)
    assert best_name in {"LogisticRegression", "DecisionTreeClassifier"}
    assert best_model is not None
    assert not scores.empty


def run_regression():
    X, y = make_regression(n_samples=80, n_features=5, n_informative=4, noise=0.1, random_state=42)
    split = 60
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        regressors=[LinearRegression, DecisionTreeRegressor],
    )
    scores, _ = reg.fit(X_train, X_test, y_train, y_test)
    assert not scores.empty

    auto_reg = AutoLazyRegressor(
        metric="Adjusted R-Squared",
        regressors=[LinearRegression, DecisionTreeRegressor],
    )
    best_name, best_model, scores = auto_reg.fit(X_train, X_test, y_train, y_test)
    assert best_name in {"LinearRegression", "DecisionTreeRegressor"}
    assert best_model is not None
    assert not scores.empty


def run_time_series():
    series = np.arange(80).reshape(-1, 1)
    labels = np.array([0, 1] * 40)

    ts_clf = LazyTimeSeriesClassification(
        lookback=5,
        horizon=1,
        test_size=0.2,
        classifiers=[DecisionTreeClassifier],
    )
    scores, _ = ts_clf.fit(series, labels)
    assert not scores.empty

    ts_reg = LazyTimeSeriesForecasting(
        lookback=5,
        horizon=2,
        test_size=0.2,
        regressors=[DecisionTreeRegressor],
    )
    scores, _ = ts_reg.fit(series)
    assert not scores.empty

    auto_ts_clf = AutoLazyTimeSeriesClassifier(
        metric="Balanced Accuracy",
        classifiers=[DecisionTreeClassifier],
    )
    best_name, best_model, scores = auto_ts_clf.fit(series, labels)
    assert best_name == "DecisionTreeClassifier"
    assert best_model is not None
    assert not scores.empty

    auto_ts_reg = AutoLazyTimeSeriesForecasting(
        metric="Adjusted R-Squared",
        regressors=[DecisionTreeRegressor],
    )
    best_name, best_model, scores = auto_ts_reg.fit(series)
    assert best_name == "DecisionTreeRegressor"
    assert best_model is not None
    assert not scores.empty


def main():
    run_classification()
    run_regression()
    run_time_series()
    print("End-to-end smoke test completed successfully.")


if __name__ == "__main__":
    main()