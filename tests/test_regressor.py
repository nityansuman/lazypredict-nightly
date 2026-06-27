import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from lazypredict import LazyRegressor


def test_lazy_regressor_runs_on_dummy_data():
    X, y = make_regression(n_samples=80, n_features=5, n_informative=4, noise=0.1, random_state=42)
    split = 60
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        regressors=[LinearRegression, DecisionTreeRegressor],
        predictions=True,
    )

    scores, predictions = reg.fit(X_train, X_test, y_train, y_test)

    assert not scores.empty
    assert list(predictions.columns) == ["LinearRegression", "DecisionTreeRegressor"]
    assert np.isfinite(scores["R-Squared"]).all()


def test_lazy_regressor_handles_multi_output_targets():
    X, y = make_regression(n_samples=90, n_features=6, n_informative=5, noise=0.1, random_state=42)
    y = np.column_stack([y, y * 0.25])

    split = 65
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True,
        regressors=[DecisionTreeRegressor],
        predictions=False,
    )

    scores, _ = reg.fit(X_train, X_test, y_train, y_test)

    assert not scores.empty
    assert np.isfinite(scores["RMSE"]).all()