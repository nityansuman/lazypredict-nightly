# Lazy Predict [Nightly]

Lazy Predict 2.0 to help you benchmark models without much code and understand what works better without any hyperparameter tuning.

[![image](https://img.shields.io/pypi/v/lazypredict-nightly.svg)](https://pypi.python.org/pypi/lazypredict-nightly)
[![Downloads](https://pepy.tech/badge/lazypredict-nightly)](https://pepy.tech/project/lazypredict-nightly)

## Getting started

To install Lazy Predict Nightly:

    pip install lazypredict-nightly

Optional extras for boosted trees are available when you want the full model catalog:

    pip install "lazypredict-nightly[boosting]"

To use Lazy Predict in a project:

    import lazypredict

### Public API

Lazy Predict Nightly keeps the main benchmarking classes at the package root:

```python
    from lazypredict import (
        LazyClassifier,
        LazyRegressor,
        LazyTimeSeriesClassification,
        LazyTimeSeriesForecasting,
        AutoLazyClassifier,
        AutoLazyRegressor,
        ClassificationMetrics,
        RegressionMetrics,
        TimeSeriesMetrics,
    )
```

## Examples

The examples below use multiple datasets from `sklearn` so you can see how the package fits different use cases.

### Classification

Binary and multiclass examples using breast cancer and iris.

```python
    from lazypredict import LazyClassifier

    from sklearn.datasets import load_breast_cancer, load_iris
    from sklearn.model_selection import train_test_split

    # Binary classification example
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    models,predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Expected output: a DataFrame sorted by Balanced Accuracy and a predictions table.
    print(models)

    # Multiclass classification example
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=123)

    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Expected output: a ranking table with multiclass metrics.
    print(models.head())

    # Lazy Predict prints the full ranking table at runtime; the README keeps this sample short.
    # Top rows typically look like:
    # | Model              | Accuracy | Balanced Accuracy | ROC AUC | F1 Score | Time Taken |
    # | LinearSVC          |    ...   |        ...        |   ...   |   ...    |    ...     |
    # | SGDClassifier      |    ...   |        ...        |   ...   |   ...    |    ...     |
    # | SVC                |    ...   |        ...        |   ...   |   ...    |    ...     |
```

### Regression

Single-output and multi-output examples using California housing and diabetes.

```python
    from lazypredict import LazyRegressor

    from sklearn.datasets import fetch_california_housing, load_diabetes
    from sklearn.utils import shuffle
    import numpy as np

    # Single-output regression example
    housing = fetch_california_housing()
    X, y = shuffle(housing.data, housing.target, random_state=13)
    X = X.astype(np.float32)

    offset = int(X.shape[0] * 0.9)

    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    # Expected output: a DataFrame sorted by Adjusted R-Squared and optional predictions.
    print(models)

    # LightGBM may print informational warnings while fitting; this is expected.

    # Multi-output style regression example built from diabetes features
    diabetes = load_diabetes()
    X = diabetes.data.astype(np.float32)
    y = np.column_stack([diabetes.target, diabetes.target * 0.5])

    offset = int(X.shape[0] * 0.9)

    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    # Expected output: a ranking table for multi-output regression support.
    print(models.head())

    # Lazy Predict prints the full ranking table at runtime; the README keeps this sample short.
    # Top rows typically look like:
    # | Model              | Adjusted R-Squared | R-Squared | RMSE | Time Taken |
    # | SVR                |         ...        |    ...    | ...  |    ...     |
    # | RandomForestRegressor |      ...        |    ...    | ...  |    ...     |
    # | XGBRegressor / LGBMRegressor may appear when the optional boosting extra is installed.
```

### Time Series

Lazy Predict Nightly also provides wrappers for supervised time series workflows.

### 1-step forecasting

```python
    from lazypredict import LazyTimeSeriesForecasting
    from sklearn.datasets import load_diabetes

    # Use the diabetes features as a multivariate time-series-style input
    data = load_diabetes()
    series = data.data

    model = LazyTimeSeriesForecasting(lookback=10, horizon=1, test_size=0.2)
    scores, predictions = model.fit(series)

    # Expected output: a DataFrame ranked by Adjusted R-Squared.
    print(scores.head())
```

```python
    from lazypredict import LazyTimeSeriesForecasting
    from sklearn.datasets import load_diabetes

    # Walk-forward backtesting with expanding chronological splits
    data = load_diabetes()
    series = data.data

    model = LazyTimeSeriesForecasting(lookback=10, horizon=1, test_size=0.2, n_splits=3)
    scores = model.backtest(series)

    # Expected output: a model ranking averaged across walk-forward splits.
    print(scores.head())
```

### Multi-step forecasting

```python
    from lazypredict import LazyTimeSeriesForecasting
    from sklearn.datasets import fetch_california_housing

    # Use California housing columns as multivariate inputs
    housing = fetch_california_housing()
    series = housing.data

    model = LazyTimeSeriesForecasting(lookback=10, horizon=3, test_size=0.2)
    scores, predictions = model.fit(series)

    # Expected output: a multi-step forecasting benchmark table.
    print(scores.head())
```

### 1-step classification

```python
    from lazypredict import LazyTimeSeriesClassification
    from sklearn.datasets import load_breast_cancer

    # Use breast cancer features as a multivariate sequence with binary labels
    data = load_breast_cancer()
    series = data.data
    labels = data.target

    model = LazyTimeSeriesClassification(lookback=10, horizon=1, test_size=0.2)
    scores, predictions = model.fit(series, labels)

    # Expected output: a classification benchmark table ranked by Balanced Accuracy.
    print(scores.head())
```

```python
    from lazypredict import LazyTimeSeriesClassification
    from sklearn.datasets import load_breast_cancer

    # Walk-forward backtesting with chronological classification windows
    data = load_breast_cancer()
    series = data.data
    labels = data.target

    model = LazyTimeSeriesClassification(lookback=10, horizon=1, test_size=0.2, n_splits=3)
    scores = model.backtest(series, labels)

    # Expected output: a classification benchmark table averaged across splits.
    print(scores.head())
```

### Multi-step classification

```python
    from lazypredict import LazyTimeSeriesClassification
    from sklearn.datasets import load_iris

    # Use iris features with labels aligned to successive windows
    data = load_iris()
    series = data.data
    labels = data.target

    model = LazyTimeSeriesClassification(lookback=10, horizon=3, test_size=0.2)
    scores, predictions = model.fit(series, labels)

    # Expected output: a multi-step classification benchmark table.
    print(scores.head())
```

### Metric containers

```python
    from lazypredict import ClassificationMetrics, RegressionMetrics, TimeSeriesMetrics

    classification_metrics = TimeSeriesMetrics.classification(y_true, y_pred)
    forecasting_metrics = TimeSeriesMetrics.forecasting(y_true, y_pred)

    # The underlying metric containers are also available directly.
    accuracy = ClassificationMetrics.accuracy(y_true, y_pred)
    balanced_accuracy = ClassificationMetrics.balanced_accuracy(y_true, y_pred)
    f1_score_value = ClassificationMetrics.f1(y_true, y_pred)
    r_squared = RegressionMetrics.r_squared(y_true, y_pred)
    adjusted_r_squared = RegressionMetrics.adjusted_r_squared(y_true, y_pred, n_features=X_test.shape[1])
```

### Automatic Model Selection

If you want Lazy Predict Nightly to return the best model directly, use the auto wrappers.

```python
    from lazypredict import AutoLazyClassifier, AutoLazyRegressor
    from sklearn.datasets import load_iris, load_diabetes

    # Best classification model from the iris dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.3, random_state=123)

    auto_clf = AutoLazyClassifier(metric="Balanced Accuracy")
    best_name, best_model, scores = auto_clf.fit(X_train, X_test, y_train, y_test)

    # Expected output: best_name + fitted pipeline + full score table.
    print(best_name)

    # Best regression model from the diabetes dataset
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=.3, random_state=123)
    auto_reg = AutoLazyRegressor(metric="Adjusted R-Squared")
    best_name, best_model, scores = auto_reg.fit(X_train, X_test, y_train, y_test)

    print(best_name)
```

```python
    from lazypredict import AutoLazyTimeSeriesClassifier, AutoLazyTimeSeriesForecasting
    from sklearn.datasets import load_breast_cancer, fetch_california_housing

    # Best time-series classifier from breast cancer windows
    data = load_breast_cancer()
    auto_ts_clf = AutoLazyTimeSeriesClassifier(metric="Balanced Accuracy")
    best_name, best_model, scores = auto_ts_clf.fit(data.data, data.target)

    print(best_name)

    # Best time-series forecaster from California housing windows
    housing = fetch_california_housing()
    auto_ts_reg = AutoLazyTimeSeriesForecasting(metric="Adjusted R-Squared")
    best_name, best_model, scores = auto_ts_reg.fit(housing.data)

    print(best_name)
```
