# Lazy Predict [Nightly]

Lazy Predict 2.0 to help you benchmark models without much code and understand what works better without any hyperparameter tuning.

[![image](https://img.shields.io/pypi/v/lazypredict-nightly.svg)](https://pypi.python.org/pypi/lazypredict-nightly)
[![Downloads](https://pepy.tech/badge/lazypredict-nightly)](https://pepy.tech/project/lazypredict-nightly)

## Getting started

To install Lazy Predict Nightly:

    pip install lazypredict-nightly

To use Lazy Predict in a project:

    import lazypredict

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


    | Model                          |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
    |:-------------------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
    | LinearSVC                      |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0150008 |
    | SGDClassifier                  |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0109992 |
    | MLPClassifier                  |   0.985965 |            0.986904 |  0.986904 |   0.985994 |    0.426     |
    | Perceptron                     |   0.985965 |            0.984797 |  0.984797 |   0.985965 |    0.0120046 |
    | LogisticRegression             |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.0200036 |
    | LogisticRegressionCV           |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.262997  |
    | SVC                            |   0.982456 |            0.979942 |  0.979942 |   0.982437 |    0.0140011 |
    | CalibratedClassifierCV         |   0.982456 |            0.975728 |  0.975728 |   0.982357 |    0.0350015 |
    | PassiveAggressiveClassifier    |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0130005 |
    | LabelPropagation               |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0429988 |
    | LabelSpreading                 |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0310006 |
    | RandomForestClassifier         |   0.97193  |            0.969594 |  0.969594 |   0.97193  |    0.033     |
    | GradientBoostingClassifier     |   0.97193  |            0.967486 |  0.967486 |   0.971869 |    0.166998  |
    | QuadraticDiscriminantAnalysis  |   0.964912 |            0.966206 |  0.966206 |   0.965052 |    0.0119994 |
    | HistGradientBoostingClassifier |   0.968421 |            0.964739 |  0.964739 |   0.968387 |    0.682003  |
    | RidgeClassifierCV              |   0.97193  |            0.963272 |  0.963272 |   0.971736 |    0.0130029 |
    | RidgeClassifier                |   0.968421 |            0.960525 |  0.960525 |   0.968242 |    0.0119977 |
    | AdaBoostClassifier             |   0.961404 |            0.959245 |  0.959245 |   0.961444 |    0.204998  |
    | ExtraTreesClassifier           |   0.961404 |            0.957138 |  0.957138 |   0.961362 |    0.0270066 |
    | KNeighborsClassifier           |   0.961404 |            0.95503  |  0.95503  |   0.961276 |    0.0560005 |
    | BaggingClassifier              |   0.947368 |            0.954577 |  0.954577 |   0.947882 |    0.0559971 |
    | BernoulliNB                    |   0.950877 |            0.951003 |  0.951003 |   0.951072 |    0.0169988 |
    | LinearDiscriminantAnalysis     |   0.961404 |            0.950816 |  0.950816 |   0.961089 |    0.0199995 |
    | GaussianNB                     |   0.954386 |            0.949536 |  0.949536 |   0.954337 |    0.0139935 |
    | NuSVC                          |   0.954386 |            0.943215 |  0.943215 |   0.954014 |    0.019989  |
    | DecisionTreeClassifier         |   0.936842 |            0.933693 |  0.933693 |   0.936971 |    0.0170023 |
    | NearestCentroid                |   0.947368 |            0.933506 |  0.933506 |   0.946801 |    0.0160074 |
    | ExtraTreeClassifier            |   0.922807 |            0.912168 |  0.912168 |   0.922462 |    0.0109999 |
    | CheckingClassifier             |   0.361404 |            0.5      |  0.5      |   0.191879 |    0.0170043 |
    | DummyClassifier                |   0.512281 |            0.489598 |  0.489598 |   0.518924 |    0.0119965 |
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


    | Model                         | Adjusted R-Squared | R-Squared |  RMSE | Time Taken |
    |:------------------------------|-------------------:|----------:|------:|-----------:|
    | SVR                           |               0.83 |      0.88 |  2.62 |       0.01 |
    | BaggingRegressor              |               0.83 |      0.88 |  2.63 |       0.03 |
    | NuSVR                         |               0.82 |      0.86 |  2.76 |       0.03 |
    | RandomForestRegressor         |               0.81 |      0.86 |  2.78 |       0.21 |
    | XGBRegressor                  |               0.81 |      0.86 |  2.79 |       0.06 |
    | GradientBoostingRegressor     |               0.81 |      0.86 |  2.84 |       0.11 |
    | ExtraTreesRegressor           |               0.79 |      0.84 |  2.98 |       0.12 |
    | AdaBoostRegressor             |               0.78 |      0.83 |  3.04 |       0.07 |
    | HistGradientBoostingRegressor |               0.77 |      0.83 |  3.06 |       0.17 |
    | PoissonRegressor              |               0.77 |      0.83 |  3.11 |       0.01 |
    | LGBMRegressor                 |               0.77 |      0.83 |  3.11 |       0.07 |
    | KNeighborsRegressor           |               0.77 |      0.83 |  3.12 |       0.01 |
    | DecisionTreeRegressor         |               0.65 |      0.74 |  3.79 |       0.01 |
    | MLPRegressor                  |               0.65 |      0.74 |  3.80 |       1.63 |
    | HuberRegressor                |               0.64 |      0.74 |  3.84 |       0.01 |
    | GammaRegressor                |               0.64 |      0.73 |  3.88 |       0.01 |
    | LinearSVR                     |               0.62 |      0.72 |  3.96 |       0.01 |
    | RidgeCV                       |               0.62 |      0.72 |  3.97 |       0.01 |
    | BayesianRidge                 |               0.62 |      0.72 |  3.97 |       0.01 |
    | Ridge                         |               0.62 |      0.72 |  3.97 |       0.01 |
    | TransformedTargetRegressor    |               0.62 |      0.72 |  3.97 |       0.01 |
    | LinearRegression              |               0.62 |      0.72 |  3.97 |       0.01 |
    | ElasticNetCV                  |               0.62 |      0.72 |  3.98 |       0.04 |
    | LassoCV                       |               0.62 |      0.72 |  3.98 |       0.06 |
    | LassoLarsIC                   |               0.62 |      0.72 |  3.98 |       0.01 |
    | LassoLarsCV                   |               0.62 |      0.72 |  3.98 |       0.02 |
    | Lars                          |               0.61 |      0.72 |  3.99 |       0.01 |
    | LarsCV                        |               0.61 |      0.71 |  4.02 |       0.04 |
    | SGDRegressor                  |               0.60 |      0.70 |  4.07 |       0.01 |
    | TweedieRegressor              |               0.59 |      0.70 |  4.12 |       0.01 |
    | GeneralizedLinearRegressor    |               0.59 |      0.70 |  4.12 |       0.01 |
    | ElasticNet                    |               0.58 |      0.69 |  4.16 |       0.01 |
    | Lasso                         |               0.54 |      0.66 |  4.35 |       0.02 |
    | RANSACRegressor               |               0.53 |      0.65 |  4.41 |       0.04 |
    | OrthogonalMatchingPursuitCV   |               0.45 |      0.59 |  4.78 |       0.02 |
    | PassiveAggressiveRegressor    |               0.37 |      0.54 |  5.09 |       0.01 |
    | GaussianProcessRegressor      |               0.23 |      0.43 |  5.65 |       0.03 |
    | OrthogonalMatchingPursuit     |               0.16 |      0.38 |  5.89 |       0.01 |
    | ExtraTreeRegressor            |               0.08 |      0.32 |  6.17 |       0.01 |
    | DummyRegressor                |              -0.38 |     -0.02 |  7.56 |       0.01 |
    | LassoLars                     |              -0.38 |     -0.02 |  7.56 |       0.01 |
    | KernelRidge                   |             -11.50 |     -8.25 | 22.74 |       0.01 |
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

### Evaluation helpers

```python
    from lazypredict import evaluate_ts_classification, evaluate_ts_forecasting

    classification_metrics = evaluate_ts_classification(y_true, y_pred)
    forecasting_metrics = evaluate_ts_forecasting(y_true, y_pred)
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
