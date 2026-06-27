"""Automatic model selection helpers built on top of LazyPredict benchmarkers."""

from .classifier import LazyClassifier
from .regressor import LazyRegressor
from .timeseries_classification import LazyTimeSeriesClassifier
from .timeseries_forecasting import LazyTimeSeriesRegressor


def select_best_model(scores, metric=None, ascending=False):
    """Return the best row from a benchmark score table.

    Parameters
    ----------
    scores : pandas.DataFrame
        Benchmark output indexed by model name.
    metric : str, optional
        Metric column used for ranking. Defaults to the first column.
    ascending : bool, optional
        Sort direction for the ranking metric.

    Returns
    -------
    tuple
        ``(model_name, score_row)`` for the top-ranked model.
    """
    if scores.empty:
        raise ValueError("scores must not be empty")

    sort_metric = metric or scores.columns[0]
    ranked = scores.sort_values(by=sort_metric, ascending=ascending)
    best_name = ranked.index[0]
    return best_name, ranked.iloc[0]


class AutoLazyClassifier:
    """Fit all classifiers and return the best-scoring classifier."""

    def __init__(self, metric="Balanced Accuracy", ascending=False, **classifier_kwargs):
        self.metric = metric
        self.ascending = ascending
        self.classifier_kwargs = classifier_kwargs
        self.best_model_name = None
        self.best_model = None
        self.scores = None

    def fit(self, X_train, X_test, y_train, y_test):
        """Run LazyClassifier and keep only the top-ranked model."""
        benchmarker = LazyClassifier(**self.classifier_kwargs)
        scores, _ = benchmarker.fit(X_train, X_test, y_train, y_test)
        best_name, _ = select_best_model(scores, metric=self.metric, ascending=self.ascending)

        self.scores = scores
        self.best_model_name = best_name
        self.best_model = benchmarker.models.get(best_name)
        return self.best_model_name, self.best_model, self.scores


class AutoLazyRegressor:
    """Fit all regressors and return the best-scoring regressor."""

    def __init__(self, metric="Adjusted R-Squared", ascending=False, **regressor_kwargs):
        self.metric = metric
        self.ascending = ascending
        self.regressor_kwargs = regressor_kwargs
        self.best_model_name = None
        self.best_model = None
        self.scores = None

    def fit(self, X_train, X_test, y_train, y_test):
        """Run LazyRegressor and keep only the top-ranked model."""
        benchmarker = LazyRegressor(**self.regressor_kwargs)
        scores, _ = benchmarker.fit(X_train, X_test, y_train, y_test)
        best_name, _ = select_best_model(scores, metric=self.metric, ascending=self.ascending)

        self.scores = scores
        self.best_model_name = best_name
        self.best_model = benchmarker.models.get(best_name)
        return self.best_model_name, self.best_model, self.scores


class AutoLazyTimeSeriesClassifier:
    """Fit time-series classifiers and return the best-scoring classifier."""

    def __init__(self, metric="Balanced Accuracy", ascending=False, **classifier_kwargs):
        self.metric = metric
        self.ascending = ascending
        self.classifier_kwargs = classifier_kwargs
        self.best_model_name = None
        self.best_model = None
        self.scores = None

    def fit(self, series, labels):
        """Run LazyTimeSeriesClassifier and keep only the top-ranked model."""
        benchmarker = LazyTimeSeriesClassifier(**self.classifier_kwargs)
        scores, _ = benchmarker.fit(series, labels)
        best_name, _ = select_best_model(scores, metric=self.metric, ascending=self.ascending)

        self.scores = scores
        self.best_model_name = best_name
        self.best_model = benchmarker.models.get(best_name)
        return self.best_model_name, self.best_model, self.scores


class AutoLazyTimeSeriesForecasting:
    """Fit time-series regressors and return the best-scoring regressor."""

    def __init__(self, metric="Adjusted R-Squared", ascending=False, **regressor_kwargs):
        self.metric = metric
        self.ascending = ascending
        self.regressor_kwargs = regressor_kwargs
        self.best_model_name = None
        self.best_model = None
        self.scores = None

    def fit(self, series):
        """Run LazyTimeSeriesRegressor and keep only the top-ranked model."""
        benchmarker = LazyTimeSeriesRegressor(**self.regressor_kwargs)
        scores, _ = benchmarker.fit(series)
        best_name, _ = select_best_model(scores, metric=self.metric, ascending=self.ascending)

        self.scores = scores
        self.best_model_name = best_name
        self.best_model = benchmarker.models.get(best_name)
        return self.best_model_name, self.best_model, self.scores