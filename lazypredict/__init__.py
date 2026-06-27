# -*- coding: utf-8 -*-
from .classifier import LazyClassifier
from .regressor import LazyRegressor
from .auto import (
	AutoLazyClassifier,
	AutoLazyRegressor,
	AutoLazyTimeSeriesClassifier,
	AutoLazyTimeSeriesForecasting,
	select_best_model,
)
from .timeseries_classification import LazyTimeSeriesClassification, LazyTimeSeriesClassifier
from .timeseries_forecasting import LazyTimeSeriesForecasting, LazyTimeSeriesRegressor, make_supervised_windows
from .utils.classification_metrics import ClassificationMetrics
from .utils.regression_metrics import RegressionMetrics
from .utils.time_series_metrics import TimeSeriesMetrics

__version__ = "0.4.0"

__all__ = [
	"LazyClassifier",
	"AutoLazyClassifier",
	"LazyRegressor",
	"AutoLazyRegressor",
	"LazyTimeSeriesClassification",
	"LazyTimeSeriesClassifier",
	"AutoLazyTimeSeriesClassifier",
	"LazyTimeSeriesForecasting",
	"LazyTimeSeriesRegressor",
	"AutoLazyTimeSeriesForecasting",
	"ClassificationMetrics",
	"RegressionMetrics",
	"TimeSeriesMetrics",
	"select_best_model",
	"make_supervised_windows",
]
