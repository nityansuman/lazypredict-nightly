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
from .timeseries_classification import (
	LazyTimeSeriesClassification,
	LazyTimeSeriesClassifier,
	evaluate_ts_classification,
)
from .timeseries_forecasting import (
	LazyTimeSeriesForecasting,
	LazyTimeSeriesRegressor,
	evaluate_ts_forecasting,
	make_supervised_windows,
)

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
	"evaluate_ts_classification",
	"evaluate_ts_forecasting",
	"select_best_model",
	"make_supervised_windows",
]
