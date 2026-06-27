# -*- coding: utf-8 -*-
from .classifier import LazyClassifier
from .regressor import LazyRegressor
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
	"LazyRegressor",
	"LazyTimeSeriesClassification",
	"LazyTimeSeriesClassifier",
	"LazyTimeSeriesForecasting",
	"LazyTimeSeriesRegressor",
	"evaluate_ts_classification",
	"evaluate_ts_forecasting",
	"make_supervised_windows",
]
