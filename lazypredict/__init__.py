# -*- coding: utf-8 -*-
from .supervised import LazyClassifier, LazyRegressor
from .timeseries import (
	LazyTimeSeriesClassification,
	LazyTimeSeriesClassifier,
	LazyTimeSeriesForecasting,
	LazyTimeSeriesRegressor,
	evaluate_ts_classification,
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
