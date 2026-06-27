import time
import warnings

import lightgbm
import numpy as np
import pandas as pd
import xgboost
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import all_estimators
from tqdm import tqdm

from .utils.regression_metrics import RegressionMetrics
from .classifier import get_card_split

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

REMOVED_REGRESSORS = [
    "TheilSenRegressor",
    "ARDRegression",
    "CCA",
    "IsotonicRegression",
    "StackingRegressor",
    "MultiOutputRegressor",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "PLSCanonical",
    "PLSRegression",
    "RadiusNeighborsRegressor",
    "RegressorChain",
    "VotingRegressor",
]

REGRESSORS = [
    est
    for est in all_estimators()
    if issubclass(est[1], RegressorMixin) and est[0] not in REMOVED_REGRESSORS
]

REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OrdinalEncoder()),
    ]
)


def filter_models(models, exclude_models=None):
    """Remove excluded model names from a list of estimator tuples."""
    if not exclude_models:
        return list(models)

    excluded = {model_name for model_name in exclude_models}
    return [model for model in models if model[0] not in excluded]


class LazyRegressor:
    """Benchmark scikit-learn regression algorithms on tabular data."""

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        regressors="all",
        exclude_models=None,
        time_limit=None,
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.regressors = regressors
        self.exclude_models = exclude_models or []
        self.time_limit = time_limit

    def fit(self, X_train, X_test, y_train, y_test):
        """Fit regression algorithms and return scores plus optional predictions."""
        R2 = []
        ADJR2 = []
        RMSE = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns
        categorical_low, categorical_high = get_card_split(X_train, categorical_features)

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )

        if self.regressors == "all":
            self.regressors = filter_models(REGRESSORS, self.exclude_models)
        else:
            temp_list = []
            for regressor in self.regressors:
                temp_list.append((regressor.__name__, regressor))
            self.regressors = filter_models(temp_list, self.exclude_models)

        benchmark_start = time.time()

        for name, model in tqdm(self.regressors):
            if self.time_limit is not None and (time.time() - benchmark_start) >= self.time_limit:
                if self.ignore_warnings is False:
                    print(f"Stopped early after reaching the {self.time_limit} second time limit")
                break

            start = time.time()
            try:
                if "random_state" in model().get_params().keys():
                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("regressor", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model())])

                pipe.fit(X_train, y_train)
                self.models[name] = pipe
                y_pred = pipe.predict(X_test)

                r_squared = r2_score(y_test, y_pred)
                adj_rsquared = RegressionMetrics.adjusted_rsquared(r_squared, X_test.shape[0], X_test.shape[1])
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                names.append(name)
                R2.append(r_squared)
                ADJR2.append(adj_rsquared)
                RMSE.append(rmse)
                TIME.append(time.time() - start)

                if self.custom_metric:
                    CUSTOM_METRIC.append(self.custom_metric(y_test, y_pred))

                if self.verbose > 0:
                    scores_verbose = {
                        "Model": name,
                        "R-Squared": r_squared,
                        "Adjusted R-Squared": adj_rsquared,
                        "RMSE": rmse,
                        "Time taken": time.time() - start,
                    }
                    if self.custom_metric:
                        scores_verbose[self.custom_metric.__name__] = CUSTOM_METRIC[-1]
                    print(scores_verbose)

                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)

        scores = {
            "Model": names,
            "Adjusted R-Squared": ADJR2,
            "R-Squared": R2,
            "RMSE": RMSE,
            "Time Taken": TIME,
        }

        if self.custom_metric:
            scores[self.custom_metric.__name__] = CUSTOM_METRIC

        scores = pd.DataFrame(scores)
        scores = scores.sort_values(by="Adjusted R-Squared", ascending=False).set_index("Model")

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        """Return the fitted model pipelines, training on demand if needed."""
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models
