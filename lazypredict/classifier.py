import warnings
import time

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from .utils.classification_metrics import ClassificationMetrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils import all_estimators
from tqdm import tqdm

try:
    import xgboost
except ImportError:  # pragma: no cover - optional dependency
    xgboost = None

try:
    import lightgbm
except ImportError:  # pragma: no cover - optional dependency
    lightgbm = None

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

REMOVED_CLASSIFIERS = [
    "ClassifierChain",
    "ComplementNB",
    "GradientBoostingClassifier",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    "MLPClassifier",
    "LogisticRegressionCV",
    "MultiOutputClassifier",
    "MultinomialNB",
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
]

CLASSIFIERS = [
    est
    for est in all_estimators()
    if issubclass(est[1], ClassifierMixin) and est[0] not in REMOVED_CLASSIFIERS
]

if xgboost is not None:
    CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))

if lightgbm is not None:
    CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))


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


def get_card_split(df, cols, n=11):
    """Split categorical columns into low- and high-cardinality groups."""
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high


def filter_models(models, exclude_models=None):
    """Remove excluded model names from a list of estimator tuples."""
    if not exclude_models:
        return list(models)

    excluded = {model_name for model_name in exclude_models}
    return [model for model in models if model[0] not in excluded]


class LazyClassifier:
    """Benchmark scikit-learn classification algorithms on tabular data."""

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        classifiers="all",
        exclude_models=None,
        time_limit=None,
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.classifiers = classifiers
        self.exclude_models = exclude_models or []
        self.time_limit = time_limit

    def fit(self, X_train, X_test, y_train, y_test):
        """Fit classification algorithms and return scores plus optional predictions."""
        Accuracy = []
        B_Accuracy = []
        ROC_AUC = []
        F1 = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric is not None:
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

        if self.classifiers == "all":
            self.classifiers = filter_models(CLASSIFIERS, self.exclude_models)
        else:
            temp_list = []
            for classifier in self.classifiers:
                temp_list.append((classifier.__name__, classifier))
            self.classifiers = filter_models(temp_list, self.exclude_models)

        benchmark_start = time.time()

        for name, model in tqdm(self.classifiers):
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
                            ("classifier", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model())])

                pipe.fit(X_train, y_train)
                self.models[name] = pipe
                y_pred = pipe.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                try:
                    roc_auc = ClassificationMetrics.roc_auc(pipe, X_test, y_test)
                except Exception as exception:
                    roc_auc = None
                    if self.ignore_warnings is False:
                        print("ROC AUC couldn't be calculated for " + name)
                        print(exception)

                names.append(name)
                Accuracy.append(accuracy)
                B_Accuracy.append(b_accuracy)
                ROC_AUC.append(roc_auc)
                F1.append(f1)
                TIME.append(time.time() - start)

                if self.custom_metric is not None:
                    CUSTOM_METRIC.append(self.custom_metric(y_test, y_pred))

                if self.verbose > 0:
                    payload = {
                        "Model": name,
                        "Accuracy": accuracy,
                        "Balanced Accuracy": b_accuracy,
                        "ROC AUC": roc_auc,
                        "F1 Score": f1,
                        "Time taken": time.time() - start,
                    }
                    if self.custom_metric is not None:
                        payload[self.custom_metric.__name__] = CUSTOM_METRIC[-1]
                    print(payload)

                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)

        if self.custom_metric is None:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    "Time Taken": TIME,
                }
            )
        else:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    self.custom_metric.__name__: CUSTOM_METRIC,
                    "Time Taken": TIME,
                }
            )

        scores = scores.sort_values(by="Balanced Accuracy", ascending=False).set_index("Model")

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        """Return the fitted model pipelines, training on demand if needed."""
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models
