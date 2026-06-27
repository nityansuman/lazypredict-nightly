import pandas as pd

from lazypredict.auto import select_best_model


def test_select_best_model_uses_requested_metric():
    scores = pd.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "Balanced Accuracy": [0.7, 0.9, 0.8],
            "Adjusted R-Squared": [0.2, 0.1, 0.3],
        }
    ).set_index("Model")

    best_name, best_row = select_best_model(scores, metric="Balanced Accuracy", ascending=False)

    assert best_name == "B"
    assert best_row["Balanced Accuracy"] == 0.9


def test_select_best_model_handles_minimization():
    scores = pd.DataFrame(
        {
            "Model": ["A", "B", "C"],
            "RMSE": [2.0, 1.0, 3.0],
        }
    ).set_index("Model")

    best_name, best_row = select_best_model(scores, metric="RMSE", ascending=True)

    assert best_name == "B"
    assert best_row["RMSE"] == 1.0