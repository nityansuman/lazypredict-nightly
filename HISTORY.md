---
Lastest Release
---

# 0.5.0 (2026-06-27)

- Refactored the package into dedicated classifier, regressor, auto-selection, and time-series modules.
- Added class-based metric containers for classification, regression, and time-series evaluation.
- Added leakage-aware time-series holdout splitting and walk-forward backtesting.
- Added automatic model selection wrappers for classification, regression, and time-series workflows.
- Improved ROC AUC handling to use probabilistic or margin-based outputs.
- Made xgboost and lightgbm optional extras instead of hard requirements.
- Expanded test coverage for imbalanced multiclass classification, multi-output regression, and time-series edge cases.
- Updated the README with the new API, backtest examples, and shorter example outputs.

## Notable changes

- supervised.py was reduced to a compatibility layer, while the real implementations now live in separate modules.
- __init__.py now exposes a cleaner public API.
requirements.txt is lighter, with boosting libraries now installable via extras.
- Time-series benchmarking is now more leakage-aware and supports backtesting.
- Tests and smoke coverage were expanded significantly.

## Potential breaking changes

- Some old helper-style metric imports were removed in favor of class-based access.
- Time-series usage is now documented and structured around the new backtest() API.
- If you relied on xgboost or lightgbm being installed automatically, you now need the boosting extra.

## Install notes

- Base install
```
pip install lazypredict-nightly
```
- With boosting models
```
pip install "lazypredict-nightly[boosting]"
```
