from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List


# Set up grid of parameters to optimize over
# 3*3=9 points
param_grid: Dict[str, List[Any]] = {
    'randomforestclassifier__n_estimators': [100, 500, 1000],
    'randomforestclassifier__max_features': ['sqrt', 0.1, 0.333],
}
n_jobs = 9

estimator = make_pipeline(
    StandardScaler(
        with_mean=True,
        with_std=True,
    ),
    RandomForestClassifier(),
)
