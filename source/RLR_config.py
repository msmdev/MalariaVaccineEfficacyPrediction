from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List
from source.config import seed


# Set up grid of parameters to optimize over
# 3*3=9 points
param_grid: Dict[str, List[Any]] = {
    'logisticregression__l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'logisticregression__C': [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4],
}
n_jobs = -1

estimator = make_pipeline(
    StandardScaler(
        with_mean=True,
        with_std=True,
    ),
    LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        max_iter=10000,
        random_state=seed,
    ),
)
