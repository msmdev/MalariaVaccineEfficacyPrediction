from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List
import numpy as np
from source.config import seed

# Set up grid of parameters to optimize over
# 3*3=9 points
param_grid: Dict[str, List[Any]] = {
    'svc__gamma': 10.0 ** np.arange(-6, 6),
    'svc__C': 10.0 ** np.arange(-4, 5),
}
n_jobs = -1

estimator = make_pipeline(
    StandardScaler(
        with_mean=True,
        with_std=True,
    ),
    SVC(
        kernel='rbf',
        probability=True,
        random_state=seed,
        cache_size=500,
    ),
)
