from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# Set up grid of parameters to optimize over
# 3*3=9 points
param_grid = {
    'svc__gamma': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
    'svc__C': [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4],
}
n_jobs = 9

estimator = make_pipeline(
    StandardScaler(
        with_mean=True,
        with_std=True,
    ),
    SVC(
        kernel='rbf',
        probability=True,
        random_state=1337,
        cache_size=500,
    ),
)
