from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from source.utils import DataSelector
from typing import Dict, Union, Tuple, Any, List
import numpy as np
from source.utils import make_kernel_combinations
from source.config import seed


kernel_params: Dict[str, np.ndarray] = {
    'SA': np.array([0.25, 0.5, 0.75]),
    'SO': np.arange(-2.0, 3.0, dtype=np.float64),
    'R0': 10.0 ** np.arange(-6, 6),
    'R1': 10.0 ** np.arange(-6, 6),
    'R2': 10.0 ** np.arange(-6, 6),
    'P1': np.arange(2, 6),
    'P2': np.arange(2, 6),
}


def configurator(
    *,
    combination: str,
    identifier: str,
    kernel_dir: str,
) -> Tuple[Dict[str, List[Union[float, str]]], Any, int]:

    param_grid: Dict[str, List[Union[float, str]]]
    # Set up grid of parameters to optimize over
    if combination == 'SPP':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_params, "sigmoid_kernel", "poly_kernel", "poly_kernel"
            ).items()
        }
    elif combination == 'SPR':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_params, "sigmoid_kernel", "poly_kernel", "rbf_kernel"
            ).items()
        }
    elif combination == 'SRP':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_params, "sigmoid_kernel", "rbf_kernel", "poly_kernel"
            ).items()
        }
    elif combination == 'SRR':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_params, "sigmoid_kernel", "rbf_kernel", "rbf_kernel"
            ).items()
        }
    elif combination == 'RPP':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_params, "rbf_kernel", "poly_kernel", "poly_kernel"
            ).items()
        }
    elif combination == 'RPR':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_params, "rbf_kernel", "poly_kernel", "rbf_kernel"
            ).items()
        }
    elif combination == 'RRP':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_params, "rbf_kernel", "rbf_kernel", "poly_kernel"
            ).items()
        }
    elif combination == 'RRR':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_params, "rbf_kernel", "rbf_kernel", "rbf_kernel"
            ).items()
        }
    param_grid['svc__C'] = 10.0 ** np.arange(-4, 5)
    n_jobs = 8

    estimator = make_pipeline(
        DataSelector(
            kernel_directory=kernel_dir,
            identifier=f'{identifier}_{combination}',
        ),
        SVC(
            kernel='precomputed',
            probability=True,
            random_state=seed,
            cache_size=500,
        ),
    )

    return param_grid, estimator, n_jobs
