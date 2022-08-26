from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from source.utils import DataSelector
from typing import Dict, Union, Tuple, Any, List
import os
import pandas as pd
import numpy as np
from source.utils import make_kernel_combinations


def configurator(
    *,
    combination: str,
    identifier: str,
    kernel_dir: str,
) -> Tuple[Dict[str, List[Union[float, str]]], Any, int]:

    kernel_param = np.array(
        pd.read_csv(
            os.path.join('/'.join(os.getcwd().split('/')[:-1]), "data/kernel_parameter.csv")
        )
    )

    param_grid: Dict[str, List[Union[float, str]]]
    # Set up grid of parameters to optimize over
    if combination == 'SPP':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_param, "sigmoid_kernel", "poly_kernel", "poly_kernel"
            ).items()
        }
    elif combination == 'SPR':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_param, "sigmoid_kernel", "poly_kernel", "rbf_kernel"
            ).items()
        }
    elif combination == 'SRP':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_param, "sigmoid_kernel", "rbf_kernel", "poly_kernel"
            ).items()
        }
    elif combination == 'SRR':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_param, "sigmoid_kernel", "rbf_kernel", "rbf_kernel"
            ).items()
        }
    elif combination == 'RPP':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_param, "rbf_kernel", "poly_kernel", "poly_kernel"
            ).items()
        }
    elif combination == 'RPR':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_param, "rbf_kernel", "poly_kernel", "rbf_kernel"
            ).items()
        }
    elif combination == 'RRP':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_param, "rbf_kernel", "rbf_kernel", "poly_kernel"
            ).items()
        }
    elif combination == 'RRR':
        param_grid = {
            f'dataselector__{k}': v for k, v in make_kernel_combinations(
                kernel_param, "rbf_kernel", "rbf_kernel", "rbf_kernel"
            ).items()
        }
    param_grid['svc__C'] = [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4]
    n_jobs = 8

    estimator = make_pipeline(
        DataSelector(
            kernel_directory=kernel_dir,
            identifier=f'{identifier}_{combination}',
        ),
        SVC(
            kernel='precomputed',
            probability=True,
            random_state=1337,
            cache_size=500,
        ),
    )

    return param_grid, estimator, n_jobs
