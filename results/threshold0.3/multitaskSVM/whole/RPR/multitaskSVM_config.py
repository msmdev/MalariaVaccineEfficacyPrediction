from typing import Any, Dict, List, Tuple, Union

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from source.config import seed
from source.utils import DataSelector, make_kernel_combinations

kernel_params: Dict[str, np.ndarray] = {
    "SA": np.array([0.25, 0.5, 0.75]),
    "SO": np.arange(-2.0, 3.0, dtype=np.float64),
    "R0": 10.0 ** np.arange(-6, 6),
    "R1": 10.0 ** np.arange(-6, 6),
    "R2": 10.0 ** np.arange(-6, 6),
    "P1": np.arange(2, 6, dtype=int),
    "P2": np.arange(2, 6, dtype=int),
}


def configurator(
    *,
    combination: str,
    identifier: str,
    kernel_dir: str,
) -> Tuple[Dict[str, List[Union[float, int, str]]], Any]:

    param_grid: Dict[str, List[Union[float, int, str]]]
    # Set up grid of parameters to optimize over
    if combination == "SPP":
        param_grid = {
            f"dataselector__{k}": v
            for k, v in make_kernel_combinations(
                kernel_params=kernel_params,
                kernel_time_series="sigmoid_kernel",
                kernel_dosage="poly_kernel",
                kernel_abSignal="poly_kernel",
            ).items()
        }
    elif combination == "SPR":
        param_grid = {
            f"dataselector__{k}": v
            for k, v in make_kernel_combinations(
                kernel_params=kernel_params,
                kernel_time_series="sigmoid_kernel",
                kernel_dosage="poly_kernel",
                kernel_abSignal="rbf_kernel",
            ).items()
        }
    elif combination == "SRP":
        param_grid = {
            f"dataselector__{k}": v
            for k, v in make_kernel_combinations(
                kernel_params=kernel_params,
                kernel_time_series="sigmoid_kernel",
                kernel_dosage="rbf_kernel",
                kernel_abSignal="poly_kernel",
            ).items()
        }
    elif combination == "SRR":
        param_grid = {
            f"dataselector__{k}": v
            for k, v in make_kernel_combinations(
                kernel_params=kernel_params,
                kernel_time_series="sigmoid_kernel",
                kernel_dosage="rbf_kernel",
                kernel_abSignal="rbf_kernel",
            ).items()
        }
    elif combination == "RPP":
        param_grid = {
            f"dataselector__{k}": v
            for k, v in make_kernel_combinations(
                kernel_params=kernel_params,
                kernel_time_series="rbf_kernel",
                kernel_dosage="poly_kernel",
                kernel_abSignal="poly_kernel",
            ).items()
        }
    elif combination == "RPR":
        param_grid = {
            f"dataselector__{k}": v
            for k, v in make_kernel_combinations(
                kernel_params=kernel_params,
                kernel_time_series="rbf_kernel",
                kernel_dosage="poly_kernel",
                kernel_abSignal="rbf_kernel",
            ).items()
        }
    elif combination == "RRP":
        param_grid = {
            f"dataselector__{k}": v
            for k, v in make_kernel_combinations(
                kernel_params=kernel_params,
                kernel_time_series="rbf_kernel",
                kernel_dosage="rbf_kernel",
                kernel_abSignal="poly_kernel",
            ).items()
        }
    elif combination == "RRR":
        param_grid = {
            f"dataselector__{k}": v
            for k, v in make_kernel_combinations(
                kernel_params=kernel_params,
                kernel_time_series="rbf_kernel",
                kernel_dosage="rbf_kernel",
                kernel_abSignal="rbf_kernel",
            ).items()
        }
    else:
        raise ValueError(f"Unknown combination {combination} supplied.")

    param_grid["svc__C"] = 10.0 ** np.arange(-4, 5)

    estimator = make_pipeline(
        DataSelector(
            kernel_directory=kernel_dir,
            identifier=f"{identifier}_{combination}",
        ),
        SVC(
            kernel="precomputed",
            probability=True,
            random_state=seed,
        ),
    )

    return param_grid, estimator
