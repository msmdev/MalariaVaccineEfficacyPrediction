# Copyright (c) 2022 Bernhard Reuter and Jacqueline Wistuba-Hamprecht.
# ------------------------------------------------------------------------------------------------
# If you use this code or parts of it, cite the following reference:
# ------------------------------------------------------------------------------------------------
# Jacqueline Wistuba-Hamprecht and Bernhard Reuter (2022)
# https://github.com/jacqui20/MalariaVaccineEfficacyPrediction
# ------------------------------------------------------------------------------------------------
# This is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------------------------

"""
@Author: Bernhard Reuter and Jacqueline Wistuba-Hamprecht

"""

import ast
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, sigmoid_kernel
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator
from sklearn.preprocessing import KernelCenterer
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import column_or_1d


def normalize(
    X: np.ndarray,
    assert_sym: bool = False,
) -> np.ndarray:
    """Utility function (loop version) to normalize a square symmetric matrix by
    normalized_element[i, j] = element[i, j] / np.sqrt(element[i, i] * element[j, j]).

    Parameters
    ----------
    X : np.ndarray
        Square symmetric Matrix to normalize.
    assert_sym : bool, default = False
        If True, assert that the supplied Gram matrix is symmetric.

    Returns
    -------
    X_norm : np.ndarray
        Normalized square symmetric matrix.
    """
    # assert that X matrix is square
    if not X.shape[0] == X.shape[1]:
        raise ValueError("Matrix is not square.")
    if assert_sym:
        # assert that X matrix is symmetric:
        if not np.allclose(X, X.T):
            raise ValueError("Matrix isn't symmetric.")
    # check, if all diagonal elements are positive:
    if np.any(np.diag(X) <= 0.0):
        raise ValueError("Some diagonal elements of the matrix are <=0. Can't normalize.")
    X_norm = np.zeros(X.shape, dtype=np.double)
    for i in range(X.shape[0]):
        for j in range(i, X.shape[1]):
            X_norm[i, j] = X[i, j] / np.sqrt(X[i, i] * X[j, j])
            X_norm[j, i] = X_norm[i, j]
    return X_norm


def check_complex_or_negative(
    ev: np.ndarray,
    warn: bool = True,
) -> Tuple[bool, bool]:
    complex = False
    negative = False
    if np.any(np.iscomplex(ev)):
        if np.max(np.abs(ev.imag)) > np.finfo(ev.dtype).eps:
            if warn:
                print(
                    "Complex eigenvalues with significant imaginary part found:\n"
                    f"{ev[np.abs(ev.imag) > np.finfo(ev.dtype).eps]}"
                )
            complex = True
    if np.any(ev.real < 0.0):
        if warn:
            print("Eigenvalues with negative real parts found:\n" f"{ev[ev.real < 0.0]}")
        negative = True
    return complex, negative


def make_symmetric_matrix_psd(
    X: np.ndarray,
    epsilon_factor: float = 1.0e3,
    iterations: int = 1000,
) -> Tuple[np.ndarray, List[float], List[str], str]:
    """Spectral Translation approach
    Tests, if a given symmetric matrix is positive semi-definite (psd) and, if not,
    a damping value `c` is iteratively (up to `iterations + 1` iterations) added to the matrix
    diagonal, i.e., `X = X + c * I`, until the kernel matrix becomes positive semi-definite.
    Non-positive-semi-definiteness will be tested via a spectral criterion:
    If `X` has negative or complex eigenvalues, `X` can not be psd.
    The damping value `c` is derived from the spectrum of `X`: If negative eigenvalues exist,
    `c = max([epsilon_factor * np.finfo(X.dtype).eps, np.min(np.linalg.eigvals(X).real)])`,
    else,if complex but non-negative eigenvalues exist,
    `c = max([epsilon_factor * np.finfo(X.dtype).eps, np.max(np.abs(eigenvalues.imag))])`.

    Parameters
    ----------
    X: np.ndarray
        Symmetric matrix to test and make psd, if necessary.
    epsilon_factor : float, default = 1e3
        Factor `epsilon_factor` to multiply the machine precision `np.finfo(X.dtype).eps` with,
        resulting in `epsilon = epsilon_factor * np.finfo(X.dtype).eps` being the mimimum value
        to add to the diagonal of `X`, if `X` is not psd.
    iterations : int, default = 1000
        `iterations + 1` is the maximal number of iterations (of adding a damping value)
        to perform before stopping and returning the resulting matrix,
        even if the matrix coultn't be made psd.

    Returns
    -------
    X_psd : np.ndarray
        Symmetric positive semi-definite matrix.
    c_list : list
        List of damping_values `c` added to the diagonal.
    info_list : list
        List of strings telling, if the damping was applied due to a negative real part
        ('negative') or a imaginary part ('imaginary') of an eigenvalue. If the damping value `c`
        is smaller than `epsilon`, i.e. `c < epsilon = epsilon_factor * np.finfo(X.dtype).eps`,
        a str '_epsilon' will be appended resulting in 'negative_epsilon' or 'imaginary_epsilon'.
    warning : str, default = ''
        If the matrix couldn't be made psd and the returned matrix has negative eigenvalues,
        a warning message will be returned. Otherwise, `warning = ''`.
    """
    epsilon = epsilon_factor * np.finfo(X.dtype).eps
    c_list = []
    info_list = []
    warning = ""

    # check, if X is square
    if X.shape[0] != X.shape[1]:
        raise ValueError("Matrix is not square.")

    # check, if X is symmetric:
    if not np.allclose(X, X.T):
        raise ValueError("Matrix is not symmetric.")

    eigenvalues = np.linalg.eigvals(X)

    # check, if all eigenvalues are real and non-negative
    complex, negative = check_complex_or_negative(eigenvalues)

    if complex or negative:
        print(
            "Matrix is not positive semi-definite.\n" "Will try to make it positive semi-definite."
        )

        n_negative = 0
        n_imaginary = 0
        counter = 0
        while (complex or negative) and counter <= iterations:
            counter += 1
            if negative:
                n_negative += 1
                info = "negative"
                c = np.abs(np.min(eigenvalues).real)
            else:
                n_imaginary += 1
                info = "imaginary"
                c = np.max(np.abs(eigenvalues.imag))
            if c < epsilon:
                info += "_epsilon"
                c = epsilon
            info_list.append(info)
            c_list.append(c)
            np.fill_diagonal(X, np.diag(X) + c)
            eigenvalues = np.linalg.eigvals(X)
            complex, negative = check_complex_or_negative(eigenvalues, warn=False)
        # if counter == iterations:
        #     counter += 1
        #     c = 1.0
        #     temp = X + np.diag([c for i in range(X.shape[0])])
        #     temp_eigenvalues = np.linalg.eigvals(temp)
        #     temp_complex, temp_negative = check_complex_or_negative(temp_eigenvalues, warn=False)
        #     if not (temp_complex or temp_negative):
        #         if negative:
        #             n_negative += 1
        #             info = 'negative_1.0'
        #         else:
        #             n_imaginary += 1
        #             info = 'imaginary_1.0'
        #         info_list.append(info)
        #         c_list.append(c)
        #         X = temp
        #         eigenvalues = temp_eigenvalues

        complex, negative = check_complex_or_negative(eigenvalues)
        if complex or negative:
            print(
                "Couldn't make matrix positive semi-definite by adding "
                f"sum_c={np.sum(c_list)} in {counter} steps to its diagonal.\n"
                f"Damped {n_negative} times for negative eigenvalues "
                f"and {n_imaginary} times for imaginary parts."
            )
            if negative:
                warning = (
                    "Couldn't make matrix positive semi-definite by adding "
                    f"sum_c={np.sum(c_list)} in {counter} steps to its diagonal.\n"
                    f"Damped {n_negative} times for negative eigenvalues "
                    f"and {n_imaginary} times for imaginary parts.\n"
                    "CAUTION: The returned matrix has negative eigenvalues."
                )
        else:
            print(
                "Made matrix positive semi-definite by adding "
                f"sum_c={np.sum(c_list)} in {counter} steps to its diagonal.\n"
                f"Damped {n_negative} times for negative eigenvalues "
                f"and {n_imaginary} times for imaginary parts."
            )
    return X, c_list, info_list, warning


# TODO: Fix description
def assign_folds(
    *,
    labels: np.ndarray,
    groups: np.ndarray,
    delta: int,
    step: int,
    n_splits: int = 5,
    shuffle: bool = True,
    print_info: bool = True,
    random_state: Optional[Union[int, np.random.mtrand.RandomState]] = None,
):
    """
    Provides 1-D arrays of train/test indices used in CustomPredefinedSplit
    to split data into train/test sets.
    This routine is designed for time-ordered data that consists of samples
    taken at several timepoints that belong to the same group (e.g. patient).
    Consider, e.g., 40 patients that underwent antibody measurements against
    malaria at multiple (3) time-points after vaccination. Thus we have 120
    samples at 3 timepoints in 40 groups. We now want to split the data
    into disjunct train/test sets, such that the samples in the test sets
    are all taken from the same time point, while the samples in the train
    sets are taken from all timepoints under the constraint that patients
    (groups) appearing in the respective test set don't appear in the
    associated train set.

    CAUTION: This routine is only tested for the given Malaria data.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        Target labels. Used for stratification.
    groups : array-like of shape (n_samples,)
        Group labels for the samples used while splitting the dataset into
        train/test set.
    delta : int
        Step-width. ``len(labels) % delta = 0`` must be ensured.
    step : int
        Step. Used together with ``delta`` to select the interval the test
        samples are selected from.
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=True
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    print_info : bool, default=True
        If True, printout detailed information about the train/test splits.
    random_state : int, RandomState instance or None, default=None
        When shuffle is True, random_state affects the ordering of the indices,
        which controls the randomness of each fold for each class. Otherwise,
        leave random_state as None. Pass an int for reproducible output across
        multiple function calls.

    Returns
    -------
    test_fold : np.ndarray of shape (n_samples,)
        The entry ``test_fold[i]`` represents the index of the test set that
        sample ``i`` belongs to. It is possible to exclude sample ``i`` from
        any test set by setting ``test_fold[i]`` equal to -1.
    train_fold : np.ndarray of shape (n_samples,)
        The entry ``train_fold[i]`` represents the index of the train set that
        sample ``i`` is excluded from. It is possible to include sample ``i``
        in any test set by setting ``train_fold[i]`` equal to -1.
    """
    labels = column_or_1d(labels)
    groups = column_or_1d(groups)
    if not len(labels) == len(groups):
        raise ValueError("labels and groups arrays must be of same length.")
    if not len(labels) % delta == 0:
        raise ValueError("len(labels) % delta = 0 must be ensured.")
    if not delta + delta * step <= len(labels):
        raise ValueError("delta + delta * step <= len(labels) must be ensured")
    labels_slice = labels[delta * step : delta + delta * step]
    groups_slice = groups[delta * step : delta + delta * step]
    skf = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
    test_fold = np.array([-1 for i in range(len(labels))])
    train_fold = np.array([-1 for i in range(len(labels))])
    for i, (train_index, test_index) in enumerate(
        skf.split(np.zeros((delta, delta)), labels_slice)
    ):
        exclude_groups = []
        for j in test_index:
            test_fold[j + delta * step] = i
            exclude_groups.append(groups_slice[j])
        train_fold = np.where(np.isin(groups, exclude_groups), i, train_fold)
        if print_info:
            print("TEST:", test_index + delta * step)
            print("exclude groups:", exclude_groups)
            print("train_fold:", train_fold)
    if print_info:
        print("test_fold:", test_fold)
        cps = CustomPredefinedSplit(test_fold, train_fold)
        for i, (train_index, test_index) in enumerate(cps.split()):
            print(
                f"TRAIN (len={len(train_index)}): {train_index} "
                f"TEST (len={len(test_index)}): {test_index}"
            )
        print("")
    return test_fold, train_fold


class CustomPredefinedSplit(BaseCrossValidator):
    """Predefined split cross-validator
    Provides train/test indices to split data into train/test sets using a
    predefined scheme specified by the user with the ``test_fold`` parameter.

    Parameters
    ----------
    test_fold : array-like of shape (n_samples,)
        The entry ``test_fold[i]`` represents the index of the test set that
        sample ``i`` belongs to. It is possible to exclude sample ``i`` from
        any test set by setting ``test_fold[i]`` equal to -1.
    train_fold : array-like of shape (n_samples,)
        The entry ``train_fold[i]`` represents the index of the train set that
        sample ``i`` is excluded from. It is possible to include sample ``i``
        in any test set by setting ``train_fold[i]`` equal to -1.
    """

    def __init__(self, test_fold, train_fold):
        self.test_fold = np.array(test_fold, dtype=int)
        self.test_fold = column_or_1d(self.test_fold)
        self.train_fold = np.array(train_fold, dtype=int)
        self.train_fold = column_or_1d(self.train_fold)
        if not len(self.test_fold) == len(self.train_fold):
            raise ValueError("test_fold and train_fold must be of equal length.")
        unique_test_folds = np.unique(self.test_fold)
        unique_test_folds = unique_test_folds[unique_test_folds != -1]
        unique_train_folds = np.unique(self.train_fold)
        unique_train_folds = unique_train_folds[unique_train_folds != -1]
        if not np.array_equal(unique_test_folds, unique_train_folds):
            raise ValueError("test_fold and train_fold must have the same fold indices.")
        self.unique_folds = unique_test_folds

    def split(self, X=None, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        ind = np.arange(len(self.test_fold))
        for test_index, train_index in self._iter_test_masks():
            train_index = ind[train_index]
            test_index = ind[test_index]
            yield train_index, test_index

    def _iter_test_masks(self):
        """Generates boolean masks corresponding to test sets."""
        for f in self.unique_folds:
            test_index = np.where(self.test_fold == f)[0]
            test_mask = np.zeros(len(self.test_fold), dtype=bool)
            test_mask[test_index] = True
            train_index = np.where(self.train_fold == f)[0]
            train_mask = np.ones(len(self.train_fold), dtype=bool)
            train_mask[train_index] = False
            yield test_mask, train_mask

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator
        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return len(self.unique_folds)


class DataSelector:
    def __init__(
        self,
        *,
        kernel_directory: str,
        identifier: str,
        SA: Union[float, str] = "X",
        SO: Union[float, str] = "X",
        R0: Union[float, str] = "X",
        R1: Union[float, str] = "X",
        R2: Union[float, str] = "X",
        P1: Union[int, str] = "X",
        P2: Union[int, str] = "X",
    ) -> None:
        self.kernel_directory = kernel_directory
        self.identifier = identifier
        self.SA = SA
        self.SO = SO
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2
        self.P1 = P1
        self.P2 = P2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DataSelector":
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        # Return the classifier
        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if isinstance(self.R0, float):
            if isinstance(self.R1, float) and isinstance(self.R2, float):
                fn = (
                    f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0:.1E}_"
                    f"R1_{self.R1:.1E}_R2_{self.R2:.1E}_P1_{self.P1}_P2_{self.P2}.npy"
                )
            elif isinstance(self.R1, float) and not isinstance(self.R2, float):
                fn = (
                    f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0:.1E}_"
                    f"R1_{self.R1:.1E}_R2_{self.R2}_P1_{self.P1}_P2_{self.P2}.npy"
                )
            elif not isinstance(self.R1, float) and isinstance(self.R2, float):
                fn = (
                    f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0:.1E}_"
                    f"R1_{self.R1}_R2_{self.R2:.1E}_P1_{self.P1}_P2_{self.P2}.npy"
                )
            else:
                fn = (
                    f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0:.1E}_"
                    f"R1_{self.R1}_R2_{self.R2}_P1_{self.P1}_P2_{self.P2}.npy"
                )
        else:
            if isinstance(self.R1, float) and isinstance(self.R2, float):
                fn = (
                    f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0}_"
                    f"R1_{self.R1:.1E}_R2_{self.R2:.1E}_P1_{self.P1}_P2_{self.P2}.npy"
                )
            elif isinstance(self.R1, float) and not isinstance(self.R2, float):
                fn = (
                    f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0}_"
                    f"R1_{self.R1:.1E}_R2_{self.R2}_P1_{self.P1}_P2_{self.P2}.npy"
                )
            elif not isinstance(self.R1, float) and isinstance(self.R2, float):
                fn = (
                    f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0}_"
                    f"R1_{self.R1}_R2_{self.R2:.1E}_P1_{self.P1}_P2_{self.P2}.npy"
                )
            else:
                fn = (
                    f"{self.identifier}_SA_{self.SA}_SO_{self.SO}_R0_{self.R0}_"
                    f"R1_{self.R1}_R2_{self.R2}_P1_{self.P1}_P2_{self.P2}.npy"
                )

        kernel = np.load(os.path.join(self.kernel_directory, fn))

        # kernel is a precomputed square kernel matrix
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Oligo kernel should be a square kernel matrix")

        return np.ascontiguousarray(kernel.flatten("C")[X.flatten("C")].reshape(X.shape))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "kernel_directory": self.kernel_directory,
            "identifier": self.identifier,
            "SA": self.SA,
            "SO": self.SO,
            "R0": self.R0,
            "R1": self.R1,
            "R2": self.R2,
            "P1": self.P1,
            "P2": self.P2,
        }

    def set_params(
        self,
        **parameters: Any,
    ) -> "DataSelector":
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def get_parameters(
    timepoint_results: pd.DataFrame,
    model: str,
) -> Dict[str, Union[float, int, str]]:
    """Return combination of parameters to initialize RLR.

    Parameters
    ----------
    timepoint_results : pd.DataFrame
        DataFrame containing optimal parameters and mean AUROC values
        for a particular time point as found via Repeated Grid-Search CV (RGSCV).
    model : str
        Model ('multitaskSVM', 'RF' or 'RLR') to select parameters for.

    Returns
    --------
    params : dict
        Parameter dictionary.
    """
    roc_results = timepoint_results[timepoint_results["scoring"].isin(["roc_auc"])]
    if not roc_results.shape == (1, 4):
        raise ValueError(f"roc_results.shape != (1, 4): {roc_results.shape} != (1, 4)")
    params_string = roc_results["best_params"].iloc[0]
    if not type(params_string) == str:
        raise ValueError(f"type(params_string) != str: {type(params_string)} != str")
    params = ast.literal_eval(params_string)
    if model == "RLR":
        from source.RLR_config import param_grid

        keys = param_grid.keys()
        if not set(params.keys()) == keys:
            raise ValueError(
                f"Expected RLR parameters but set(params.keys()) != {keys}: "
                f"{set(params.keys())} != {keys}"
            )
    elif model == "RF":
        from source.RF_config import param_grid

        keys = param_grid.keys()
        if not set(params.keys()) == keys:
            raise ValueError(
                f"Expected RF parameters but set(params.keys()) != {keys}: "
                f"{set(params.keys())} != {keys}"
            )
    elif model == "multitaskSVM":
        from source.multitaskSVM_config import configurator

        param_grid, _ = configurator(
            combination="RPR",  # see *
            identifier="",
            kernel_dir="",
        )
        # *You can supply any legal combination here,
        # since the returned param_grid isn't actually used (only the keys are used).
        keys = param_grid.keys()
        if not set(params.keys()) == keys:
            raise ValueError(
                f"Expected multitaskSVM parameters but set(params.keys()) != {keys}: "
                f"{set(params.keys())} != {keys}"
            )
    else:
        raise ValueError("`model` must be either 'RF', 'RLR' or 'multitaskSVM'.")
    return params


def select_timepoint(rgscv_results: pd.DataFrame, timepoint: str) -> pd.DataFrame:
    """Select time point to evaluate informative features from RLR.

    Parameter
    ---------
    rgscv_results : pd.DataFrame
        DataFrame containing optimal parameters and mean AUROC values
        per time point as found via Repeated Grid-Search CV (RGSCV).

    timepoint : str
        Time point to extract parameters and AUROC values for.

    Returns
    --------
    timepoint_results: pd.DataFrame
        DataFrame containing optimal parameters and mean AUROC values
        for the selected time point as found via Repeated Grid-Search CV (RGSCV).
    """
    timepoint_results = rgscv_results[rgscv_results["time"].isin([timepoint])]
    return timepoint_results


def make_kernel_combinations(
    *,
    kernel_params: Dict[str, np.ndarray],
    kernel_time_series: str,
    kernel_dosage: str,
    kernel_abSignal: str,
) -> Dict[str, List[Union[float, int, str]]]:
    """Make dictionary of kernel combination values

    For each kernel parameter a range of values is defined.

    Parameters
    ----------
    kernel_params : Dict[str, np.ndarray]
        Dict with ranges of kernel parameters.
        Expected is a dict with keys {'SA', 'SO', 'R0', 'R1', 'R2', 'P1', 'P2'}.
        `kernel_params['SA']` defines the value range of the factor `gamma` and
        `kernel_params['SO']` defines the value range of the offset parameter `coef0`
        of the time-series sigmoid kernel `K(X, Y) = tanh(gamma <X, Y> + coef0)`;
        `kernel_params['R1']` (`kernel_params['R2']` (`kernel_params['R3']`)) defines
        the range of the power `d` of the `gamma` (`gamma = 10^d`) parameter of the
        time-series (dosage (AB-signal)) RBF kernel `K(x, y) = exp(-gamma ||x-y||^2)`;
        `kernel_params['P1']` (`kernel_params['P2']`) defines the range of values for
        the degree parameter of the dosage (AB-Signal) polynomial kernel
        `K(X, Y) = (gamma <X, Y> + coef0)^degree`.
    kernel_time_series : str
        Kernel function to compute relationship between individuals based on time points.
    kernel_dosage : str
        Kernel function for to represent relationship between individuals
        based on dosage level.
    kernel_abSignal : str
        Kernel function for to represent relationship between individuals
        based on ab signal intensity

    Returns
    -------
    kernel_comb_param : dict
        Dictionary of value ranges for each kernel parameter.

    """

    # check for correct parameter keys:
    keys = {"SA", "SO", "R0", "R1", "R2", "P1", "P2"}
    if not set(kernel_params.keys()) == keys:
        raise ValueError(
            f"Expected multitaskSVM parameters but set(params.keys()) != {keys}: "
            f"{set(kernel_params.keys())} != {keys}"
        )
    # assert that the types of the parameter values are correct:
    for key in ["SA", "SO", "R0", "R1", "R2"]:
        if not isinstance(kernel_params[key], np.ndarray):
            raise ValueError(f"kernel_params[{key}] must be a np.ndarray.")
        if not kernel_params[key].ndim == 1:
            raise ValueError(f"kernel_params[{key}] must be one-dimensional.")
        if not (kernel_params[key].dtype == float or kernel_params[key].dtype.type == np.str_):
            raise ValueError(
                f"All elements of kernel_params[{key}] must be either of type str or float."
            )
    for key in ["P1", "P2"]:
        if not isinstance(kernel_params[key], np.ndarray):
            raise ValueError(f"kernel_params[{key}] must be a np.ndarray.")
        if not kernel_params[key].ndim == 1:
            raise ValueError(f"kernel_params[{key}] must be one-dimensional.")
        if not (kernel_params[key].dtype == int or kernel_params[key].dtype.type == np.str_):
            raise ValueError(
                f"All elements of kernel_params[{key}] must be either of type str or int."
            )

    kernel_comb_param: Dict[str, np.ndarray]
    if (
        kernel_time_series == "sigmoid_kernel"
        and kernel_dosage == "rbf_kernel"
        and kernel_abSignal == "rbf_kernel"
    ):
        kernel_comb_param = {
            "SA": kernel_params["SA"],
            "SO": kernel_params["SO"],
            "R0": np.array(["X"]),
            "R1": kernel_params["R1"],
            "R2": kernel_params["R2"],
            "P1": np.array(["X"]),
            "P2": np.array(["X"]),
        }
    elif (
        kernel_time_series == "sigmoid_kernel"
        and kernel_dosage == "poly_kernel"
        and kernel_abSignal == "poly_kernel"
    ):
        kernel_comb_param = {
            "SA": kernel_params["SA"],
            "SO": kernel_params["SO"],
            "R0": np.array(["X"]),
            "R1": np.array(["X"]),
            "R2": np.array(["X"]),
            "P1": kernel_params["P1"],
            "P2": kernel_params["P2"],
        }
    elif (
        kernel_time_series == "sigmoid_kernel"
        and kernel_dosage == "poly_kernel"
        and kernel_abSignal == "rbf_kernel"
    ):
        kernel_comb_param = {
            "SA": kernel_params["SA"],
            "SO": kernel_params["SO"],
            "R0": np.array(["X"]),
            "R1": np.array(["X"]),
            "R2": kernel_params["R2"],
            "P1": kernel_params["P1"],
            "P2": np.array(["X"]),
        }
    elif (
        kernel_time_series == "sigmoid_kernel"
        and kernel_dosage == "rbf_kernel"
        and kernel_abSignal == "poly_kernel"
    ):
        kernel_comb_param = {
            "SA": kernel_params["SA"],
            "SO": kernel_params["SO"],
            "R0": np.array(["X"]),
            "R1": kernel_params["R1"],
            "R2": np.array(["X"]),
            "P1": np.array(["X"]),
            "P2": kernel_params["P2"],
        }
    elif (
        kernel_time_series == "rbf_kernel"
        and kernel_dosage == "rbf_kernel"
        and kernel_abSignal == "rbf_kernel"
    ):
        kernel_comb_param = {
            "SA": np.array(["X"]),
            "SO": np.array(["X"]),
            "R0": kernel_params["R0"],
            "R1": kernel_params["R1"],
            "R2": kernel_params["R2"],
            "P1": np.array(["X"]),
            "P2": np.array(["X"]),
        }
    elif (
        kernel_time_series == "rbf_kernel"
        and kernel_dosage == "poly_kernel"
        and kernel_abSignal == "poly_kernel"
    ):
        kernel_comb_param = {
            "SA": np.array(["X"]),
            "SO": np.array(["X"]),
            "R0": kernel_params["R0"],
            "R1": np.array(["X"]),
            "R2": np.array(["X"]),
            "P1": kernel_params["P1"],
            "P2": kernel_params["P2"],
        }
    elif (
        kernel_time_series == "rbf_kernel"
        and kernel_dosage == "poly_kernel"
        and kernel_abSignal == "rbf_kernel"
    ):
        kernel_comb_param = {
            "SA": np.array(["X"]),
            "SO": np.array(["X"]),
            "R0": kernel_params["R0"],
            "R1": np.array(["X"]),
            "R2": kernel_params["R2"],
            "P1": kernel_params["P1"],
            "P2": np.array(["X"]),
        }
    elif (
        kernel_time_series == "rbf_kernel"
        and kernel_dosage == "rbf_kernel"
        and kernel_abSignal == "poly_kernel"
    ):
        kernel_comb_param = {
            "SA": np.array(["X"]),
            "SO": np.array(["X"]),
            "R0": kernel_params["R0"],
            "R1": kernel_params["R1"],
            "R2": np.array(["X"]),
            "P1": np.array(["X"]),
            "P2": kernel_params["P2"],
        }
    else:
        raise ValueError("Illegal input.")

    return {k: v.tolist() for k, v in kernel_comb_param.items()}


def multitask(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Multitask approach

    Combination of two kernel matrices are combined by
    element-wise multiplication to one kernel matrix.

    Parameters
    ----------
    a : np.ndarray
        Kernel matrix a.
    b : np.ndarray
        Kernel matrix b.

    Returns
    -------
    a * b : np.ndarray
        Element-wise multiplicated kernel matrix a * b.
    """
    return a * b


def make_kernel_matrix(
    *,
    AB_signals: pd.DataFrame,
    time_series: pd.Series,
    dose: pd.Series,
    model: Dict[str, Union[float, int, str]],
    kernel_time_series: str,
    kernel_dosage: str,
    kernel_abSignals: str,
    scale: bool = False,
) -> Tuple[np.ndarray, List[float], List[str]]:
    """Compute combined kernel matrix.

    for each task, time, PfSPZ dose and antibody signal intensity a kernel matrix
    is pre-computed given the pre-defined kernel function and combined to one
    matrix by the multitask approach.

    Parameters
    ----------
    AB_signals : pd.Dataframe
        Pre-processed proteome-microarray-based antibody signal intensities.
    time_series : pd.Series
        Series of time points at which the antibody signal intensities were recorded.
    dose : pd.Series
        Series of dosages administered.
    model : dict
        Dict of kernel parameters.
        Expected is a dict with keys `{'SA', 'SO', 'R0', 'R1', 'R2', 'P1', 'P2'}`.
        `model['SA']` defines the value of the factor `gamma` and `model['SO']` defines
        the value of the offset parameter `coef0` of the time-series sigmoid kernel
        `K(X, Y) = tanh(gamma <X, Y> + coef0)`;
        `model['R1']` (`model['R2']` (`model['R3']`)) defines the power `d` of the
        `gamma` (`gamma = 10^d`) parameter of the time-series (dosage (AB-signal)) RBF kernel
        `K(x, y) = exp(-gamma ||x-y||^2)`;
        `model['P1']` (`model['P2']`) defines the value of the degree parameter of the dosage
        (AB-Signal) polynomial kernel `K(X, Y) = (gamma <X, Y> + coef0)^degree`.
    kernel_time_series : str
        Kernel function to compute relationship between individuals based on time points.
    kernel_dosage : str
        Kernel function to compute relationship between individuals based on dosage level.
    kernel_abSignals : str
        Kernel function to compute relationship between individuals
        based on antibody signal intensity.
    scale : bool, default=False
        If True, scale the combined matrix by first centering it followed by normalization.

    Returns
    -------
    multitaskMatrix : np.ndarray
        Returns a combined matrix.
    c_list : list
        List of damping_values `c` added to the diagonal.
    info_list : list
        List of strings telleing, if the damping was applied due to a negative real part
        ('negative') or a imaginary part ('imaginary') of an eigenvalue. If the damping value `c`
        is smaller than `epsilon`, i.e. `c < epsilon = epsilon_factor * np.finfo(X.dtype).eps`,
        a str '_epsilon' will be appended resulting in 'negative_epsilon' or 'imaginary_epsilon'.
    """

    # model input parameters sanity checks:
    if not isinstance(model, dict):
        raise ValueError("The model must be given as a dict of kernel parameters.")

    # check for correct parameter keys:
    keys = {"SA", "SO", "R0", "R1", "R2", "P1", "P2"}
    if not set(model.keys()) == keys:
        raise ValueError(
            f"Expected multitaskSVM parameters but set(model.keys()) != {keys}: "
            f"{set(model.keys())} != {keys}"
        )
    for key in ["SA", "SO", "R0", "R1", "R2"]:
        if not isinstance(model[key], (float, str)):
            raise ValueError(f"model[{key}] must be either of type str or float.")
    for key in ["P1", "P2"]:
        if not isinstance(model[key], (int, str)):
            raise ValueError(f"model[{key}] must be either of type str or int.")

    # pre-compute kernel matrix of time points K(n_t,n_t')
    if kernel_time_series == "sigmoid_kernel":
        if isinstance(model["SO"], float):
            time_series_kernel_matrix = sigmoid_kernel(
                time_series.to_numpy().reshape(len(time_series), 1),
                gamma=model["SA"],
                coef0=model["SO"],
            )
        else:
            raise ValueError("Illegal input.")
    elif kernel_time_series == "rbf_kernel":
        time_series_kernel_matrix = rbf_kernel(
            time_series.to_numpy().reshape(len(time_series), 1), gamma=model["R0"]
        )
    else:
        raise ValueError("Illegal input.")

    # pre-compute kernel matrix of dosage K(n_d,n_d')
    if kernel_dosage == "rbf_kernel":
        dose_kernel_matrix = rbf_kernel(dose.to_numpy().reshape(len(dose), 1), gamma=model["R1"])
    elif kernel_dosage == "poly_kernel":
        if isinstance(model["P1"], int):
            dose_kernel_matrix = polynomial_kernel(
                dose.to_numpy().reshape(len(dose), 1), degree=model["P1"]
            )
        else:
            raise ValueError("Illegal input.")
    else:
        raise ValueError("Illegal input.")

    # pre-compute kernel matrix of antibody reactivity K(n_p,n_p')
    if kernel_abSignals == "rbf_kernel":
        AB_signals_kernel_matrix = rbf_kernel(AB_signals, gamma=model["R2"])
    elif kernel_abSignals == "poly_kernel":
        if isinstance(model["P2"], int):
            AB_signals_kernel_matrix = polynomial_kernel(AB_signals, degree=model["P2"])
        else:
            raise ValueError("Illegal input.")
    else:
        raise ValueError("Illegal input.")

    # pre-compute multitask kernel matrix K((np, nt),(np', nt'))
    multi_AB_signals_time_series_kernel_matrix = multitask(
        AB_signals_kernel_matrix,
        time_series_kernel_matrix,
    )

    # pre-compute multitask kernel matrix K((np, nt, nd),(np', nt', nd'))
    multitaskMatrix, c_list, info_list, message = make_symmetric_matrix_psd(
        multitask(
            multi_AB_signals_time_series_kernel_matrix,
            dose_kernel_matrix,
        )
    )
    if c_list:
        print("multitask Gram matrix had to be corrected.\n" f"model: {model}")
    if len(message) > 0 and kernel_time_series != "sigmoid_kernel":
        warnings.warn(
            "Correction failed for a multitask Gram matrix with non-sigmoid "
            f"time-series kernel (model: {model}):\n{message}"
        )

    if scale:
        multitaskMatrix, warn, _, message = make_symmetric_matrix_psd(
            normalize(KernelCenterer().fit_transform(multitaskMatrix))
        )
        if warn:
            print("Scaled multitask Gram matrix had to be corrected.\n" f"model: {model}")
        if np.max(np.diag(multitaskMatrix)) > 1.1:
            warnings.warn(
                "Scaled multitask Gram matrix "
                "kernel had diagonal elements > 1.1.\n"
                f"model: {model}"
            )
        if len(message) > 0 and kernel_time_series != "sigmoid_kernel":
            warnings.warn(
                "Correction failed for a scaled multitask Gram matrix with non-sigmoid "
                f"time-series kernel (model: {model}):\n{message}"
            )

    # proof Dimension and rank of kernel matrix
    print("Dimension of final multitask Gram matrix:")
    print(multitaskMatrix.shape)
    print(
        "Rank of final multitask Gram matrix (Returns matrix rank of array using SVD method. "
        "Rank of the array is the number of singular values of the array that are greater than a "
        "tolerance tol. May declare a matrix M rank deficient even if the linear combination of "
        "some columns of M is not exactly equal to another column of M but only numerically very "
        "close to another column of M.):"
    )
    print(np.linalg.matrix_rank(multitaskMatrix))
    print("\n\n")

    return multitaskMatrix, c_list, info_list
