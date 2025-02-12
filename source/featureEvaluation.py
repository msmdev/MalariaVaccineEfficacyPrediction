# Copyright (c) 2022 Jacqueline Wistuba-Hamprecht and Bernhard Reuter.
# ------------------------------------------------------------------------------------------------
# If you use this code or parts of it, cite the following reference:
# ------------------------------------------------------------------------------------------------
# Jacqueline Wistuba-Hamprecht and Bernhard Reuter (2022)
# https://github.com/msmdev/MalariaVaccineEfficacyPrediction
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
This module contains the main functionality of the ESPY approach.

@Author: Jacqueline Wistuba-Hamprecht and Bernhard Reuter

"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from source.config import seed
from source.utils import make_kernel_matrix, precision_recall_auc


def make_feature_combination(
    *,
    X: pd.DataFrame,
    upperValue: int,
    lowerValue: int,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, List[np.ndarray]]]:
    """Generate vector of feature combination.

    Generate for each single feature a vector based on upper- and lower quantile value.

    Parameter
    ---------
    X : pd.DataFrame
        Data (n_samples x n_features) used to calculate the upper and lower percentile and median.
    upperValue : int
        Upper percentile given as int.
    lowerValue : int
        Lower percentile given as int.

    Returns
    --------
    statistics : pd.Dataframe
        Statistics (median, lower and upper percentile) of features.
    median : np.ndarray
        The median (over the samples) vector `m`.
    combinations : dict
        Dict of feature combinations.
        `combinations["lower_combinations"]` is a list of length n_features,
        with `combinations["lower_combinations"][i]` equal the median vector
        but having the i-th element `m[i]` replaced by the lower percentile,
        `combinations["upper_combinations"]` is a list of length n_features,
        with `combinations["upper_combinations"][i]` equal the median vector
        but having the i-th element `m[i]` replaced by the upper percentile.
    """
    if not isinstance(upperValue, int):
        raise ValueError("`upperValue` must be int")
    if not isinstance(lowerValue, int):
        raise ValueError("`lowerValue` must be int")
    if not (0 <= upperValue <= 100):
        raise ValueError("`upperValue` must be in [0, 100]")
    if not (0 <= lowerValue <= upperValue):
        raise ValueError("`lowerValue` must be in [0, upperValue]")

    statistics = X.median(axis=0).to_frame(name="Median")
    statistics["UpperQuantile"] = X.quantile(float(upperValue) / 100.0)
    statistics["LowerQuantile"] = X.quantile(float(lowerValue) / 100.0)
    statistics = statistics.T

    median = statistics.loc["Median", :].to_numpy().copy()
    lower_quantile = statistics.loc["LowerQuantile", :].to_numpy().copy()
    upper_quantile = statistics.loc["UpperQuantile", :].to_numpy().copy()

    temp_lq = []
    temp_uq = []
    for i in range(len(median)):
        temp1 = median.copy()
        temp1[i] = upper_quantile[i]
        temp_uq.append(temp1)

        temp2 = median.copy()
        temp2[i] = lower_quantile[i]
        temp_lq.append(temp2)

    combinations = {
        "lower_combinations": temp_lq,
        "upper_combinations": temp_uq,
    }

    return statistics, median, combinations


def compute_distance_hyper(
    *,
    median: np.ndarray,
    combinations: Dict[str, List[np.ndarray]],
    model: SVC,
    labels: List[str],
    data: Optional[pd.DataFrame] = None,
    kernel_parameters: Optional[Dict[str, Union[float, int, str]]] = None,
    multitask: bool = False,
) -> pd.DataFrame:
    """Evaluate distance of each single feature to the classification boundary.

    Compute distance of support vectors to classification boundary for each feature
    and the change of each feature by upper and lower quantile on proteome data.

    Parameter
    ---------
    median : np.ndarray
        Median (over the samples) vector.
    combinations : list
        List of combinations of feature values and their upper and lower percentile.
    model : sklearn.svm.SVC
        SVC model.
    labels : list
        List of feature labels.
    data : pd.DataFrame, default=None
        Dataset from which to compute the multitask Gram matrix. Must be the same dataset
        that was used to train the multitaskSVM model. Typically, this is the full dataset
        (n x n_features) combined from all timepoints (n = n_times x n_samples).
    kernel_parameters : dict, default=None
        Combination of kernel parameters.
    multitask: bool, default=False
        If True, it will be assumed that the given sklearn.svm.SVC model is
        based on a precomputed multitask Gram matrix, i.e., SVC(kernel='precomputed').

    Returns
    --------
    get_distance_df : pd.DataFrame
        Dataframe of distance values for each feature per time point.
    """

    # empty array for lower and upper quantile
    get_distance_lower = []
    get_distance_upper = []

    # calc distances for all combinations
    d_cons = np.ndarray
    for m in range(len(median)):
        if not multitask:
            get_distance_lower.append(
                model.decision_function(combinations["lower_combinations"][m].reshape(1, -1))[0]
            )
            get_distance_upper.append(
                model.decision_function(combinations["upper_combinations"][m].reshape(1, -1))[0]
            )

            d_cons = model.decision_function(median.reshape(1, -1))
            # shape of distance should always be (n_samples, n_classes * (n_classes-1) / 2) =
            # (n_samples,) in the binary case, even if decision_function_shape=’ovr’,
            # i.e., the documentation is wrong (at least for v1.0.2) on
            # https://scikit-learn.org/1.0/modules/generated/sklearn.svm.SVC.html
            if not d_cons.shape == (1,):
                raise ValueError(f"Shape of distance is {d_cons.shape}, but should be (1,)")

        else:
            if isinstance(data, pd.DataFrame) and isinstance(kernel_parameters, dict):
                expected_keys = {"SA", "SO", "R0", "R1", "R2", "P1", "P2"}
                actual_keys = set()
                prefix = ""
                for key in kernel_parameters.keys():
                    actual_keys.add(key.split("__")[-1])
                    if key.split("__")[-1] == "SA" and len(key.split("__")[-1]) > 1:
                        prefix = key.split("__")[0]
                        prefix = prefix + "__"

                if not expected_keys.issubset(actual_keys):
                    raise ValueError(
                        f"Expected multitaskSVM parameters {expected_keys} to "
                        f"be contained in {actual_keys}."
                    )

                params = {
                    "SA": kernel_parameters[f"{prefix}SA"],
                    "SO": kernel_parameters[f"{prefix}SO"],
                    "R0": kernel_parameters[f"{prefix}R0"],
                    "R1": kernel_parameters[f"{prefix}R1"],
                    "R2": kernel_parameters[f"{prefix}R2"],
                    "P1": kernel_parameters[f"{prefix}P1"],
                    "P2": kernel_parameters[f"{prefix}P2"],
                }

                if (params["SA"] == "X" or params["SO"] == "X") and params["R0"] != "X":
                    if params["SA"] == params["SO"]:
                        kernel_time_series = "rbf_kernel"
                    else:
                        raise ValueError("Time series kernel is neither rbf nor sigmoid.")
                elif params["SA"] != "X" and params["SO"] != "X" and params["R0"] == "X":
                    kernel_time_series = "sigmoid_kernel"
                else:
                    raise ValueError("Time series kernel is neither rbf nor sigmoid.")

                if params["R1"] != "X" and params["P1"] == "X":
                    kernel_dosage = "rbf_kernel"
                elif params["R1"] == "X" and params["P1"] != "X":
                    kernel_dosage = "poly_kernel"
                else:
                    raise ValueError("Dosage kernel is neither rbf nor polynomial.")

                if params["R2"] != "X" and params["P2"] == "X":
                    kernel_abSignals = "rbf_kernel"
                elif params["R2"] == "X" and params["P2"] != "X":
                    kernel_abSignals = "poly_kernel"
                else:
                    raise ValueError("Antibody signal kernel is neither rbf nor polynomial.")

                # for lower quantile combination:
                # add test combination as new sample to data
                data.loc["eval_feature", :] = combinations["lower_combinations"][m]
                if data.index.get_loc("eval_feature") != data.shape[0] - 1:
                    raise ValueError("eval_feature is not at the end of the DataFrame.")
                # TODO: Implement function to only calculate the single actually needed row of
                # the kernel matrix instead of calculating the full matrix every single time.
                gram_matrix = make_kernel_matrix(
                    AB_signals=data.drop(columns=["TimePointOrder", "Dose"]),
                    time_series=data["TimePointOrder"],
                    dose=data["Dose"],
                    model=params,
                    kernel_time_series=kernel_time_series,
                    kernel_dosage=kernel_dosage,
                    kernel_abSignals=kernel_abSignals,
                )
                single_feature_sample = gram_matrix[0][-1, : len(gram_matrix[0]) - 1]
                # TODO: Consider using SVC(decision_function_shape='ovo') and dividing distance by
                # |coeff_| to get the exact distances. Currently distance is just proportional to
                # the actual (exact) distance. This is unproblematic, since the proportions
                # between the ESPY values aren't impaired (inverted) by this, but in the future
                # exact distances might be preferable.
                distance = model.decision_function(single_feature_sample.reshape(1, -1))
                # shape of distance should always be (n_samples, n_classes * (n_classes-1) / 2) =
                # (n_samples,) in the binary case, even if decision_function_shape=’ovr’,
                # i.e., the documentation is wrong (at least for v1.0.2) on
                # https://scikit-learn.org/1.0/modules/generated/sklearn.svm.SVC.html
                if not distance.shape == (1,):
                    raise ValueError(f"Shape of distance is {distance.shape}, but should be (1,)")
                get_distance_lower.append(distance[0])

                # for upper quantile combination:
                # add test combination as new sample to data
                data.loc["eval_feature", :] = combinations["upper_combinations"][m]
                if data.index.get_loc("eval_feature") != data.shape[0] - 1:
                    raise ValueError("eval_feature is not at the end of the DataFrame.")
                # TODO: Implement function to only calculate the single actually needed row of
                # the kernel matrix instead of calculating the full matrix every single time.
                gram_matrix = make_kernel_matrix(
                    AB_signals=data.drop(columns=["TimePointOrder", "Dose"]),
                    time_series=data["TimePointOrder"],
                    dose=data["Dose"],
                    model=params,
                    kernel_time_series=kernel_time_series,
                    kernel_dosage=kernel_dosage,
                    kernel_abSignals=kernel_abSignals,
                )
                single_feature_sample = gram_matrix[0][-1, : len(gram_matrix[0]) - 1]
                # TODO: Consider using SVC(decision_function_shape='ovo') and dividing distance by
                # |coeff_| to get the exact distances. Currently distance is just proportional to
                # the actual (exact) distance. This is unproblematic, since the proportions
                # between the ESPY values aren't impaired (inverted) by this, but in the future
                # exact distances might be preferable.
                distance = model.decision_function(single_feature_sample.reshape(1, -1))
                # shape of distance should always be (n_samples, n_classes * (n_classes-1) / 2) =
                # (n_samples,) in the binary case, even if decision_function_shape=’ovr’,
                # i.e., the documentation is wrong (at least for v1.0.2) on
                # https://scikit-learn.org/1.0/modules/generated/sklearn.svm.SVC.html
                if not distance.shape == (1,):
                    raise ValueError(f"Shape of distance is {distance.shape}, but should be (1,)")
                get_distance_upper.append(distance[0])

                # for consensus sample:
                # add median as new sample to data
                data.loc["eval_feature", :] = median
                if data.index.get_loc("eval_feature") != data.shape[0] - 1:
                    raise ValueError("eval_feature is not at the end of the DataFrame.")
                # TODO: Implement function to only calculate the single actually needed row of
                # the kernel matrix instead of calculating the full matrix every single time.
                gram_matrix = make_kernel_matrix(
                    AB_signals=data.drop(columns=["TimePointOrder", "Dose"]),
                    time_series=data["TimePointOrder"],
                    dose=data["Dose"],
                    model=params,
                    kernel_time_series=kernel_time_series,
                    kernel_dosage=kernel_dosage,
                    kernel_abSignals=kernel_abSignals,
                )
                feature_consensus_sample = gram_matrix[0][-1, : len(gram_matrix[0]) - 1]
                # TODO: Consider using SVC(decision_function_shape='ovo') and dividing distance by
                # |coeff_| to get the exact distances. Currently distance is just proportional to
                # the actual (exact) distance. This is unproblematic, since the proportions
                # between the ESPY values aren't impaired (inverted) by this, but in the future
                # exact distances might be preferable.
                d_cons = model.decision_function(feature_consensus_sample.reshape(1, -1))
                # shape of distance should always be (n_samples, n_classes * (n_classes-1) / 2) =
                # (n_samples,) in the binary case, even if decision_function_shape=’ovr’,
                # i.e., the documentation is wrong (at least for v1.0.2) on
                # https://scikit-learn.org/1.0/modules/generated/sklearn.svm.SVC.html
                if not d_cons.shape == (1,):
                    raise ValueError(f"Shape of distance is {d_cons.shape}, but should be (1,)")

            else:
                raise ValueError("You must supply `data` and `kernel_parameters`.")

    # get data frame of distances values for median, lower and upper quantile
    get_distance_df = pd.DataFrame(
        [get_distance_upper, get_distance_lower],
        columns=labels,
        index=["UpperQuantile [d]", "LowerQuantile [d]"],
    )

    # add distance of consensus feature
    get_distance_df.loc["consensus [d]"] = np.repeat(d_cons, len(get_distance_df.columns))
    print("Number of evaluated features:")
    print(len(get_distance_df.columns))

    # calculate absolute distance value |d| based on lower and upper quantile
    d_up: float
    d_down: float
    for col in get_distance_df.columns:
        # get evaluated distance based on upper quantile minus consensus
        d_up = (
            get_distance_df.at["UpperQuantile [d]", col] - get_distance_df.at["consensus [d]", col]
        )
        get_distance_df.at["UQ - consensus [d]", col] = d_up

        # get evaluated distance based on lower quantile minus consensus
        d_down = (
            get_distance_df.at["LowerQuantile [d]", col] - get_distance_df.at["consensus [d]", col]
        )
        get_distance_df.at["LQ - consensus [d]", col] = d_down

        # calculate importance from distance based on lower quantile
        # and distance_based on upper quantile
        if d_up >= 0.0 and d_down <= 0.0 and d_up != d_down:
            effect = "+"
            d_value = abs(d_up) + abs(d_down)
        elif d_up <= 0.0 and d_down >= 0.0 and d_up != d_down:
            effect = "-"
            d_value = abs(d_up) + abs(d_down)
        elif d_up == 0.0 and d_down == 0.0:
            effect = "0"
            d_value = 0.0
        else:
            effect = "NAN"
            d_value = np.nan
        get_distance_df.loc["|d|", col] = d_value
        get_distance_df.loc["effect", col] = effect

    # sort values by abs-value of |d|
    get_distance_df = get_distance_df.T.sort_values(
        by="|d|", axis=0, kind="stable", ascending=False, na_position="last"
    ).T

    # Normalize the distance values
    bool_idx = get_distance_df.loc["|d|"].notna()
    sum_of_distance = get_distance_df.loc["|d|", :].sum(skipna=True)
    for col in get_distance_df.columns[bool_idx]:
        get_distance_df.at["|d_norm|", col] = get_distance_df.at["|d|", col] / sum_of_distance
    if not np.allclose(get_distance_df.loc["|d_norm|", :].sum(skipna=True), 1.0):
        raise ValueError(
            "np.allclose(get_distance_df.loc['|d_norm|', :].sum(skipna=True), 1.0) is False."
        )

    print("Dimension of distance matrix:")
    print(get_distance_df.shape)
    print("")

    return get_distance_df


def make_plot(
    data: pd.DataFrame,
    name: str,
    outputdir: str,
) -> None:
    """

    Parameter
    ---------
    data : pd.DataFrame
        Dataframe of distances.
    name : str
        Output filename.
    outputdir : str
        Directory where the plots are stored as .png and .pdf.
    """
    plt.figure(figsize=(20, 10))
    labels = data.columns

    ax = plt.subplot(111)
    w = 0.3
    opacity = 0.6

    values = np.array(data.T.effect)

    def dummy(x):
        if x == "+":
            return "red"
        elif x == "-":
            return "blue"
        else:
            return "white"

    clrs = [dummy(x) for x in values]

    index = np.arange(len(labels))
    ax.bar(
        index,
        np.absolute(data.loc["|d_norm|"].to_numpy()),
        width=w,
        color=clrs,
        align="center",
        alpha=opacity,
    )
    ax.xaxis_date()

    plt.xlabel("number of features", fontsize=20)
    plt.ylabel("ESPY value", fontsize=20)
    plt.xticks(index, labels, fontsize=10, rotation=90)

    plt.savefig(os.path.join(outputdir, name + ".png"), dpi=600)
    plt.savefig(os.path.join(outputdir, name + ".pdf"), format="pdf", bbox_inches="tight")
    plt.close()


def featureEvaluationESPY(
    *,
    eval_data: pd.DataFrame,
    model: SVC,
    lq: int,
    up: int,
    basis_data: Optional[pd.DataFrame] = None,
    kernel_parameters: Optional[Dict[str, Union[float, int, str]]] = None,
    multitask: bool = False,
) -> pd.DataFrame:
    """ESPY measurement.

    Calculate ESPY value for each feature on proteome or simulated data.

    Parameter
    -----------
    eval_data : pd.Dataframe
        Input data (typically from a single timepoint) used to calculate
        the upper and lower percentile and median utilized by ESPY.
    model : sklearn.svm.SVC
        SVC model.
    lq : int
        Lower percentile value.
    up : int
        Upper percentile value.
    basis_data : pd.DataFrame, default=None
        Dataset from which to compute the multitask Gram matrix. Must be the same dataset
        that was used to train the multitaskSVM model. Typically, this is the full dataset
        (n x n_features) combined from all timepoints (n = n_times x n_samples).
    kernel_parameters : pd.DataFrame, default=None
        Kernel parameters for real data.
    multitask: bool, default=False
        If True, it will be assumed that the given sklearn.svm.SVC model is
        based on a precomputed multitask Gram matrix, i.e., SVC(kernel='precomputed').
    Returns
    --------
    ESPY_importances : pd.Dataframe
        Dataframe of ESPY values |d| and effects for each feature in the data.

    """

    start = time.time()

    statistics, median, combinations = make_feature_combination(
        X=eval_data, lowerValue=lq, upperValue=up
    )

    print("Statistics of features:")
    print(statistics)
    print("")

    if not multitask:
        ESPY_importances = compute_distance_hyper(
            median=median,
            combinations=combinations,
            model=model,
            labels=statistics.columns.to_list(),
        )

    else:
        if isinstance(basis_data, pd.DataFrame) and isinstance(kernel_parameters, dict):
            ESPY_importances = compute_distance_hyper(
                median=median,
                combinations=combinations,
                model=model,
                labels=statistics.columns.to_list(),
                data=basis_data,
                kernel_parameters=kernel_parameters,
                multitask=True,
            )

        else:
            raise ValueError("`basis_data` and `kernel_parameters` must be supplied.")

    end = time.time()
    print(f"end of computation after: {end - start} seconds.\n")

    return ESPY_importances


def featureEvaluationRF(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    score: str = "precision_recall_auc",
) -> Tuple[pd.DataFrame, Pipeline]:
    """Evaluation of informative features of a given RF model.
    Feature importances are obtained via permutation importance evaluation.
    The model will be fitted on X, y before feature evaluation.

    Parameter
    ---------
    model : sklearn.pipeline.Pipeline
        Last step in pipeline must be RandomForestClassifier estimator.
    X : pd.DataFrame
        Feature matrix given as pd.DataFrame.
    y : np.ndarray
        Label vector.
    score : str, default='precision_recall_auc'
        Scorer to use. Can be either 'roc_auc' or 'precision_recall_auc'.

    Returns
    -------
    importances : pd.Dataframe
        Dataframe of non-zero feature importances.
    model : sklearn.pipeline.Pipeline
        Fitted model (using X, y).
    """
    if not isinstance(model[-1], RandomForestClassifier):
        raise ValueError("Last step in pipeline must be RandomForestClassifier estimator.")

    if score == "roc_auc":
        scoring = "roc_auc"
    elif score == "precision_recall_auc":
        scoring = make_scorer(precision_recall_auc, greater_is_better=True, needs_proba=True)
    else:
        raise ValueError("score must be one of 'roc_auc', 'precision_recall_auc'")

    model.fit(X.to_numpy(), y)
    result = permutation_importance(
        model,
        X.to_numpy(),
        y,
        scoring=scoring,
        n_repeats=100,
        n_jobs=-1,
        random_state=seed,
    )

    sorted_importances_idx = result.importances_mean.argsort()
    importances = pd.DataFrame(
        result.importances_mean[sorted_importances_idx],
        index=X.columns[sorted_importances_idx],
        columns=["importance"],
    )

    # Extract non-zero importances
    print("Number of non-zero importances:", np.count_nonzero(result.importances_mean))

    return importances[importances["importance"] != 0], model


def featureEvaluationRLR(
    model: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[pd.DataFrame, Pipeline]:
    """Evaluation of informative features of a given RLR model.
    The model will be fitted on X, y before feature evaluation.

    Parameter
    ---------
    model : sklearn.pipeline.Pipeline
        Last step in pipeline must be LogisticRegression estimator.
    X : pd.DataFrame
        Feature matrix given as pd.DataFrame.
    y : np.ndarray
        Label vector.

    Returns
    -------
    importances : pd.Dataframe
        Dataframe of non-zero feature importances.
    model : sklearn.pipeline.Pipeline
        Fitted model (using X, y).
    """

    model.fit(X.to_numpy(), y)
    LR = model[-1]
    if isinstance(LR, LogisticRegression):
        coefs = LR.coef_
    else:
        raise ValueError("Last step in pipeline must be LogisticRegression estimator.")
    if coefs.shape[0] == 1:
        coefs = coefs.flatten()
    else:
        raise ValueError(
            "RLR model appears to be a multiclass model, thus, "
            "coef_ array can't be flattened securely."
        )

    sorted_importances_idx = coefs.argsort()
    importances = pd.DataFrame(
        coefs[sorted_importances_idx],
        index=X.columns[sorted_importances_idx],
        columns=["importance"],
    )

    # Extract non-zero coefficients
    print("Number of non-zero importances (weights):", np.count_nonzero(coefs))

    return importances[importances["importance"] != 0], model
