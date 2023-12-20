# Copyright (c) 2022 Jacqueline Wistuba-Hamprecht and Bernhard Reuter.
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
This script performs the evaluation of informative features from simulated data
using both the ESPY and the SHAP (SHapley Additive exPlanations) framework.

@Author: Jacqueline Wistuba-Hamprecht and Bernhard Reuter

"""

import argparse
import os
import pathlib
import warnings
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC

from source.config import seed
from source.featureEvaluation import featureEvaluationESPY, make_plot


def optimize_svm_model(
    *,
    X_train_data: np.ndarray,
    y_train_data: np.ndarray,
    X_test_data: np.ndarray,
    y_test_data: np.ndarray,
) -> SVC:
    """Initialize SVM model on simulated data.

    Initialize SVM model with a RBF kernel on simulated data and
    perform a grid-search for kernel parameter evaluation.
    Returns the SVM model with the best parameters based on the highest mean AUC score.
    CAUTION: All calls to functions with randomization are seeded to ensure reproducible behaviour.

    Parameters
    ----------
    X_train_data : np.ndarray
        Training data.
    y_train_data : np.ndarray
        y labels for training.
    X_test_data : np.ndarray
        Test data.
    y_test_data : np.ndarray
        y labels for testing.

    Returns
    -------
    model : sklearn.svm.SVC object
        Trained SVM model with best kernel parameters found via GridSearchCV.
    """

    from source.SVM_config import estimator, param_grid

    for key in param_grid.keys():
        new_key = key.split("__")[-1]
        param_grid[new_key] = param_grid.pop(key)

    svm = estimator["svc"]

    # grid-search on simulated data
    clf = GridSearchCV(
        svm,
        param_grid,
        scoring="roc_auc",
        refit=True,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
    )
    clf.fit(X_train_data, y_train_data)

    print(f"The best parameters are {clf.best_params_} with a mean AUC score of {clf.best_score_}.")

    # run RBF SVM with best parameters from grid-search,
    # probability has to be TRUE to evaluate features via SHAP
    params = dict()
    for key in param_grid.keys():
        params[key] = clf.best_params_.get(key)

    svm.set_params(**params)

    svm.fit(X_train_data, y_train_data)

    y_pred = svm.predict(X_test_data)

    AUC = roc_auc_score(y_test_data, y_pred)

    print(f"AUC score on unseen data: {AUC}")

    return svm


def main(
    *,
    data_dir: str,
    data_file: str,
    out_dir: str,
):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(os.path.join(data_dir, data_file))

    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:, :1000].to_numpy(),
        data.iloc[:, 1000].to_numpy(),
        test_size=0.3,
        random_state=seed,
    )
    model = optimize_svm_model(
        X_train_data=np.array(X_train),
        y_train_data=np.array(y_train),
        X_test_data=np.array(X_test),
        y_test_data=np.array(y_test),
    )

    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
    print(f"ESPY value measurement started at {timestamp}.")

    output_filename = "ESPY_values_on_simulated_data"

    distance_result = featureEvaluationESPY(
        eval_data=pd.DataFrame(X_test),
        model=model,
        lq=25,
        up=75,
    )
    print(distance_result)

    make_plot(
        data=distance_result.iloc[:, :25],
        name=output_filename,
        outputdir=out_dir,
    )

    distance_result.to_csv(
        os.path.join(out_dir, f"{output_filename}.tsv"),
        sep="\t",
        na_rep="nan",
    )
    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
    print(f"ESPY evaluation has terminated at {timestamp} and results are saved in {out_dir}\n")

    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
    print(f"Evaluation of informative features based on SHAP values started at {timestamp}.")

    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    # Shap_values returns a list of two arrays:
    # the first for the negative class probabilities and
    # the second for the positive class probabilties.
    shap_values = explainer.shap_values(X_test, l1_reg="num_features(25)")

    if shap_values is None:
        raise ValueError("Can't proceed since shap_values=None.")

    # Plot shap values for the positive class probabilities:
    shap.initjs()
    shap.summary_plot(shap_values[1], X_test, max_display=25, plot_type="dot", show=False)
    plt.savefig(
        os.path.join(out_dir, f"SHAP_values_simulated_data_dot_{timestamp}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()
    shap.summary_plot(shap_values[1], X_test, max_display=25, plot_type="bar", show=False)
    plt.savefig(
        os.path.join(out_dir, f"SHAP_values_simulated_data_bar_{timestamp}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()
    shap.summary_plot(shap_values[1], X_test, max_display=25, plot_type="violin", show=False)
    plt.savefig(
        os.path.join(out_dir, f"SHAP_values_simulated_data_violin_{timestamp}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    if isinstance(shap_values, np.ndarray):
        np.save(os.path.join(out_dir, f"SHAP_values_simulated_data_{timestamp}.npy"), shap_values)
    elif isinstance(shap_values, list):
        np.savez(os.path.join(out_dir, f"SHAP_values_simulated_data_{timestamp}.npz"), *shap_values)
    else:
        warnings.warn(
            "Couldn't save shap values, since they were of " f"unexpected type {type(shap_values)}."
        )

    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
    print(f"SHAP evaluation has terminated at {timestamp} and results are saved in {out_dir}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        metavar="DIR",
        required=True,
        help=(
            "Path to the directory were the simulated data or "
            "preprocessed proteome data is located."
        ),
    )
    parser.add_argument(
        "--data-file",
        dest="data_file",
        metavar="FILE",
        required=True,
        help=("Full name of the data file (located in the directory given via --data-dir)."),
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        metavar="DIR",
        required=True,
        help="Path to the directory were the results shall be saved.",
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        data_file=args.data_file,
        out_dir=args.out_dir,
    )
