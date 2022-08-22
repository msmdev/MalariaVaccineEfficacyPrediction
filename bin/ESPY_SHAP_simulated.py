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

from datetime import datetime
import numpy as np
import pandas as pd
import os
import pathlib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import shap
import warnings
import argparse
from source.featureEvaluationESPY import ESPY_measurement, make_plot
from source.utils import svm_model


def main(
    *,
    data_dir: str,
    data_file: str,
    out_dir: str,
    uq: int,
    lq: int,
):
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(os.path.join(data_dir, data_file))

    X_train, X_test, y_train, y_test = train_test_split(
        data.iloc[:, :1000].to_numpy(),
        data.iloc[:, 1000].to_numpy(),
        test_size=0.3,
        random_state=123,
    )
    rbf_SVM_model = svm_model(
        X_train_data=X_train,
        y_train_data=y_train,
        X_test_data=X_test,
        y_test_data=y_test,
    )

    print("ESPY value measurement started on simulated data with the following parameters:")
    print(f"value of upper percentile: {uq}")
    print(f"value of lower percentile: {lq}\n")

    output_filename = "ESPY_values_on_simulated_data"

    distance_result = ESPY_measurement(
        identifier='simulated',
        data=pd.DataFrame(X_test),
        model=rbf_SVM_model,
        lq=lq,
        up=uq,
    )
    print(distance_result)

    make_plot(
        data=distance_result.iloc[:, :25],
        name=output_filename,
        outputdir=out_dir,
    )

    distance_result.to_csv(
        os.path.join(out_dir, f"{output_filename}.tsv"),
        sep='\t',
        na_rep='nan',
    )
    print("ESPY results were saved in: ", os.path.join(out_dir, f"{output_filename}.tsv\n"))

    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
    print(
        f"Evaluation of informative features based on SHAP values has started at {timestamp}."
    )

    explainer = shap.KernelExplainer(rbf_SVM_model.predict_proba, X_train)
    # Shap_values returns a list of two arrays:
    # the first for the negative class probabilities and
    # the second for the positive class probabilties.
    shap_values = explainer.shap_values(X_test, l1_reg='num_features(25)')

    # Plot shap values for the positive class probabilities:
    shap.initjs()
    shap.summary_plot(shap_values[1], X_test, max_display=25, plot_type='dot', show=False)
    plt.savefig(os.path.join(out_dir, f"SHAP_values_simulated_data_dot_{timestamp}.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close()
    shap.summary_plot(shap_values[1], X_test, max_display=25, plot_type='bar', show=False)
    plt.savefig(os.path.join(out_dir, f"SHAP_values_simulated_data_bar_{timestamp}.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close()
    shap.summary_plot(shap_values[1], X_test, max_display=25, plot_type='violin', show=False)
    plt.savefig(os.path.join(out_dir, f"SHAP_values_simulated_data_violin_{timestamp}.pdf"),
                format="pdf", bbox_inches="tight")
    plt.close()

    if isinstance(shap_values, np.ndarray):
        np.save(os.path.join(out_dir, f"SHAP_values_simulated_data_{timestamp}.npy"),
                shap_values)
    elif isinstance(shap_values, list):
        np.savez(os.path.join(out_dir, f"SHAP_values_simulated_data_{timestamp}.npz"),
                 *shap_values)
    else:
        warnings.warn("Couldn't save shap values, since they were of "
                      f"unexpected type {type(shap_values)}.")

    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
    print(
        f"SHAP evaluation has terminated at {timestamp} and results are saved in "
        "../results/SVM/simulated/SHAP as SHAP_value_simulated_data.png."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-dir',
        dest='data_dir',
        metavar='DIR',
        required=True,
        help=('Path to the directory were the simulated data or '
              'preprocessed proteome data is located.'),
    )
    parser.add_argument(
        '--data-file',
        dest='data_file',
        metavar='FILE',
        required=True,
        help=("Full name of the data file (located in the directory given via --data-dir).")
    )
    parser.add_argument(
        '--out-dir',
        dest='out_dir',
        metavar='DIR',
        required=True,
        help='Path to the directory were the results shall be saved.',
    )
    parser.add_argument(
        '--lower-percentile',
        dest='lq',
        type=int,
        default=25,
        required=True,
        help='Lower percentile given as int, by default 25%.',
    )
    parser.add_argument(
        '--upper-percentile',
        dest='uq',
        type=int,
        default=75,
        required=True,
        help='Upper percentile given as int, by default 75%.',
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        data_file=args.data_file,
        out_dir=args.out_dir,
        uq=args.uq,
        lq=args.lq,
    )
