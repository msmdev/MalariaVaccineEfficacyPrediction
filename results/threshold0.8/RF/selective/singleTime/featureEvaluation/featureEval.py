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
Evaluation of informative features.

Will save the results.

@Author: Jacqueline Wistuba-Hamprecht and Bernhard Reuter

"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
from nestedcv import save_model

from source.featureEvaluation import (
    featureEvaluationESPY,
    featureEvaluationRF,
    featureEvaluationRLR,
    make_plot,
)
from source.utils import get_parameters, select_timepoint


def main(
    *,
    data_dir: str,
    data_file_id: str,
    rgscv_path: str,
    out_dir: str,
    timepoint: str,
    method: str,
    kernel_dir: Optional[str] = None,
    kernel_identifier: Optional[str] = None,
    combination: Optional[str] = None,
):
    """
    Evaluation of informative features.
    """

    data = pd.read_csv(
        os.path.join(data_dir, f"{data_file_id}.csv"),
        header=0,
    )

    if timepoint == "III14":
        # after immunization III 14
        data_at_timePoint = data.loc[data["TimePointOrder"] == 2, :].copy()
    elif timepoint == "C-1":
        # before re-infection C-1
        data_at_timePoint = data.loc[data["TimePointOrder"] == 3, :].copy()
    else:
        raise ValueError("timepoint must be one of 'III14' or 'C-1'")

    X = data_at_timePoint.drop(
        columns=["Patient", "group", "Protection", "TimePointOrder"]
    )  # including dose
    y = data_at_timePoint.loc[:, "Protection"].to_numpy()
    rgscv_results = pd.read_csv(rgscv_path, sep="\t", header=0, index_col=0)

    timepoint_results = select_timepoint(rgscv_results=rgscv_results, timepoint=timepoint)

    # Initialize, fit, and evaluate
    if method == "RF":
        from source.RF_config import estimator

        params = get_parameters(
            timepoint_results=timepoint_results,
            model="RF",
        )
        estimator.set_params(**params)

        importances, model = featureEvaluationRF(estimator, X, y)

    elif method == "RLR":
        from source.RLR_config import estimator

        params = get_parameters(
            timepoint_results=timepoint_results,
            model="RLR",
        )
        estimator.set_params(**params)

        importances, model = featureEvaluationRLR(estimator, X, y)

    elif method == "multitaskSVM":
        from source.multitaskSVM_config import configurator

        if not (
            isinstance(kernel_dir, str)
            and isinstance(kernel_identifier, str)
            and isinstance(combination, str)
        ):
            raise ValueError(
                "`kernel_dir`, `kernel_identifier` and `combination` must be given "
                "if `method`='multitaskSVM'."
            )

        _, model = configurator(
            combination=combination,
            identifier=kernel_identifier,
            kernel_dir=kernel_dir,
        )

        params = get_parameters(
            timepoint_results=timepoint_results,
            model="multitaskSVM",
        )

        # initialize running index array for DataSelector
        data = pd.read_csv(
            os.path.join(data_dir, f"{data_file_id}.csv"),
            header=0,
        )
        y = data.loc[:, "Protection"].to_numpy()
        if not y.size * y.size < np.iinfo(np.uint32).max:
            raise ValueError(f"y is to large: y.size * y.size >= {np.iinfo(np.uint32).max}")
        X = np.array([x for x in range(y.size * y.size)], dtype=np.uint32).reshape((y.size, y.size))

        model.set_params(**params)
        model.fit(X, y)

        importances = featureEvaluationESPY(
            eval_data=data_at_timePoint.drop(
                columns=["Patient", "group", "Protection"]
            ),  # including dose AND timepoint
            model=model["svc"],
            lq=25,
            up=75,
            basis_data=data.drop(
                columns=["Patient", "group", "Protection"]
            ),  # including dose AND timepoint,
            kernel_parameters=params,
            multitask=True,
        )

        make_plot(
            data=importances.iloc[:, :50],
            name=f"top50_informative_features_{timepoint}",
            outputdir=out_dir,
        )
    else:
        raise ValueError(f"Illegal input: unknown method {method}.")

    fn = f"best_{method}_model_{timepoint}_"
    for key in params.keys():
        fn = fn + f"_{key.split('__')[-1]}_{params[key]}"
    save_model(
        model,
        out_dir,
        fn,
        timestamp=False,
        compress=False,
        method="joblib",
    )

    print(f"Parameter combination with best mean performance at time point {timepoint} :")
    print(params)
    print("")

    fn = os.path.join(out_dir, f"informative_features_{timepoint}.tsv")
    pd.DataFrame(data=importances).to_csv(fn, sep="\t", na_rep="nan")

    print(f"Results are saved in: {fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Function to run an analysis of informative features from RF.")
    )
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        metavar="DIR",
        required=True,
        help="Path to the directory were the data is located.",
    )
    parser.add_argument(
        "--data-file-id",
        dest="data_file_id",
        required=True,
        help=(
            "String identifying the data file (located in the directory given via --data-dir)."
            "This string will be appended by '.csv'. If you pass, for example, "
            "'--data-file-id preprocessed_whole_data', the resulting file "
            "name will be 'preprocessed_whole_data.csv'."
        ),
    )
    parser.add_argument(
        "--rgscv-path",
        dest="rgscv_path",
        metavar="FILE",
        required=True,
        help="Path to the File were the RGSCV results are stored.",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        metavar="DIR",
        required=True,
        help="Path to the directory to wihich the output shall be written.",
    )
    parser.add_argument(
        "--timepoint",
        dest="timepoint",
        type=str,
        required=True,
        help=(
            "Time point for which the analysis shall be performed. "
            "Must be one of 'III14' or 'C-1'."
        ),
    )
    parser.add_argument(
        "--method",
        dest="method",
        required=True,
        help="Choose the method (either 'RF', 'RLR' or 'multitaskSVM') used to model the data.",
    )
    parser.add_argument(
        "--kernel-dir",
        dest="kernel_dir",
        metavar="DIR",
        help="Path to the directory were the precomputed kernel matrix is stored.",
    )
    parser.add_argument(
        "--kernel-identifier",
        dest="kernel_identifier",
        help="Filename prefix of the precomputed kernel matrix.",
    )
    parser.add_argument(
        "--combination",
        dest="combination",
        help=(
            "Kernel combination. Supply 'RPP', 'RPR', 'RRP', 'RRR', 'SPP', 'SPR', 'SRP', or 'SRR'."
        ),
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        data_file_id=args.data_file_id,
        rgscv_path=args.rgscv_path,
        out_dir=args.out_dir,
        timepoint=args.timepoint,
        method=args.method,
        kernel_dir=args.kernel_dir,
        kernel_identifier=args.kernel_identifier,
        combination=args.combination,
    )
