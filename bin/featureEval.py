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

import numpy as np
import pandas as pd
import os
import argparse
from source.featureEvaluation import featureEvaluationRF, featureEvaluationRLR
from source.featureEvaluation import featureEvaluationESPY, make_plot
from source.utils import select_timepoint, get_parameters
from source.utils import DataSelector
from typing import Optional
from nestedcv import save_model


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
):
    """
    Evaluation of informative features.
    """

    if timepoint not in ['III14', 'C-1', 'C28']:
        raise ValueError("timepoint must be one of 'III14', 'C-1' or 'C28'.")

    data_at_timePoint = pd.read_csv(
        os.path.join(data_dir, f'{data_file_id}_{timepoint}.csv'),
        header=0,
    )
    X = data_at_timePoint.drop(
        columns=['Patient', 'group', 'Protection', 'TimePointOrder']
    )  # including dose
    y = data_at_timePoint.loc[:, 'Protection'].to_numpy()
    rgscv_results = pd.read_csv(rgscv_path, sep="\t", header=0, index_col=0)

    timepoint_results = select_timepoint(
        rgscv_results=rgscv_results,
        timepoint=timepoint
    )

    # Initialize, fit, and evaluate
    if method == 'RF':
        from source.RF_config import estimator
        params = get_parameters(
            timepoint_results=timepoint_results,
            model='RF',
        )
        estimator.set_params(**params)
        importances, model = featureEvaluationRF(estimator, X, y)
    elif method == 'RLR':
        from source.RLR_config import estimator
        params = get_parameters(
            timepoint_results=timepoint_results,
            model='RLR',
        )
        estimator.set_params(**params)
        importances, model = featureEvaluationRLR(estimator, X, y)
    elif method == 'multitaskSVM':
        if not (isinstance(kernel_dir, str) and isinstance(kernel_identifier, str)):
            raise ValueError(
                "`kernel_dir` and `kernel_identifier` must be given if `method`='multitaskSVM'."
            )
        from source.multitaskSVM_config import estimator
        params = get_parameters(
            timepoint_results=timepoint_results,
            model='multitask',
        )

        # initialize running index array for DataSelector
        data = pd.read_csv(
            os.path.join(data_dir, f'{data_file_id}_all.csv'),
            header=0,
        )
        y = data.loc[:, 'Protection'].to_numpy()
        if not y.size * y.size < np.iinfo(np.uint32).max:
            raise ValueError(f"y is to large: y.size * y.size >= {np.iinfo(np.uint32).max}")
        X = np.array(
            [x for x in range(y.size * y.size)],
            dtype=np.uint32
        ).reshape((y.size, y.size))

        kernel_matrix = DataSelector(
            kernel_directory=kernel_dir,
            identifier=kernel_identifier,
            SA=params['SA'],
            SO=params['SO'],
            R0=params['R0'],
            R1=params['R1'],
            R2=params['R2'],
            P1=params['P1'],
            P2=params['P2'],
        ).fit(X, y).transform(X)

        model = estimator['svc']
        model.set_params(**{'C': params['C']})
        model.fit(kernel_matrix, y)

        importances = featureEvaluationESPY(
            eval_data=data_at_timePoint.drop(
                columns=['Patient', 'group', 'Protection']
            ),  # including dose AND timepoint
            model=model,
            lq=25,
            up=75,
            basis_data=data.drop(
                columns=['Patient', 'group', 'Protection']
            ),  # including dose AND timepoint,
            kernel_parameters=params,
            multitask=True,
        )

        make_plot(
            data=importances.iloc[:, :50],
            name=f"top50_informative_features_{timepoint}",
            outputdir=out_dir,
        )

    fn = 'best_{method}_model_'
    for key in params.keys():
        fn = fn + f"{key}_{params[key]}"
    save_model(
        model,
        out_dir,
        fn,
        timestamp=True,
        compress=False,
        method='joblib',
    )

    print(f"Parameter combination for best mean AUROC at time point {timepoint} :")
    print(params)
    print('')

    fn = os.path.join(out_dir, f"informative_features_{timepoint}.tsv")
    pd.DataFrame(data=importances).to_csv(fn, sep='\t', na_rep='nan')

    print(f"Results are saved in: {fn}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=('Function to run an analysis of informative features from RF.')
    )
    parser.add_argument(
        '--data-dir',
        dest='data_dir',
        metavar='DIR',
        required=True,
        help='Path to the directory were the data is located.'
    )
    parser.add_argument(
        '--data-file-id',
        dest='data_file_id',
        required=True,
        help=(
            "String identifying the data file (located in the directory given via --data-dir)."
            "This string will be appended by the time point and '.csv'. If you pass, for example, "
            "'--timepoint III14' and '--data-file-id preprocessed_whole_data', the resulting file "
            "name will be 'preprocessed_whole_data_III14.csv'.")
    )
    parser.add_argument(
        '--rgscv-path',
        dest='rgscv_path',
        metavar='FILE',
        required=True,
        help='Path to the File were the RGSCV results are stored.'
    )
    parser.add_argument(
        '--out-dir',
        dest='out_dir',
        metavar='DIR',
        required=True,
        help='Path to the directory to wihich the output shall be written.'
    )
    parser.add_argument(
        '--timepoint',
        dest='timepoint',
        type=str,
        required=True,
        help=(
            "Time point for which the analysis shall be performed. "
            "Must be one of 'III14', 'C-1' or 'C28'.")
    )
    parser.add_argument(
        '--method', dest='method', required=True,
        help="Choose the method (either 'RF', 'RLR' or 'multitaskSVM') used to model the data."
    )
    parser.add_argument(
        '--kernel-dir',
        dest='kernel_dir',
        metavar='DIR',
        help='Path to the directory were the precomputed kernel matrix is stored.',
    )
    parser.add_argument(
        '--kernel-identifier',
        dest='kernel_identifier',
        help='Filename prefix of the precomputed kernel matrix.',
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
    )
