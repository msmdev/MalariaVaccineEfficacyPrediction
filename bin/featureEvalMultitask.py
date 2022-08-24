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
Parser for the feature evaluation approach

This module computes the ESPY value of each single feature.
The ESPY value is the distances of each single features to the classification boundary in the
multitask-SVM model and compares the change of the distance with a consensus sample.

@Author: Bernhard Reuter and Jacqueline Wistuba-Hamprecht
"""

import argparse
import os
import pandas as pd
import numpy as np
from source.featureEvaluation import featureEvaluationESPY, make_plot
from source.utils import DataSelector, get_parameters
from source.utils import select_timepoint
from source.multitaskSVM_config import estimator


def main(
    *,
    data_dir: str,
    out_dir: str,
    uq: int,
    lq: int,
    data_file_id: str,
    rgscv_path: str,
    kernel_dir: str,
    kernel_identifier: str,
    timepoint: str,
) -> None:
    """
    Call ESPY measurement.
    """
    print("ESPY value measurement started with the following parameters:")
    print("value of upper percentile      = ", str(uq))
    print("value of lower percentile      = ", str(lq))
    print("at time point                = ", str(timepoint))
    print("\n")

    for i in [rgscv_path, kernel_dir, kernel_identifier, timepoint]:
        if not isinstance(i, str):
            raise ValueError(
                "`rgscv_path`, `kernel_dir`, `kernel_identifier`, "
                "and `timepoint` must be of type str."
            )
    if timepoint not in ['III14', 'C-1', 'C28']:
        raise ValueError("timepoint must be one of 'III14', 'C-1' or 'C28'.")

    data = pd.read_csv(
        os.path.join(data_dir, f'{data_file_id}_all.csv'),
        header=0,
    )
    y = data.loc[:, 'Protection'].to_numpy()

    rgscv_results = pd.read_csv(rgscv_path, delimiter="\t", header=0, index_col=0)

    output_filename = f"ESPY_values_{timepoint}"

    timepoint_results = select_timepoint(rgscv_results, timepoint)
    params = get_parameters(timepoint_results, "multitask")

    # initialize running index array for DataSelector
    assert y.size * y.size < np.iinfo(np.uint32).max, \
        f"y is to large: y.size * y.size >= {np.iinfo(np.uint32).max}"
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

    multitask_classifier = estimator['svc']
    multitask_classifier.set_params(**{'C': params['C']})
    multitask_classifier.fit(kernel_matrix, y)

    data_at_timePoint = pd.read_csv(
        os.path.join(data_dir, f'{data_file_id}_{timepoint}.csv'),
        header=0,
    )
    if not np.all(
        np.isin(data_at_timePoint.drop(
            columns=['Patient', 'group', 'Protection', 'TimePointOrder', 'Dose']
        ).dtypes.to_list(), ['float64'])
    ):
        raise ValueError(
            f"Not all antibody intensities read from {data_file_id}_{timepoint}.csv "
            "are of type float64."
        )

    distance_result = featureEvaluationESPY(
        eval_data=data_at_timePoint.drop(
            columns=['Patient', 'group', 'Protection']
        ),  # including dose AND timepoint
        model=multitask_classifier,
        lq=lq,
        up=uq,
        basis_data=data.drop(
            columns=['Patient', 'group', 'Protection']
        ),  # including dose AND timepoint,
        kernel_parameters=params,
        multitask=True,
    )

    print("Distances for features:")
    print(distance_result)
    print('')

    make_plot(
        data=distance_result.iloc[:, :50],
        name=output_filename,
        outputdir=out_dir,
    )

    distance_result.to_csv(
        os.path.join(out_dir, f"{output_filename}.tsv"),
        sep='\t',
        na_rep='nan'
    )
    print('Results were saved in:', os.path.join(out_dir, f"{output_filename}.tsv"))
    print('')


if __name__ == "__main__":
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)

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
        '--kernel-dir',
        dest='kernel_dir',
        metavar='DIR',
        required=True,
        help='Path to the directory were the precomputed kernel matrix is stored.',
    )
    parser.add_argument(
        '--kernel-identifier',
        dest='kernel_identifier',
        required=True,
        help='Filename prefix of the precomputed kernel matrix.',
    )
    parser.add_argument(
        '--rgscv-path',
        dest='rgscv_path',
        metavar='FILEPATH',
        required=True,
        help='Path to the File were the RGSCV results are stored.',
    )
    parser.add_argument(
        '--timepoint',
        dest='timepoint',
        choices=['III14', 'C-1', 'C28'],
        required=True,
        help='Time point for which the analysis shall be performed.',
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        uq=args.uq,
        lq=args.lq,
        data_file_id=args.data_file_id,
        rgscv_path=args.rgscv_path,
        kernel_dir=args.kernel_dir,
        kernel_identifier=args.kernel_identifier,
        timepoint=args.timepoint,
    )
