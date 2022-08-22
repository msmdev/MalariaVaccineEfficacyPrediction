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
Evaluation of informative features from RLR models

Will save the results to various .tsv files

@Author: Jacqueline Wistuba-Hamprecht and Bernhard Reuter

"""

import numpy as np
import pandas as pd
import scipy
import sklearn
import sys
import os
import argparse
from source.featureEvaluationRLR import featureEvaluationRLR


def main(
    *,
    data_dir: str,
    data_file_id: str,
    rgscv_path: str,
    out_dir: str,
    timepoint: str,
):
    """
    Evaluation of informative features from RLR.
    """

    if timepoint not in ['III14', 'C-1', 'C28']:
        raise ValueError("timepoint must be one of 'III14', 'C-1' or 'C28'.")

    data_at_timePoint = pd.read_csv(
        os.path.join(data_dir, f'{data_file_id}_{timepoint}.csv'),
        header=0,
    )
    rgscv_results = pd.read_csv(rgscv_path, sep="\t", header=0, index_col=0)

    coefs = featureEvaluationRLR(
        X=data_at_timePoint.drop(
            columns=['Patient', 'group', 'Protection', 'TimePointOrder']
        ).to_numpy(),  # including dose,
        y=data_at_timePoint.loc[:, 'Protection'].to_numpy(),
        feature_labels=data_at_timePoint.iloc[:, 3:].columns.to_list(),
        rgscv_results=rgscv_results,
        timepoint=timepoint,
    )

    fn = os.path.join(out_dir, f"RLR_informative_features_{timepoint}.tsv")
    pd.DataFrame(data=coefs).to_csv(fn, sep='\t', na_rep='nan')

    print(f"Results are saved in: {fn}")


if __name__ == "__main__":
    print('sys.path:', sys.path)
    print('scikit-learn version:', sklearn.__version__)
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)
    print('scipy version:', scipy.__version__)

    parser = argparse.ArgumentParser(
        description=('Function to run an analysis of informative features from RLR.')
    )
    parser.add_argument(
        '--data-dir', dest='data_dir', metavar='DIR', required=True,
        help='Path to the directory were the data is located.'
    )
    parser.add_argument(
        '--data-file-id', dest='data_file_id', required=True,
        help=(
            "String identifying the data file (located in the directory given via --data-dir)."
            "This string will be appended by the time point and '.csv'. If you pass, for example, "
            "'--timepoint III14' and '--data-file-id preprocessed_whole_data', the resulting file "
            "name will be 'preprocessed_whole_data_III14.csv'.")
    )
    parser.add_argument(
        '--rgscv-path', dest='rgscv_path', metavar='FILE', required=True,
        help='Path to the File were the RGSCV results are stored.'
    )
    parser.add_argument(
        '--out-dir', dest='out_dir', metavar='DIR', required=True,
        help='Path to the directory to wihich the output shall be written.'
    )
    parser.add_argument(
        '--timepoint', dest='timepoint', required=True, type=str,
        help=(
            "Time point for which the analysis shall be performed. "
            "Must be one of 'III14', 'C-1' or 'C28'.")
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        data_file_id=args.data_file_id,
        rgscv_path=args.rgscv_path,
        out_dir=args.out_dir,
        timepoint=args.timepoint,
    )
