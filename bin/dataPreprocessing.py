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
This script performs the preprocessing of the raw proteome array data.

@Author: Bernhard Reuter and Jacqueline Wistuba-Hamprecht

"""

import argparse
import os

import pandas as pd
from source.preprocessing import preprocessing


def main(
    data_dir: str,
    data_file: str,
    out_name: str,
):

    fn = os.path.join(data_dir, data_file)
    data = pd.read_csv(fn)

    preprocessed_data = preprocessing(data)

    preprocessed_data.to_csv(
        os.path.join(data_dir, f"{out_name}.csv"),
        index=False,
    )

    print("The proteome data was successfully preprocessed.\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=("Script to preprocess the raw proteome data."))
    parser.add_argument(
        "--data-dir",
        dest="data_dir",
        metavar="DIR",
        required=True,
        help=(
            "Path to the directory were the proteome data is located. "
            "The preprocessed data will be also stored here."
        ),
    )
    parser.add_argument(
        "--data-file",
        dest="data_file",
        metavar="FILE",
        required=True,
        help=("Name of the file were the raw proteome data is stored."),
    )
    parser.add_argument(
        "--out-name",
        dest="out_name",
        required=True,
        help=(
            "Identifier to use when naming the file were the preprocessed proteome data "
            "is stored (WITHOUT a file ending like '.csv'!)."
        ),
    )
    args = parser.parse_args()

    main(
        args.data_dir,
        args.data_file,
        args.out_name,
    )
