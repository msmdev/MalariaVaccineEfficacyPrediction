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
Function to group strongly covarying features together.

Will save the results to .csv files

@Author: Bernhard Reuter

"""

import argparse
import os
import pathlib
import warnings
from typing import Dict, List

import nestedcv as ncv
import pandas as pd


def main(
    *,
    data_dir: str,
    identifier: str,
    out_dir: str,
    correlation_threshold: float,
    correlation_method: str,
):
    # some input sanity checks:
    if not 0.0 <= correlation_threshold <= 1.0:
        raise ValueError("correlation_threshold is not in [0, 1]")
    if correlation_method not in ["spearman", "pearson"]:
        raise ValueError("correlation_method is not in {'spearman', 'pearson'}")

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    fn = os.path.join(data_dir, f"{identifier}.csv")
    data = pd.read_csv(fn, sep=",")

    print(f"Shape of dataframe loaded from {fn}: {data.shape}\n")

    # check, if there are NaN DF entries:
    if data.isna().sum().sum() != 0:
        raise ValueError(f"{fn} contains NaN entries.")

    if correlation_threshold == 1.0:
        print(
            f"Correlation grouping with threshold {correlation_threshold} requested. "
            "Since 1.0 is the maximal possible correlation, the data will be saved unchanged.\n"
        )
    else:
        print(
            f"Grouping features based on {correlation_method} correlation "
            f"with threshold {correlation_threshold}:\n"
        )

        # warn if 'Dose' is strongly correlated with any AB signal:
        correlation = data.drop(columns=["Patient", "group", "Protection", "TimePointOrder"]).corr(
            method=correlation_method
        )
        if (
            correlation.loc[~correlation.index.isin(["Dose"]), "Dose"]
            .gt(correlation_threshold)
            .any()
        ):
            warnings.warn(
                f"The Dose is threshold-exceedingly correlated (>{correlation_threshold}) "
                "with at least one antibody signal."
            )
        del correlation

        intensities = data.drop(
            columns=["Patient", "group", "Protection", "TimePointOrder", "Dose"]
        )
        print(f"Shape of intensities dataframe: {intensities.shape}")
        print(f"# of variants: {intensities.shape[1]}\n")

        # check if there are constant features (and drop them):
        const_features = intensities.columns[intensities.std(axis=0.0) == 0].to_list()
        if const_features:
            warnings.warn(f"{fn} contains constant features.")
            intensities.drop(columns=const_features, inplace=True)
            print(f"{fn} contains constant features:")
            for i in const_features:
                print(i)
            print(f"Shape of dataframe after dropping constant features: {intensities.shape}\n")

        # calculate either pearson or spearman correlation dataframe
        correlation = intensities.corr(method=correlation_method)
        print(f"correlation matrix shape: {correlation.shape}")
        print(
            "correlation matrix memory usage (kB): "
            f"{correlation.memory_usage(deep=True).sum()/1000}\n"
        )
        if not correlation.index.equals(correlation.columns):
            raise ValueError("correlation.index != correlation.columns")

        # check for variants with NaN correlation:
        if correlation.isna().sum().sum() != 0:
            raise ValueError(
                f"{fn}: There are NaN correlation matrix entries "
                "not belonging to constant features."
            )

        # construct a dict of correlated features
        if len(correlation.index) != len(set(correlation.index)):
            raise ValueError("len(correlation.index) != len(set(correlation.index))")
        all_features = set(correlation.index)
        groups: Dict[str, List[str]] = dict()
        for i in correlation.index.to_list():
            group = set(
                correlation.loc[
                    i,
                    ~correlation.loc[i, :].between(
                        -correlation_threshold, correlation_threshold, inclusive="both"
                    ),
                ].index
            )
            if i not in group:
                raise ValueError(f"{i} is not contained in its own group.")
            groups[i] = sorted(group & all_features)
            all_features = all_features - group
        groups_values_set = set()
        for i in groups.values():
            groups_values_set.update(i)
        if groups_values_set != set(correlation.index):
            raise ValueError("groups_values_set != set(correlation.index)")

        # save dict as json:
        fn = (
            f"{identifier}_{correlation_method}_correlation_"
            f"grouped_features_threshold{correlation_threshold}"
        )
        ncv.save_json(groups, out_dir, fn, timestamp=False, overwrite=True)

        # keep only one representative per group:
        keep = []
        keep = []
        count = []
        for key in groups.keys():
            if len(groups[key]) > 0:
                if len(groups[key]) > 1:
                    keep.append(
                        correlation.loc[groups[key], groups[key]]
                        .mean(axis="index", skipna=True)
                        .idxmax(skipna=True)
                    )
                else:
                    keep.append(key)
                count.append(len(groups[key]))
        count, keep_sorted = zip(*sorted(zip(count, keep), reverse=True, key=lambda x: x[0]))
        hist = pd.DataFrame(list(count), index=list(keep_sorted), columns=["# correlated features"])

        fn = os.path.join(
            out_dir,
            (
                f"{identifier}_{correlation_method}_correlated_group_sizes"
                f"_threshold{correlation_threshold}.csv"
            ),
        )
        hist.to_csv(fn, sep=",")
        fn = os.path.join(
            out_dir,
            (
                f"{identifier}_{correlation_method}_correlated_group_sizes"
                f"_threshold{correlation_threshold}.pdf"
            ),
        )
        # hist_max = np.amax(hist.to_numpy())
        # fig, ax = plt.subplots()
        # ax.hist(hist.to_numpy(), bins=hist_max, log=True, rwidth=1.0)
        # ax.set_xlim(0, hist_max + 1)
        # plt.title(f"Histogram of group sizes S (S_max = #bins = {hist_max})")
        # plt.savefig(fn, format="pdf")
        # plt.close()

        print(f"# of (key-)variants to keep: {len(keep)}\n")

        intensities = intensities.loc[:, keep]
        metadata = data[["Patient", "group", "Protection", "TimePointOrder", "Dose"]]
        if intensities.index.to_list() != metadata.index.to_list():
            raise ValueError("intensities.index != data.index")
        if set(intensities.columns.to_list()) & set(metadata.columns.to_list()):
            raise ValueError(
                f"Columns overlap when joining intensities dataframe "
                f"with metadata for {identifier} data for threshold {correlation_threshold}."
            )
        data = metadata.join(intensities, how="left", sort=False)
        if set(data.columns.to_list()) != set(intensities.columns.to_list()) | set(
            metadata.columns.to_list()
        ):
            raise ValueError(
                "set(data.columns.to_list) != set(intensities.columns.to_list()) "
                "| set(metadata.columns.to_list())"
            )

    # save reduced DF:
    fn = os.path.join(
        out_dir,
        (f"{identifier}_{correlation_method}_filtered" f"_threshold{correlation_threshold}.csv"),
    )
    data.to_csv(fn, sep=",", index=False)

    print(f"Shape of kept {identifier} data: {data.shape}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Function to group strongly covarying features together.")
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
            "This string will be appended by the time point and '.csv'. If you pass, for example, "
            "'--timepoint III14' and '--data-file-id preprocessed_whole_data', the resulting file "
            "name will be 'preprocessed_whole_data_III14.csv'."
        ),
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        metavar="DIR",
        required=True,
        help="Path to the directory to which the output shall be written.",
    )
    parser.add_argument(
        "--correlation_threshold",
        dest="correlation_threshold",
        required=True,
        type=float,
        help=(
            "The correlation coefficient threshold t determining if two features are "
            "grouped together: if |correlation| > t they are grouped together."
        ),
    )
    parser.add_argument(
        "--correlation_method",
        dest="correlation_method",
        required=True,
        type=str,
        help=(
            "The method used to calculate the correlation. "
            "Use either 'spearman' (Spearman rank correlation) or 'pearson' (Pearson correlation)."
        ),
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        identifier=args.data_file_id,
        out_dir=args.out_dir,
        correlation_threshold=args.correlation_threshold,
        correlation_method=args.correlation_method,
    )
