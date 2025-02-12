# Copyright (c) 2022 Bernhard Reuter and Jacqueline Wistuba-Hamprecht.
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
This script precomputes multitask Gram matrices for preprocessed proteome data.

@Author: Bernhard Reuter

"""

import argparse
import json
import os
import pathlib
from itertools import product
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from source.utils import make_kernel_combinations, make_kernel_matrix


def helper(
    m: Tuple[Any, ...],
) -> Dict[str, Union[float, int, str]]:
    # input sanity checks:
    if not isinstance(m, tuple):
        raise ValueError("The model must be given as a tuple of kernel parameters.")
    if not len(m) == 7:
        raise ValueError("Expected model tuple of length 7.")
    for i in range(5):
        if not isinstance(m[i], (float, str)):
            raise ValueError(f"m[{i}] must be either of type float or str.")
    for i in [5, 6]:
        if not isinstance(m[i], (int, str)):
            raise ValueError(f"m[{i}] must be either of type int or str.")

    model = {
        "SA": m[0],
        "SO": m[1],
        "R0": m[2],
        "R1": m[3],
        "R2": m[4],
        "P1": m[5],
        "P2": m[6],
    }
    return model


def calc_and_save_GramMatrices(
    identifier: str,
    data: pd.DataFrame,
    kernel_param: Dict[str, List[Union[float, int, str]]],
    kernel_for_time_series: str,
    kernel_for_dosage: str,
    kernel_for_abSignal: str,
    save_to_dir: str,
    scale: bool = False,
) -> Dict[str, List[Union[np.ndarray, List[str], List[float], Tuple[Any, ...]]]]:
    # check for correct parameter keys:
    keys = {"SA", "SO", "R0", "R1", "R2", "P1", "P2"}
    if not set(kernel_param.keys()) == keys:
        raise ValueError(
            f"Expected multitaskSVM parameters but set(kernel_param.keys()) != {keys}: "
            f"{set(kernel_param.keys())} != {keys}"
        )
    for key in ["SA", "SO", "R0", "R1", "R2"]:
        if not isinstance(kernel_param[key], list):
            raise ValueError(f"kernel_params[{key}] must be a list.")
        if not (
            all(isinstance(x, str) for x in kernel_param[key])
            or all(isinstance(x, float) for x in kernel_param[key])
        ):
            raise ValueError(
                f"All elements of kernel_params[{key}] must be either of type str or float."
            )
    for key in ["P1", "P2"]:
        if not isinstance(kernel_param[key], list):
            raise ValueError(f"kernel_params[{key}] must be a list.")
        if not (
            all(isinstance(x, str) for x in kernel_param[key])
            or all(isinstance(x, int) for x in kernel_param[key])
        ):
            raise ValueError(
                f"All elements of kernel_params[{key}] must be either of type str or int."
            )

    collection: Dict[str, List[Union[np.ndarray, List[str], List[float], Tuple[Any, ...]]]] = dict()
    collection["models"] = []
    collection["kernel_matrices"] = []
    collection["information"] = []
    collection["damping_values_list"] = []
    if (
        kernel_for_time_series == "sigmoid_kernel"
        and kernel_for_dosage == "rbf_kernel"
        and kernel_for_abSignal == "rbf_kernel"
    ):
        for m in product(
            kernel_param["SA"],
            kernel_param["SO"],
            kernel_param["R0"],
            kernel_param["R1"],
            kernel_param["R2"],
            kernel_param["P1"],
            kernel_param["P2"],
        ):
            kernel_matrix, damping_values, info = make_kernel_matrix(
                AB_signals=data.drop(
                    columns=["Patient", "group", "Protection", "TimePointOrder", "Dose"]
                ),
                time_series=data["TimePointOrder"],
                dose=data["Dose"],
                model=helper(m),
                kernel_time_series=kernel_for_time_series,
                kernel_dosage=kernel_for_dosage,
                kernel_abSignals=kernel_for_abSignal,
                scale=scale,
            )
            collection["models"].append(m)
            collection["kernel_matrices"].append(kernel_matrix)
            collection["information"].append(info)
            collection["damping_values_list"].append(damping_values)
            output_file_name = (
                f"{identifier}_SRR_SA_{m[0]}_SO_{m[1]}_R0_{m[2]}_R1_{m[3]:.1E}_"
                f"R2_{m[4]:.1E}_P1_{m[5]}_P2_{m[6]}.npy"
            )
            np.save(os.path.join(save_to_dir, output_file_name), kernel_matrix)

    elif (
        kernel_for_time_series == "sigmoid_kernel"
        and kernel_for_dosage == "poly_kernel"
        and kernel_for_abSignal == "rbf_kernel"
    ):
        for m in product(
            kernel_param["SA"],
            kernel_param["SO"],
            kernel_param["R0"],
            kernel_param["R1"],
            kernel_param["R2"],
            kernel_param["P1"],
            kernel_param["P2"],
        ):
            kernel_matrix, damping_values, info = make_kernel_matrix(
                AB_signals=data.drop(
                    columns=["Patient", "group", "Protection", "TimePointOrder", "Dose"]
                ),
                time_series=data["TimePointOrder"],
                dose=data["Dose"],
                model=helper(m),
                kernel_time_series=kernel_for_time_series,
                kernel_dosage=kernel_for_dosage,
                kernel_abSignals=kernel_for_abSignal,
                scale=scale,
            )
            collection["models"].append(m)
            collection["kernel_matrices"].append(kernel_matrix)
            collection["information"].append(info)
            collection["damping_values_list"].append(damping_values)
            output_file_name = (
                f"{identifier}_SPR_SA_{m[0]}_SO_{m[1]}_R0_{m[2]}_"
                f"R1_{m[3]}_R2_{m[4]:.1E}_P1_{m[5]}_P2_{m[6]}.npy"
            )
            np.save(os.path.join(save_to_dir, output_file_name), kernel_matrix)

    elif (
        kernel_for_time_series == "sigmoid_kernel"
        and kernel_for_dosage == "rbf_kernel"
        and kernel_for_abSignal == "poly_kernel"
    ):
        for m in product(
            kernel_param["SA"],
            kernel_param["SO"],
            kernel_param["R0"],
            kernel_param["R1"],
            kernel_param["R2"],
            kernel_param["P1"],
            kernel_param["P2"],
        ):
            kernel_matrix, damping_values, info = make_kernel_matrix(
                AB_signals=data.drop(
                    columns=["Patient", "group", "Protection", "TimePointOrder", "Dose"]
                ),
                time_series=data["TimePointOrder"],
                dose=data["Dose"],
                model=helper(m),
                kernel_time_series=kernel_for_time_series,
                kernel_dosage=kernel_for_dosage,
                kernel_abSignals=kernel_for_abSignal,
                scale=scale,
            )
            collection["models"].append(m)
            collection["kernel_matrices"].append(kernel_matrix)
            collection["information"].append(info)
            collection["damping_values_list"].append(damping_values)
            output_file_name = (
                f"{identifier}_SRP_SA_{m[0]}_SO_{m[1]}_R0_{m[2]}_"
                f"R1_{m[3]:.1E}_R2_{m[4]}_P1_{m[5]}_P2_{m[6]}.npy"
            )
            np.save(os.path.join(save_to_dir, output_file_name), kernel_matrix)

    elif (
        kernel_for_time_series == "sigmoid_kernel"
        and kernel_for_dosage == "poly_kernel"
        and kernel_for_abSignal == "poly_kernel"
    ):
        for m in product(
            kernel_param["SA"],
            kernel_param["SO"],
            kernel_param["R0"],
            kernel_param["R1"],
            kernel_param["R2"],
            kernel_param["P1"],
            kernel_param["P2"],
        ):
            kernel_matrix, damping_values, info = make_kernel_matrix(
                AB_signals=data.drop(
                    columns=["Patient", "group", "Protection", "TimePointOrder", "Dose"]
                ),
                time_series=data["TimePointOrder"],
                dose=data["Dose"],
                model=helper(m),
                kernel_time_series=kernel_for_time_series,
                kernel_dosage=kernel_for_dosage,
                kernel_abSignals=kernel_for_abSignal,
                scale=scale,
            )
            collection["models"].append(m)
            collection["kernel_matrices"].append(kernel_matrix)
            collection["information"].append(info)
            collection["damping_values_list"].append(damping_values)
            output_file_name = (
                f"{identifier}_SPP_SA_{m[0]}_SO_{m[1]}_R0_{m[2]}_"
                f"R1_{m[3]}_R2_{m[4]}_P1_{m[5]}_P2_{m[6]}.npy"
            )
            np.save(os.path.join(save_to_dir, output_file_name), kernel_matrix)
    elif (
        kernel_for_time_series == "rbf_kernel"
        and kernel_for_dosage == "rbf_kernel"
        and kernel_for_abSignal == "rbf_kernel"
    ):
        for m in product(
            kernel_param["SA"],
            kernel_param["SO"],
            kernel_param["R0"],
            kernel_param["R1"],
            kernel_param["R2"],
            kernel_param["P1"],
            kernel_param["P2"],
        ):
            kernel_matrix, damping_values, info = make_kernel_matrix(
                AB_signals=data.drop(
                    columns=["Patient", "group", "Protection", "TimePointOrder", "Dose"]
                ),
                time_series=data["TimePointOrder"],
                dose=data["Dose"],
                model=helper(m),
                kernel_time_series=kernel_for_time_series,
                kernel_dosage=kernel_for_dosage,
                kernel_abSignals=kernel_for_abSignal,
                scale=scale,
            )
            collection["models"].append(m)
            collection["kernel_matrices"].append(kernel_matrix)
            collection["information"].append(info)
            collection["damping_values_list"].append(damping_values)
            output_file_name = (
                f"{identifier}_RRR_SA_{m[0]}_SO_{m[1]}_R0_{m[2]:.1E}_"
                f"R1_{m[3]:.1E}_R2_{m[4]:.1E}_P1_{m[5]}_P2_{m[6]}.npy"
            )
            np.save(os.path.join(save_to_dir, output_file_name), kernel_matrix)

    elif (
        kernel_for_time_series == "rbf_kernel"
        and kernel_for_dosage == "poly_kernel"
        and kernel_for_abSignal == "rbf_kernel"
    ):
        for m in product(
            kernel_param["SA"],
            kernel_param["SO"],
            kernel_param["R0"],
            kernel_param["R1"],
            kernel_param["R2"],
            kernel_param["P1"],
            kernel_param["P2"],
        ):
            kernel_matrix, damping_values, info = make_kernel_matrix(
                AB_signals=data.drop(
                    columns=["Patient", "group", "Protection", "TimePointOrder", "Dose"]
                ),
                time_series=data["TimePointOrder"],
                dose=data["Dose"],
                model=helper(m),
                kernel_time_series=kernel_for_time_series,
                kernel_dosage=kernel_for_dosage,
                kernel_abSignals=kernel_for_abSignal,
                scale=scale,
            )
            collection["models"].append(m)
            collection["kernel_matrices"].append(kernel_matrix)
            collection["information"].append(info)
            collection["damping_values_list"].append(damping_values)
            output_file_name = (
                f"{identifier}_RPR_SA_{m[0]}_SO_{m[1]}_R0_{m[2]:.1E}_"
                f"R1_{m[3]}_R2_{m[4]:.1E}_P1_{m[5]}_P2_{m[6]}.npy"
            )
            np.save(os.path.join(save_to_dir, output_file_name), kernel_matrix)

    elif (
        kernel_for_time_series == "rbf_kernel"
        and kernel_for_dosage == "rbf_kernel"
        and kernel_for_abSignal == "poly_kernel"
    ):
        for m in product(
            kernel_param["SA"],
            kernel_param["SO"],
            kernel_param["R0"],
            kernel_param["R1"],
            kernel_param["R2"],
            kernel_param["P1"],
            kernel_param["P2"],
        ):
            kernel_matrix, damping_values, info = make_kernel_matrix(
                AB_signals=data.drop(
                    columns=["Patient", "group", "Protection", "TimePointOrder", "Dose"]
                ),
                time_series=data["TimePointOrder"],
                dose=data["Dose"],
                model=helper(m),
                kernel_time_series=kernel_for_time_series,
                kernel_dosage=kernel_for_dosage,
                kernel_abSignals=kernel_for_abSignal,
                scale=scale,
            )
            collection["models"].append(m)
            collection["kernel_matrices"].append(kernel_matrix)
            collection["information"].append(info)
            collection["damping_values_list"].append(damping_values)
            output_file_name = (
                f"{identifier}_RRP_SA_{m[0]}_SO_{m[1]}_R0_{m[2]:.1E}_"
                f"R1_{m[3]:.1E}_R2_{m[4]}_P1_{m[5]}_P2_{m[6]}.npy"
            )
            np.save(os.path.join(save_to_dir, output_file_name), kernel_matrix)

    elif (
        kernel_for_time_series == "rbf_kernel"
        and kernel_for_dosage == "poly_kernel"
        and kernel_for_abSignal == "poly_kernel"
    ):
        for m in product(
            kernel_param["SA"],
            kernel_param["SO"],
            kernel_param["R0"],
            kernel_param["R1"],
            kernel_param["R2"],
            kernel_param["P1"],
            kernel_param["P2"],
        ):
            kernel_matrix, damping_values, info = make_kernel_matrix(
                AB_signals=data.drop(
                    columns=["Patient", "group", "Protection", "TimePointOrder", "Dose"]
                ),
                time_series=data["TimePointOrder"],
                dose=data["Dose"],
                model=helper(m),
                kernel_time_series=kernel_for_time_series,
                kernel_dosage=kernel_for_dosage,
                kernel_abSignals=kernel_for_abSignal,
                scale=scale,
            )
            collection["models"].append(m)
            collection["kernel_matrices"].append(kernel_matrix)
            collection["information"].append(info)
            collection["damping_values_list"].append(damping_values)
            output_file_name = (
                f"{identifier}_RPP_SA_{m[0]}_SO_{m[1]}_R0_{m[2]:.1E}_"
                f"R1_{m[3]}_R2_{m[4]}_P1_{m[5]}_P2_{m[6]}.npy"
            )
            np.save(os.path.join(save_to_dir, output_file_name), kernel_matrix)

    return collection


def heatmap2d(arr: np.ndarray) -> None:
    plt.figure(figsize=(9.5, 8))
    plt.imshow(arr, cmap="hot")
    plt.colorbar()
    plt.show()
    plt.close()


def overview(
    collection: Dict[str, List[Union[np.ndarray, List[str], List[float], Tuple[Any, ...]]]],
    plot: bool = False,
) -> None:
    damping_value_sums = []
    for damping_values in collection["damping_values_list"]:
        damping_value_sums.append(np.sum(damping_values))
    print("maximum sum:", np.max(damping_value_sums))
    print("sorted sums:", np.sort(damping_value_sums)[::-1])
    print("")

    if plot:
        for i in range(10):
            print(collection["models"][i])
            print(np.sum(collection["damping_values_list"][i]))
            plt.plot(
                np.sort(np.linalg.eigvals(collection["kernel_matrices"][i]), kind="mergesort")[
                    ::-1
                ],
                marker=".",
            )
            plt.show()
            plt.close()
            heatmap2d(np.array(collection["kernel_matrices"][i]))


def looper(
    *,
    data: pd.DataFrame,
    params: Dict[str, np.ndarray],
    identifier: str,
    out_dir: str,
    combinations: List[str],
) -> None:
    for id in combinations:
        if id == "SRR":
            kernel_for_time_series = "sigmoid_kernel"
            kernel_for_dosage = "rbf_kernel"
            kernel_for_abSignal = "rbf_kernel"
        elif id == "SRP":
            kernel_for_time_series = "sigmoid_kernel"
            kernel_for_dosage = "rbf_kernel"
            kernel_for_abSignal = "poly_kernel"
        elif id == "SPR":
            kernel_for_time_series = "sigmoid_kernel"
            kernel_for_dosage = "poly_kernel"
            kernel_for_abSignal = "rbf_kernel"
        elif id == "SPP":
            kernel_for_time_series = "sigmoid_kernel"
            kernel_for_dosage = "poly_kernel"
            kernel_for_abSignal = "poly_kernel"
        elif id == "RRR":
            kernel_for_time_series = "rbf_kernel"
            kernel_for_dosage = "rbf_kernel"
            kernel_for_abSignal = "rbf_kernel"
        elif id == "RRP":
            kernel_for_time_series = "rbf_kernel"
            kernel_for_dosage = "rbf_kernel"
            kernel_for_abSignal = "poly_kernel"
        elif id == "RPR":
            kernel_for_time_series = "rbf_kernel"
            kernel_for_dosage = "poly_kernel"
            kernel_for_abSignal = "rbf_kernel"
        elif id == "RPP":
            kernel_for_time_series = "rbf_kernel"
            kernel_for_dosage = "poly_kernel"
            kernel_for_abSignal = "poly_kernel"
        else:
            raise ValueError(f"Unknown combination {id}")

        kernel_param = make_kernel_combinations(
            kernel_params=params,
            kernel_time_series=kernel_for_time_series,
            kernel_dosage=kernel_for_dosage,
            kernel_abSignal=kernel_for_abSignal,
        )

        fn = os.path.join(out_dir, f"kernel_comb_{id}.json")
        with open(fn, "w") as fp:
            json.dump(kernel_param, fp)

        collection = calc_and_save_GramMatrices(
            identifier=identifier,
            data=data,
            kernel_param=kernel_param,
            kernel_for_time_series=kernel_for_time_series,
            kernel_for_dosage=kernel_for_dosage,
            kernel_for_abSignal=kernel_for_abSignal,
            save_to_dir=out_dir,
        )

        fn = os.path.join(out_dir, f"collected_damping_info_{id}.json")
        with open(fn, "w") as fp:
            json.dump({x: collection[x] for x in collection if x not in "kernel_matrices"}, fp)

        overview(collection)


def main(
    *,
    data_file: str,
    out_dir: str,
    identifier: str,
) -> None:
    from source.multitaskSVM_config import kernel_params as params

    combinations = ["RRR", "RPR", "RRP", "RPP"]

    fn = os.path.join(data_file)
    data = pd.read_csv(fn)

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    looper(
        data=data, params=params, identifier=identifier, out_dir=out_dir, combinations=combinations
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Script to precompute multitask Gram matrices from preprocessed proteome data."
        )
    )
    parser.add_argument(
        "--data-file",
        dest="data_file",
        metavar="PATH",
        required=True,
        help=("Path to the file were the preprocessed proteome data is stored."),
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        metavar="DIR",
        required=True,
        help="Path to the directory to which the output shall be written.",
    )
    parser.add_argument(
        "--identifier",
        dest="identifier",
        help=(
            "Prefix to identify the precomputed kernel matrices (stored as .npy files), "
            "i.e., 'kernel_matrix'."
        ),
    )
    args = parser.parse_args()

    main(
        data_file=args.data_file,
        out_dir=args.out_dir,
        identifier=args.identifier,
    )
