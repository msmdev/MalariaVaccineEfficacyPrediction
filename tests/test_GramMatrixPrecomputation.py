# Copyright (c) 2022 Bernhard Reuter and Jacqueline Wistuba-Hamprecht.
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
@Author: Bernhard Reuter

"""

__author__ = __maintainer__ = "Bernhard Reuter"
__email__ = "bernhard-reuter@gmx.de"
__copyright__ = "Copyright 2022, Bernhard Reuter"

import numpy as np
from tests.conftest import assert_allclose
import os
from source.utils import make_kernel_combinations
import pandas as pd
from source.utils import DataSelector
from itertools import product


def test_GramMatrices() -> None:

    from source.multitaskSVM_config import kernel_params as params

    # test_dir = os.getcwd()
    # main_dir = '/'.join(test_dir.split('/')[:-1])
    main_dir = os.getcwd()
    ground_truth_dir = os.path.join(
        main_dir, 'data/precomputed_multitask_kernels/threshold1.0backup'
    )
    actual_dir = os.path.join(main_dir, 'data/precomputed_multitask_kernels/threshold1.0')

    for dataset, identifier in zip(
        ['selective', 'whole'], ['kernel_matrix_SelectiveSet', 'kernel_matrix']
    ):

        # initialize running index array for DataSelector
        data = pd.read_csv(
            (f"{main_dir}/data/proteome_data/correlationFiltering/preprocessed_{dataset}_"
             "data_spearman_filtered_threshold1.0_all.csv"),
            header=0,
        )
        y = data.loc[:, 'Protection'].to_numpy()
        assert y.size * y.size < np.iinfo(np.uint32).max, \
            f"y is to large: y.size * y.size >= {np.iinfo(np.uint32).max}"
        X = np.array(
            [x for x in range(y.size * y.size)],
            dtype=np.uint32
        ).reshape((y.size, y.size))

        for id in ['SRR', 'SPR', 'SRP', 'SPP', 'RRR', 'RPR', 'RRP', 'RPP']:

            if id == 'SRR':
                kernel_for_time_series = "sigmoid_kernel"
                kernel_for_dosage = "rbf_kernel"
                kernel_for_abSignal = "rbf_kernel"
            elif id == 'SRP':
                kernel_for_time_series = "sigmoid_kernel"
                kernel_for_dosage = "rbf_kernel"
                kernel_for_abSignal = "poly_kernel"
            elif id == 'SPR':
                kernel_for_time_series = "sigmoid_kernel"
                kernel_for_dosage = "poly_kernel"
                kernel_for_abSignal = "rbf_kernel"
            elif id == 'SPP':
                kernel_for_time_series = "sigmoid_kernel"
                kernel_for_dosage = "poly_kernel"
                kernel_for_abSignal = "poly_kernel"
            elif id == 'RRR':
                kernel_for_time_series = "rbf_kernel"
                kernel_for_dosage = "rbf_kernel"
                kernel_for_abSignal = "rbf_kernel"
            elif id == 'RRP':
                kernel_for_time_series = "rbf_kernel"
                kernel_for_dosage = "rbf_kernel"
                kernel_for_abSignal = "poly_kernel"
            elif id == 'RPR':
                kernel_for_time_series = "rbf_kernel"
                kernel_for_dosage = "poly_kernel"
                kernel_for_abSignal = "rbf_kernel"
            elif id == 'RPP':
                kernel_for_time_series = "rbf_kernel"
                kernel_for_dosage = "poly_kernel"
                kernel_for_abSignal = "poly_kernel"
            else:
                raise ValueError(f'Unknown combination {id}')

            kernel_param = make_kernel_combinations(
                kernel_params=params,
                kernel_time_series=kernel_for_time_series,
                kernel_dosage=kernel_for_dosage,
                kernel_abSignal=kernel_for_abSignal,
            )

            for m in product(kernel_param["SA"], kernel_param["SO"], kernel_param["R0"],
                             kernel_param["R1"], kernel_param["R2"], kernel_param["P1"],
                             kernel_param["P2"]):

                P1 = m[5]
                P2 = m[6]
                if isinstance(P1, int):
                    P1 = float(P1)
                if isinstance(P2, int):
                    P2 = float(P2)

                gram_ground_truth = DataSelector(
                    kernel_directory=ground_truth_dir,
                    identifier=f'{identifier}_{id}',
                    SA=m[0],
                    SO=m[1],
                    R0=m[2],
                    R1=m[3],
                    R2=m[4],
                    P1=P1,
                    P2=P2,
                ).fit(X, y).transform(X)

                gram_actual = DataSelector(
                    kernel_directory=actual_dir,
                    identifier=f'{identifier}_{id}',
                    SA=m[0],
                    SO=m[1],
                    R0=m[2],
                    R1=m[3],
                    R2=m[4],
                    P1=m[5],
                    P2=m[6],
                ).fit(X, y).transform(X)

                assert_allclose(gram_actual, gram_ground_truth)
