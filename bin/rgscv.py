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
Find the best parameters using 10-times repeated NestedGridsearchCV.

Will save the results to a various .tsv/.xslx files.

Activate associated environment first: conda activate malaria_env

@Author: Bernhard Reuter

"""

import argparse
import numpy as np
import os.path
import pandas as pd
import pathlib
from pprint import pprint
import scipy
import sklearn
import sys
import traceback
import warnings
import nestedcv as ncv
from source.utils import CustomPredefinedSplit, assign_folds
from typing import Any, Dict, Union, Optional, List


def main(
    *,
    ana_dir: str,
    data_dir: str,
    data_file_id: str,
    method: str,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]],
    estimator,
    n_jobs: Optional[int] = None,
    Nexp: int = 10,
) -> None:

    # Generate a timestamp
    timestamp = ncv.generate_timestamp()

    print('========================================')
    print(f'numpy version: {np.__version__}')
    print(f'pandas version: {pd.__version__}')
    print(f'scikit-learn version: {sklearn.__version__}')
    print(f'scipy version: {scipy.__version__}')
    print('========================================\n')
    print(f'data file identifier: {data_file_id}')
    print(f'estimator: {type(estimator)}')
    print(f'parameter grid: {param_grid}\n')
    print(f'start time: {timestamp}\n')

    # Create directories for the output files
    maindir = os.path.join(ana_dir, 'RGSCV/')
    pathlib.Path(maindir).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(os.path.join(data_dir, f'{data_file_id}_all.csv'), header=0)

    groups_all = data.loc[:, 'group'].to_numpy()
    y_all = data.loc[:, 'Protection'].to_numpy()

    # initialize result dict
    results: Dict[str, List[Any]] = dict()
    results['time'] = []
    results['scoring'] = []
    results['best_params'] = []
    results['best_score'] = []

    times = ['III14', 'C-1', 'C28']
    scorings = ['mcc', 'precision_recall_auc', 'roc_auc']
    for step, time in enumerate(times):

        print('++++++++++++++++++++++++++++++++++++++++')
        print(f'{time} start: {ncv.generate_timestamp()}\n')

        # define prefix for filenames:
        prefix = f'{time}'

        if method == 'multitaskSVM':
            y = y_all
            groups = groups_all
            # initialize running index array for DataSelector
            if not y.size * y.size < np.iinfo(np.uint32).max:
                raise ValueError(f"y is to large: y.size * y.size >= {np.iinfo(np.uint32).max}")
            X = np.array(
                [x for x in range(y.size * y.size)],
                dtype=np.uint32
            ).reshape((y.size, y.size))
            print(f'shape of running index array: {X.shape}/n')
        else:
            data_at_timePoint = pd.read_csv(
                os.path.join(data_dir, f'{data_file_id}_{time}.csv'),
                header=0,
            )
            X = data_at_timePoint.drop(
                columns=['Patient', 'group', 'Protection', 'TimePointOrder']
            ).to_numpy()  # including dose
            groups = data_at_timePoint.loc[:, 'group'].to_numpy()
            y = data_at_timePoint.loc[:, 'Protection'].to_numpy()

        print('shape of binary response array:', y.size)
        print('number of positives:', np.sum(y))
        print('number of positives divided by total number of samples:', np.sum(y)/y.size)
        print('')

        # initialize test folds and CV splitters for outer CV
        cv = []
        print('----------------------------------------')
        print('Predefined CV folds:\n')
        for rep in range(Nexp):
            print(f'CV folds for repetition {rep}:')
            test_fold, train_fold = assign_folds(
                labels=y_all,
                groups=groups_all,
                delta=40,
                step=step,
                n_splits=5,
                shuffle=True,
                print_info=False,
                random_state=rep,
            )
            if method != 'multitaskSVM':
                test_fold = test_fold[40 * step: 40 * (step + 1)]
                train_fold = train_fold[40 * step: 40 * (step + 1)]
                if not np.all(test_fold == train_fold):
                    raise ValueError(f"test_fold != train_fold: {test_fold} != {train_fold}")
            cv.append(CustomPredefinedSplit(test_fold, train_fold))
            print(
                f"train_fold: {train_fold} "
                f"test_fold: {test_fold}"
            )
            cps = CustomPredefinedSplit(test_fold, train_fold)
            for i, (train_index, test_index) in enumerate(cps.split()):
                print(
                    f"TRAIN (len={len(train_index)}): {train_index} "
                    f"TEST (len={len(test_index)}): {test_index}"
                )
            print('')
        print('----------------------------------------\n')

        gs = ncv.RepeatedGridSearchCV(
            estimator,
            param_grid,
            scoring=scorings,
            cv=cv,
            n_jobs=n_jobs,
            Nexp=Nexp,
            save_to=None,
            reproducible=False,
        )
        gs.fit(X, y, groups)

        filename = f"{prefix}_RepeatedGridSearchCV_cv_results"
        fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=False)
        pd.DataFrame(data=gs.cv_results_).to_csv(fn, sep='\t', na_rep='nan')
        fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=False)
        pd.DataFrame(data=gs.cv_results_).to_excel(fn, na_rep='nan')

        opt_params = gs.opt_params_
        print(f'{time} opt_params:')
        pprint(opt_params)
        print('')

        opt_scores = gs.opt_scores_
        print(f'{time} opt_scores:')
        pprint(opt_scores)
        print('')

        for scoring in scorings:
            results['time'].append(time)
            results['scoring'].append(scoring)
            results['best_params'].append(opt_params[scoring])
            results['best_score'].append(opt_scores[scoring])

        print(f'{time} end: {ncv.generate_timestamp()}')
        print('++++++++++++++++++++++++++++++++++++++++\n')

    print('results:')
    pprint(results)
    print('')

    filename = "RepeatedGridSearchCV_results"
    fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=False)
    pd.DataFrame(data=results).to_csv(fn, sep='\t', na_rep='nan')
    fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=False)
    pd.DataFrame(data=results).to_excel(fn, na_rep='nan')

    print(f'end time: {ncv.generate_timestamp()}')
    print('========================================\n')


if __name__ == "__main__":

    warning_file = open("warnings.log", "w")

    def warn_with_traceback(
        message, category, filename, lineno, file=None, line=None
    ):
        try:
            warning_file.write(warnings.formatwarning(message, category, filename, lineno, line))
            traceback.print_stack(file=warning_file)
        except Exception:
            sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
            traceback.print_stack(file=sys.stderr)

    warnings.showwarning = warn_with_traceback
    warnings.simplefilter("default")
    warnings.filterwarnings(
        "ignore",
        message=(
            r"`np.bool` is a deprecated alias for the builtin `bool`.*"
        ),
        category=DeprecationWarning,
        module=r".*_ranking",
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            "Divide through zero encountered while trying to calculate the MCC. "
            "MCC is set to zero."
        ),
        category=UserWarning,
        module=r".*nestedcv"
    )

    parser = argparse.ArgumentParser(
        description=('Function to run repeated grid-search cross-validation.')
    )
    parser.add_argument(
        '--analysis-dir', dest='analysis_dir', metavar='DIR', required=True,
        help='Path to the directory were the analysis shall be performed and stored.'
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
            "'--data-file-id preprocessed_whole_data', the resulting file name at time point "
            "'III14' will be 'preprocessed_whole_data_III14.csv'.")
    )
    parser.add_argument(
        '--method', dest='method', required=True,
        help="Choose the method (either 'RF', 'RLR' or 'SVM') to model the data."
    )
    parser.add_argument(
        '--Nexp',
        dest='Nexp',
        type=int,
        default=10,
        help='Number of grid-search cross-validation repetitions.',
    )
    parser.add_argument(
        '--kernel-dir',
        dest='kernel_dir',
        metavar='DIR',
        help='Path to the directory were the precomputed Gram matrices are located.'
    )
    parser.add_argument(
        '--combination',
        dest='combination',
        help="Kernel combination. Supply 'SPP', 'SPR', 'SRP', or 'SRR'."
    )
    parser.add_argument(
        '--identifier',
        dest='identifier',
        help=("Prefix to identify the precomputed kernel matrices (stored as .npy files), "
              "i.e., 'kernel_matrix' or 'kernel_matrix_SelectiveSet'.")
    )
    args = parser.parse_args()

    method = args.method
    if method == 'SVM':
        from source.SVM_config import estimator, param_grid, n_jobs
    elif method == 'RLR':
        from source.RLR_config import estimator, param_grid, n_jobs
    elif method == 'RF':
        from source.RF_config import estimator, param_grid, n_jobs
    elif method == 'multitaskSVM':
        from source.multitaskSVM_config import configurator
        param_grid, estimator, n_jobs = configurator(
            combination=args.combination,
            identifier=args.identifier,
            kernel_dir=args.kernel_dir,
        )
    else:
        raise ValueError(f"Unexpected method '{method}' passed.")

    try:
        main(
            ana_dir=args.analysis_dir,
            data_dir=args.data_dir,
            data_file_id=args.data_file_id,
            method=method,
            param_grid=param_grid,
            estimator=estimator,
            n_jobs=n_jobs,
            Nexp=args.Nexp,
        )
    finally:

        warning_file.close()
