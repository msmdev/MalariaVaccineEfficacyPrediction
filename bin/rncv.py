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
import re
import scipy
import sklearn
import sys
import traceback
import warnings
from sklearn.model_selection import StratifiedGroupKFold
from typing import Any, Dict, Union, Optional, List
import nestedcv as ncv
from source.utils import CustomPredefinedSplit, assign_folds


def main(
    *,
    ana_dir: str,
    data_dir: str,
    data_file_id: str,
    method: str,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]],
    estimator,
    n_jobs: Optional[int] = None,
    Nexp1: int = 1,
    Nexp2: int = 10,
) -> None:

    # Generate a timestamp
    timestamp = ncv.generate_timestamp()

    print('========================================')
    print('sys.path:', sys.path)
    print('scikit-learn version:', sklearn.__version__)
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)
    print('scipy version:', scipy.__version__)
    print(f'estimator: {type(estimator)}')
    print('========================================\n')

    print(f'data file identifier: {data_file_id}\n')
    print(f'parameter grid: {param_grid}\n')
    print(f'start time: {timestamp}\n')

    # Create directories for the output files
    maindir = os.path.join(ana_dir, 'RNCV/')
    pathlib.Path(maindir).mkdir(parents=True, exist_ok=True)
    rfiledir = os.path.join(ana_dir, 'RNCV/results/')
    pathlib.Path(rfiledir).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(os.path.join(data_dir, f'{data_file_id}_all.csv'), header=0)

    groups_all = data.loc[:, 'group'].to_numpy()
    y_all = data.loc[:, 'Protection'].to_numpy()

    # initialize the key result collector:
    key_results: Dict[str, List[Any]] = {}
    key_results['time'] = []
    key_results['scoring'] = []
    key_results['top_params'] = []
    key_result_keys = [
        'ncv_mcc',
        'test_mcc',
        'ncv_precision_recall_auc',
        'test_precision_recall_auc',
        'ncv_roc_auc',
        'test_roc_auc',
        'mean_ncv_mcc',
        'mean_test_mcc',
        'mean_ncv_precision_recall_auc',
        'mean_test_precision_recall_auc',
        'mean_ncv_roc_auc',
        'mean_test_roc_auc',
    ]
    for key in key_result_keys:
        key_results[key] = []

    for step, time in enumerate(['III14', 'C-1', 'C28']):

        print('++++++++++++++++++++++++++++++++++++++++')
        print(f'{time} start: {ncv.generate_timestamp()}\n')

        # define prefix for filenames:
        prefix = f'{time}'

        rng = np.random.RandomState(0)

        if method == 'multitaskSVM':
            y = y_all
            groups = groups_all
            # initialize running index array for DataSelector
            assert y.size * y.size < np.iinfo(np.uint32).max, \
                f"y is to large: y.size * y.size >= {np.iinfo(np.uint32).max}"
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
        outer_cv = []
        print('----------------------------------------')
        print('Predefined CV folds:\n')
        for rep in range(Nexp2):
            print(f'CV folds for repetition {rep}:')
            test_fold, train_fold = assign_folds(
                y_all, groups_all, 40, step, random_state=rep, print_info=False
            )
            if method != 'multitaskSVM':
                test_fold = test_fold[40 * step: 40 * (step + 1)]
                train_fold = train_fold[40 * step: 40 * (step + 1)]
            assert np.all(test_fold == train_fold), \
                f"test_fold != train_fold: {test_fold} != {train_fold}"
            outer_cv.append(CustomPredefinedSplit(test_fold, train_fold))
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

        # define identifiers used for naming of output files
        rfile = f'{prefix}_RGSCV'

        # set options for NestedGridSearchCV
        cv_options = {
            'scoring': ['precision_recall_auc',
                        'roc_auc',
                        'mcc'],
            'refit': False,
            'tune_threshold': False,
            'save_to': {'directory': rfiledir, 'ID': rfile},
            'save_pred': None,
            'save_inner_to': None,
            'reproducible': False,
            'inner_cv': StratifiedGroupKFold(n_splits=5, random_state=rng, shuffle=True),
            'outer_cv': outer_cv,
            'Nexp1': Nexp1,
            'Nexp2': Nexp2,
            'n_jobs': n_jobs,
        }

        clf_grid = ncv.RepeatedStratifiedNestedCV(
            estimator=estimator,
            param_grid=param_grid,
            cv_options=cv_options
        )
        clf_grid.fit(X, y, None, groups)

        result = clf_grid.repeated_nested_cv_results_

        try:
            ncv.save_json(
                {key: dict(sorted(value.items())) for key, value in result.items()},
                rfiledir,
                f'{prefix}_repeated_nested_cv_results',
                timestamp=False
            )
        except TypeError:
            warnings.warn('Saving the results as JSON file failed due to a TypeError.')

        print(f'{time} results:')
        pprint(result, width=200)
        print('')

        scorings = sorted(result.keys())
        # collect the key results:
        for scoring in scorings:
            key_results['time'].append(time)
            key_results['scoring'].append(scoring)
            key_results['top_params'].append(result[scoring]['ranked_best_inner_params'][0])
            for key in key_result_keys:
                key_results[key].append(result[scoring][key])

        performances = []
        ncv_performances = []
        for scoring in scorings:

            performance = {}
            performance['best_inner_params'] = result[scoring]['best_inner_params']
            scores: List[str] = []
            for score in result[scoring].keys():
                if (bool(re.match(r'best', score)) and not
                        bool(re.search(r'ncv|best_inner|best_params', score))):
                    performance[score] = result[scoring][score]
                elif (bool(re.match(r'test|train', score)) and not
                        bool(re.search(r'ncv|best_inner|best_params', score))):
                    scores.append(re.sub(r'test_|train_', '', score, count=1))
            for score in sorted(scores):
                performance[f'test_{score}'] = result[scoring][f'test_{score}']
                performance[f'train_{score}'] = result[scoring][f'train_{score}']

            assert len(performance['best_inner_params']) % cv_options['Nexp2'] == 0, \
                'len(best_inner_params[%s] modulo Nexp2 != 0 for %s.)' % (prefix, scoring)
            index = []
            n_cv_splits = len(performance['best_inner_params']) // cv_options['Nexp2']
            for x2 in range(cv_options['Nexp2']):
                for x1 in range(n_cv_splits):
                    index.append('outer-repetition' + str(x2) + '_outer-split' + str(x1))
            performances.append(pd.DataFrame(performance, index=index))

            ncv_performance = {}
            scores = []
            for score in result[scoring].keys():
                if bool(re.match(r'ncv', score)):
                    scores.append(re.sub(r'ncv_', '', score, count=1))
            for score in sorted(scores):
                ncv_performance[f'ncv_{score}'] = result[scoring][f'ncv_{score}']
            index = []
            for x2 in range(cv_options['Nexp2']):
                index.append('outer-repetition' + str(x2))
            ncv_performances.append(pd.DataFrame(ncv_performance, index=index))

        # save train/test performance scores to an excel file with one sheet per scoring
        fn = ncv.filename_generator(
            f'{prefix}_train_test_performance_scores',
            extension=".xlsx",
            directory=rfiledir,
            timestamp=False
        )
        if not os.path.exists(fn):
            with pd.ExcelWriter(fn) as writer:
                for dataframe, scoring in zip(performances, scorings):
                    dataframe.to_excel(writer, sheet_name=scoring, na_rep='nan')
        else:
            warnings.warn(f"Overwriting already existing file {fn}.")
            with pd.ExcelWriter(fn) as writer:
                for dataframe, scoring in zip(performances, scorings):
                    dataframe.to_excel(writer, sheet_name=scoring, na_rep='nan')

        # save ncv performance scores to an excel file with one sheet per scoring
        fn = ncv.filename_generator(
            f'{prefix}_ncv_performance_scores',
            extension=".xlsx",
            directory=rfiledir,
            timestamp=False
        )
        if not os.path.exists(fn):
            with pd.ExcelWriter(fn) as writer:
                for dataframe, scoring in zip(ncv_performances, scorings):
                    dataframe.to_excel(writer, sheet_name=scoring, na_rep='nan')
        else:
            warnings.warn(f"Overwriting already existing file {fn}.")
            with pd.ExcelWriter(fn) as writer:
                for dataframe, scoring in zip(ncv_performances, scorings):
                    dataframe.to_excel(writer, sheet_name=scoring, na_rep='nan')
        del performances, ncv_performances

        # collect and save outer cross-validation train- and test-split performance scores
        for i, scoring in enumerate(scorings):

            performance_mean: Dict[str, List[str]] = {}
            performance_min_max_nice: Dict[str, List[str]] = {}
            scores = []
            counter = 0
            threshold_tuning_score = None

            for score in result[scoring].keys():
                if (bool(re.match(r'best', score)) and not
                        bool(re.search(r'ncv|best_inner|best_params', score))):
                    if bool(re.fullmatch(r'best_threshold', score)):
                        dummy = score
                    else:
                        threshold_tuning_score = re.sub(r'best_', '', score)
                        dummy = 'best_threshold_tuning_score'
                        counter += 1
                    if counter > 1:
                        warnings.warn(f'Unexpected "best" score {score}')
                    mean = f'mean_{score}'
                    std = f'std_{score}'
                    min_ = f'min_{score}'
                    max_ = f'max_{score}'
                    dummy_mean = f'mean_{dummy}'
                    dummy_min_max = f'min_max_{dummy}'
                    performance_mean[dummy_mean] = [
                        "%.3f" % result[scoring][mean] + " " + u"\u00B1"
                        + " " + "%.6f" % result[scoring][std]
                    ]
                    performance_min_max_nice[dummy_min_max] = [
                        "[%.3f, %.3f]" % (result[scoring][min_], result[scoring][max_])
                    ]
                elif (bool(re.match(r'test|train', score)) and not
                        bool(re.search(r'ncv|best_inner|best_params', score))):
                    scores.append(re.sub(r'test_|train_', '', score, count=1))

            if not threshold_tuning_score:
                performance_mean['mean_best_threshold_tuning_score'] = ['nan']
                performance_min_max_nice['min_max_best_threshold_tuning_score'] = ['[nan, nan]']
                performance_mean['mean_best_threshold'] = ['nan']
                performance_min_max_nice['min_max_best_threshold'] = ['[nan, nan]']

            for score in sorted(scores):
                for x in ['test', 'train']:
                    mean = f'mean_{x}_{score}'
                    std = f'std_{x}_{score}'
                    min_ = f'min_{x}_{score}'
                    max_ = f'max_{x}_{score}'
                    min_max = f'min_max_{x}_{score}'
                    if bool(re.search(r'brier_loss', score)):
                        performance_mean[mean] = [
                            "%.5f" % result[scoring][mean] + " " + u"\u00B1"
                            + " " + "%.6f" % result[scoring][std]
                        ]
                        performance_min_max_nice[min_max] = [
                            "[%.5f, %.5f]" % (result[scoring][min_], result[scoring][max_])
                        ]
                    elif bool(re.search(r'log_loss', score)):
                        performance_mean[mean] = [
                            "%.4f" % result[scoring][mean] + " " + u"\u00B1"
                            + " " + "%.6f" % result[scoring][std]
                        ]
                        performance_min_max_nice[min_max] = [
                            "[%.4f, %.4f]" % (result[scoring][min_], result[scoring][max_])
                        ]
                    else:
                        performance_mean[mean] = [
                            "%.3f" % result[scoring][mean] + " " + u"\u00B1"
                            + " " + "%.6f" % result[scoring][std]
                        ]
                        performance_min_max_nice[min_max] = [
                            "[%.3f, %.3f]" % (result[scoring][min_], result[scoring][max_])
                        ]

            print(f'{time} mean outer cross-validation train- and test-split performance '
                  f'scores ({scoring} optimized hyperparameters):')
            pprint(performance_mean)
            print('')

            if threshold_tuning_score:
                index = [f'{scoring} | {threshold_tuning_score}']
            else:
                index = [scoring]

            if i == 0:
                performance_mean_df = pd.DataFrame(
                    performance_mean, index=index
                )
                performance_min_max_nice_df = pd.DataFrame(
                    performance_min_max_nice, index=index
                )
            else:
                performance_mean_df = performance_mean_df.append(
                    pd.DataFrame(performance_mean, index=index),
                    verify_integrity=True
                )
                performance_min_max_nice_df = performance_min_max_nice_df.append(
                    pd.DataFrame(performance_min_max_nice, index=index),
                    verify_integrity=True
                )

        filename = f"{prefix}_collected_mean_train_test_performance_scores"
        fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=False)
        performance_mean_df.to_csv(fn, sep='\t', na_rep='')
        fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=False)
        performance_mean_df.to_excel(fn, na_rep='')
        filename = f"{prefix}_collected_formatted_min_max_train_test_performance_scores"
        fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=False)
        performance_min_max_nice_df.to_csv(fn, sep='\t', na_rep='')
        fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=False)
        performance_min_max_nice_df.to_excel(fn, na_rep='')
        del performance_mean_df, performance_min_max_nice_df

        # collect and save nested cross-validation performance scores
        for i, scoring in enumerate(scorings):

            performance_mean = {}
            performance_min_max_nice = {}
            scores = []
            for score in result[scoring].keys():
                if bool(re.match(r'ncv', score)):
                    scores.append(re.sub(r'ncv_', '', score, count=1))
            for score in sorted(scores):
                for x in ['ncv']:
                    mean = f'mean_{x}_' + score
                    std = f'std_{x}_' + score
                    min_ = f'min_{x}_' + score
                    max_ = f'max_{x}_' + score
                    min_max = f'min_max_{x}_' + score
                    if bool(re.search(r'brier_loss', score)):
                        performance_mean[mean] = [
                            "%.5f" % result[scoring][mean] + " " + u"\u00B1"
                            + " " + "%.6f" % result[scoring][std]
                        ]
                        performance_min_max_nice[min_max] = [
                            "[%.5f, %.5f]" % (result[scoring][min_], result[scoring][max_])
                        ]
                    elif bool(re.search(r'log_loss', score)):
                        performance_mean[mean] = [
                            "%.4f" % result[scoring][mean] + " " + u"\u00B1"
                            + " " + "%.6f" % result[scoring][std]
                        ]
                        performance_min_max_nice[min_max] = [
                            "[%.4f, %.4f]" % (result[scoring][min_], result[scoring][max_])
                        ]
                    else:
                        performance_mean[mean] = [
                            "%.3f" % result[scoring][mean] + " " + u"\u00B1"
                            + " " + "%.6f" % result[scoring][std]
                        ]
                        performance_min_max_nice[min_max] = [
                            "[%.3f, %.3f]" % (result[scoring][min_], result[scoring][max_])
                        ]

            print(f'{time} mean nested cross-validation performance scores '
                  f'({scoring} optimized hyperparameters):')
            pprint(performance_mean)
            print('')

            if i == 0:
                performance_mean_df = pd.DataFrame(
                    performance_mean, index=[scoring]
                )
                performance_min_max_nice_df = pd.DataFrame(
                    performance_min_max_nice, index=[scoring]
                )
            else:
                performance_mean_df = performance_mean_df.append(
                    pd.DataFrame(performance_mean, index=[scoring]),
                    verify_integrity=True
                )
                performance_min_max_nice_df = performance_min_max_nice_df.append(
                    pd.DataFrame(performance_min_max_nice, index=[scoring]),
                    verify_integrity=True
                )

        filename = f"{prefix}_collected_mean_ncv_performance_scores"
        fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=False)
        performance_mean_df.to_csv(fn, sep='\t', na_rep='')
        fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=False)
        performance_mean_df.to_excel(fn, na_rep='')
        filename = f"{prefix}_collected_formatted_min_max_ncv_performance_scores"
        fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=False)
        performance_min_max_nice_df.to_csv(fn, sep='\t', na_rep='')
        fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=False)
        performance_min_max_nice_df.to_excel(fn, na_rep='')
        del performance_mean, performance_mean_df
        del performance_min_max_nice, performance_min_max_nice_df
        print(f'{time} end: {ncv.generate_timestamp()}')
        print('++++++++++++++++++++++++++++++++++++++++')

    filename = "collected_key_results"
    fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=False)
    pd.DataFrame(key_results).to_csv(fn, sep='\t', na_rep='nan')
    fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=False)
    pd.DataFrame(key_results).to_excel(fn, na_rep='nan')

    print('End:', ncv.generate_timestamp())
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
            "`np.object` is a deprecated alias for the builtin `object`. "
            "To silence this warning, use `object` by itself. "
            "Doing this will not modify any behavior and is safe."
        ),
        category=DeprecationWarning,
        module=r".*nestedcv"
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
        description=('Function to run repeated nested cross-validation.')
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
        '--Nexp1',
        dest='Nexp1',
        type=int,
        default=1,
        help='Number of inner CV repetitions (for hyper-parameter search).',
    )
    parser.add_argument(
        '--Nexp2',
        dest='Nexp2',
        type=int,
        default=10,
        help='Number of nested CV repetitions.',
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
        help='Kernel combination. Supply SPP, SPR, SRP, or SRR.'
    )
    parser.add_argument(
        '--identifier',
        dest='identifier',
        help=('Prefix to identify the precomputed kernel matrices (stored as .npy files).'
              'E.g. kernel_matrix_rescale.')
    )
    args = parser.parse_args()

    param_grid: Dict[str, List[Any]]
    n_jobs: int
    method = args.method
    if method == 'multitaskSVM':
        from multitaskSVM_config import configurator
        param_grid, estimator, n_jobs = configurator(
            combination=args.combination,
            identifier=args.identifier,
            kernel_dir=args.kernel_dir,
        )
    elif method == 'SVM':
        from SVM_config import estimator, param_grid, n_jobs
    elif method == 'RLR':
        from RLR_config import estimator, param_grid, n_jobs
    elif method == 'RF':
        from RF_config import estimator, param_grid, n_jobs
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
            Nexp1=args.Nexp1,
            Nexp2=args.Nexp2,
        )
    finally:

        warning_file.close()
