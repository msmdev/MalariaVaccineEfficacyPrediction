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
This module contains functions used to preprocess the raw proteome data.

@Author: Bernhard Reuter and Jacqueline Wistuba-Hamprecht

"""

import numpy as np
import pandas as pd


def substract_preimmunization_baseline(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Substract baseline immunity and replace negative intensities.

    The baseline immunity (TimePointOrder = 1) is substracted from the signal intensities measured
    at post-immunization (TimePointOrder = 2, 3 and 4) for each sample `x_i`
    to focus on PfSPZ-CVac induced antibody responses.

    Parameters
    ----------
    X : pd.DataFrame
        Raw proteome data, n x m matrix (n = samples as rows, m = features as columns).
        The function expects that the DataFrame begins with metadata columns
        'Patient', 'group', 'Protection', 'Dose', 'TimePointOrder' (in this exact order)
        followed by antibody signal columns.

    Returns
    -------
    data_minusBS : pd.DataFrame
        Post-immunization data minus baseline immunity.
    """
    data = X.copy(deep=True)

    # baseline: preimmunization data I-1
    data_baseline = data.loc[data["TimePointOrder"] == 1, :].copy()
    data_baseline.reset_index(inplace=True, drop=True)

    # after immunization III 14
    data_III14 = data.loc[data["TimePointOrder"] == 2, :].copy()
    data_III14.reset_index(inplace=True, drop=True)

    # before re-infection C-1
    data_C1 = data.loc[data["TimePointOrder"] == 3, :].copy()
    data_C1.reset_index(inplace=True, drop=True)

    # after re-infection C+ 28
    data_C28 = data.loc[data["TimePointOrder"] == 4, :].copy()
    data_C28.reset_index(inplace=True, drop=True)

    start_indx = data.columns.get_loc("TimePointOrder") + 1

    # substract baseline
    data_III14.iloc[:, start_indx:] = data_III14.iloc[:, start_indx:].sub(
        data_baseline.iloc[:, start_indx:]
    )
    data_C1.iloc[:, start_indx:] = data_C1.iloc[:, start_indx:].sub(
        data_baseline.iloc[:, start_indx:]
    )
    data_C28.iloc[:, start_indx:] = data_C28.iloc[:, start_indx:].sub(
        data_baseline.iloc[:, start_indx:]
    )

    data_minusBS = pd.concat([data_III14, data_C1, data_C28], ignore_index=True)

    return data_minusBS


def normalization(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Normalization.

    The antibody signal intensities `x` are transformed by the inverse hyperbolic sine,
    i.e., `arcsinh(x) = ln(x + sqrt(x^2 + 1))`.

    Parameters
    ----------
    X : pd.DataFrame
        Proteome data, n x m matrix (n = samples as rows, m = features as columns).
        The function expects that the DataFrame begins with metadata columns
        'Patient', 'group', 'Protection', 'Dose', 'TimePointOrder' (in this exact order)
        followed by antibody signal columns.

    Returns
    -------
    data : pd.DataFrame
        Normalized proteome data.
    """
    data = X.copy(deep=True)

    start_indx = data.columns.get_loc("TimePointOrder") + 1

    data[data.columns[start_indx:]] = np.arcsinh(data[data.columns[start_indx:]].to_numpy())

    return data


def sort_proteome_data(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Sorting

    Input data is sorted by time point and patient ID
    to keep same patient over all four time points in order.

    Parameters
    ----------
    X : pd.DataFrame
        Proteome data, n x m pd.DataFrame (n = samples as rows, m = features as columns).
        The function expects that the DataFrame begins with metadata columns
        'Patient', 'group', 'Protection', 'Dose', 'TimePointOrder' (in this exact order)
        followed by antibody signal columns.

    Returns
    -------
    data : pd.DataFrame
        Returns sorted DataFrame
    """
    data = X.copy(deep=True)

    data.sort_values(by=["TimePointOrder", "Patient"], inplace=True)

    data.reset_index(inplace=True, drop=True)

    return data


# TODO: Remove C28
def preprocessing(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Preprocessing of raw proteome data.

    Raw proteome array data is preprocessed in the following way:
    - sorting by patients ID,
    - substraction of baseline immunity,
    - normalization (arcsinh normalization).

    Parameters
    ----------
    X : pd.DataFrame
        Raw proteome data, n x m matrix (n = samples as rows, m = features as columns).
        The function expects that the DataFrame begins with metadata columns
        'Patient', 'group', 'Protection', 'Dose', 'TimePointOrder' (in this exact order)
        followed by antibody signal columns.

    Returns
    -------
    data : pd.DataFrame
        Preprocessed proteome data.
    """
    data = X.copy(deep=True)

    # assert that there are no NaN/missing values
    if not data.notna().all(axis=None):
        raise ValueError("Data has NaN/missing values.")
    if data.columns.to_list()[:5] != ["Patient", "group", "Protection", "Dose", "TimePointOrder"]:
        raise ValueError(
            "Expected first 5 columns to be metadata, i.e., "
            "['Patient', 'group', 'Protection', 'Dose', 'TimePointOrder']."
        )

    # sorting by patients ID
    data = sort_proteome_data(data)

    # substraction of baseline immunity
    data = substract_preimmunization_baseline(data)

    # normalization (arcsinh normalization
    data = normalization(data)

    # assert that there are no infinite values
    if np.any(
        np.isinf(
            data.drop(
                columns=["Patient", "group", "Protection", "Dose", "TimePointOrder"], inplace=False
            ).to_numpy()
        ),
        axis=None,
    ):
        raise ValueError("Data has infinite values.")

    # Move dose column to the rightmost metadata column:
    if data.columns.to_list()[:5] != ["Patient", "group", "Protection", "Dose", "TimePointOrder"]:
        raise ValueError("Wrong order of metadata columns detected before reordering.")
    dose = data["Dose"]
    data.drop(columns=["Dose"], inplace=True)
    data.insert(loc=4, column="Dose", value=dose)

    # assert correct order of metadata
    if data.columns.to_list()[:5] != ["Patient", "group", "Protection", "TimePointOrder", "Dose"]:
        raise ValueError("Wrong order of metadata columns detected after reordering.")

    return data
