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
@Author: Jacqueline Wistuba-Hamprecht and Bernhard Reuter

"""

import numpy as np
import pandas as pd
from source.utils import select_timepoint, get_parameters
from source.RLR_config import estimator


def featureEvaluationRLR(
        X: pd.DataFrame,
        y: np.ndarray,
        rgscv_results: pd.DataFrame,
        timepoint: str,
):
    """Evaluation of informative features from RLR.

    Parameter
    ---------
    X : pd.DataFrame
        Feature matrix given as pd.DataFrame.
    y : np.ndarray
        Label vector.
    rgscv_results : pd.DataFrame
        DataFrame containing optimal parameters and mean AUROC values
        per time point as found via Repeated Grid-Search CV (RGSCV).
    timepoint : str
        Time point to evaluate informative features for.

    Returns
    -------
    coefs : pd.Dataframe
        Dataframe of non-zero feature importances.

    """

    print(f"RLR parameter combination for best mean AUC at time point {timepoint} :")
    timepoint_results = select_timepoint(
        rgscv_results=pd.DataFrame(rgscv_results),
        timepoint=timepoint)

    params = get_parameters(
        timepoint_results=timepoint_results,
        model='RLR',
    )
    print(f"Parameters: {params}")
    print('')

    # Initialize and fit RLR
    estimator.set_params(**params)
    estimator.fit(X.to_numpy(), y)
    coefs = estimator['logisticregression'].coef_
    if coefs.shape == (1, X.shape[1]):
        coefs = coefs.flatten()
    else:
        raise ValueError(
            "RLR model appears to be a multiclass model, thus, "
            "coef_ array can't be flattened securely."
        )

    sorted_importances_idx = coefs.argsort()
    importances = pd.DataFrame(
        coefs[sorted_importances_idx],
        index=X.columns[sorted_importances_idx],
        columns=['importance'],
    )

    # Extract non-zero coefficients
    print("Number of non-zero importances (weights):", np.count_nonzero(coefs))

    return importances[importances['importance'] != 0]
