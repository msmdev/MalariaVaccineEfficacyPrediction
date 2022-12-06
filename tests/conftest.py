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

import pytest
import numpy as np
from numpy.testing import assert_allclose as assert_allclose_np
from typing import List, Union


def assert_allclose(
    actual: Union[np.ndarray, List[Union[float, int]]],
    desired: Union[np.ndarray, List[Union[float, int]]],
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    err_msg: str = '',
    verbose: bool = True,
) -> None:
    return assert_allclose_np(actual, desired, rtol=rtol, atol=atol, err_msg=err_msg, verbose=True)


@pytest.fixture(scope="session")
def dummy():
    return None
