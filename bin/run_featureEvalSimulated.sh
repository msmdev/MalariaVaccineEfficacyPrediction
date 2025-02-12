#!/bin/bash

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

# @Author: Bernhard Reuter

# # Replace the following line by the path of your own conda.sh file:
# source "/Users/schmidtj/opt/anaconda3/etc/profile.d/conda.sh"
# # This is intended to run in the malaria_env conda environment:
# conda activate malaria_env

# This is intended to run in the bin folder of the MalariaVaccineEfficacyPrediction package.
# The MalariaVaccineEfficacyPrediction package should be situated in the users home directory.
topdir="${HOME}/MalariaVaccineEfficacyPrediction"
if [ ! -d "$topdir" ]; then
    { echo "${topdir} doesn't exists."; exit 1; }
fi
data_dir="${topdir}/data/simulated_data"
data_file="simulated_data.csv"
ana_dir="${topdir}/results/simulated"
if [ ! -d "$ana_dir" ]; then
    mkdir "$ana_dir" || { echo "mkdir ${ana_dir} failed"; exit 1; }
fi

err="run_ESPY_SHAP_simulated.err"
out="run_ESPY_SHAP_simulated.out"
        
cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
cp "${topdir}/bin/featureEvalSimulated.py" . || { echo "cp ${topdir}/bin/featureEvalSimulated.py . failed"; exit 1; }
python featureEvalSimulated.py --data-dir "$data_dir" --data-file "$data_file" --out-dir "$ana_dir" 1> "${out}" 2> "${err}"
