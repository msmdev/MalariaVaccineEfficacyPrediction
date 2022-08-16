#!/bin/bash

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

# @Author: Bernhard Reuter

# # Replace the following line by the path of your own conda.sh file:
# source "/home/breuter/anaconda3/etc/profile.d/conda.sh"
# # This is intended to run in the malaria_env conda environment:
# conda activate malaria_env

# This is intended to run in the bin folder of the MalariaVaccineEfficacyPrediction package.
# The MalariaVaccineEfficacyPrediction package should be situated in the users home directory.
threshold='0.95'
topdir="${HOME}/MalariaVaccineEfficacyPrediction"
if [ ! -d "$topdir" ]; then
    { echo "${topdir} doesn't exists."; exit 1; }
fi
data_dir="${topdir}/data/proteome_data/correlationFiltering"

for method in 'RF' 'RLR'; do
    maindir="${topdir}/results/filtered/threshold${threshold}/${method}"
    if [ ! -d "$maindir" ]; then
        mkdir "$maindir" || { echo "mkdir ${maindir} failed"; exit 1; }
    fi

    for dataset in 'whole' 'selective'; do

        err="runRGSCV.err"
        out="runRGSCV.out"
        ana_dir="${maindir}/${dataset}"
        mkdir "${ana_dir}"
        cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
        cp "${topdir}/bin/rgscv.py" . || { echo "cp ${topdir}/bin/rgscv.py . failed"; exit 1; }
        cp "${topdir}/bin/${method}_config.py" . || { echo "cp ${topdir}/bin/${method}_config.py . failed"; exit 1; }
        python -u rgscv.py --analysis-dir "${ana_dir}" --data-dir "${data_dir}" --data-file-id "preprocessed_${dataset}_data_spearman_filtered_threshold${threshold}" --method "${method}" 1> "${out}" 2> "${err}"
    done

done
