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
maindir="${topdir}/data/precomputed_multitask_kernels"
if [ ! -d "$maindir" ]; then
    mkdir -p "$maindir" || { echo "mkdir ${maindir} failed"; exit 1; }
fi
identifier='kernel_matrix'

for threshold in '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0'; do

    for dataset in 'whole' 'selective'; do

        out="${maindir}/run_precomputeGramMatrices_${dataset}_threshold${threshold}.out"
        err="${maindir}/run_precomputeGramMatrices_${dataset}_threshold${threshold}.err"

        python -u precomputeGramMatrices.py --data-file "${topdir}/data/proteome_data/correlationFiltering/preprocessed_${dataset}_data_spearman_filtered_threshold${threshold}.csv" --out-dir "${maindir}/threshold${threshold}/${dataset}" --identifier "$identifier" 1> "${out}" 2> "${err}"

    done

done
