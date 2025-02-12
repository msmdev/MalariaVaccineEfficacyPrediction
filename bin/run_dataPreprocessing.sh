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
data_dir="${topdir}/data/proteome_data"
ana_dir="${data_dir}/correlationFiltering"
if [ ! -d "$ana_dir" ]; then
    mkdir "$ana_dir" || { echo "mkdir ${ana_dir} failed"; exit 1; }
fi
method='spearman'

for dataset in 'whole' 'selective'; do

    cd "${data_dir}" || { echo "Couldn't cd into ${data_dir}"; exit 1; }
    cp "${topdir}/bin/dataPreprocessing.py" . || { echo "cp ${topdir}/bin/dataPreprocessing.py . failed"; exit 1; }

    out="run_dataPreprocessing_${dataset}.out"
    err="run_dataPreprocessing_${dataset}.err"

    python -u dataPreprocessing.py --data-dir "$data_dir" --data-file "${dataset}_proteomearray_rawdata.csv" --out-name "preprocessed_${dataset}_data" 1> "${out}" 2> "${err}"

    cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
    cp "${topdir}/bin/groupCorrelatedFeatures.py" . || { echo "cp ${topdir}/bin/groupCorrelatedFeatures.py . failed"; exit 1; }

    for threshold in '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0'; do

        err="run_groupCorrelatedFeatures_${dataset}_${method}_threshold${threshold}.err"
        out="run_groupCorrelatedFeatures_${dataset}_${method}_threshold${threshold}.out"
        python -u groupCorrelatedFeatures.py --data-dir "${data_dir}" --data-file-id "preprocessed_${dataset}_data" --out-dir "$ana_dir" --correlation_threshold "$threshold" --correlation_method "$method" 1> "${out}" 2> "${err}"

    done

done
