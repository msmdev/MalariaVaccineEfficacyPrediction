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
topdir="${HOME}/MalariaVaccineEfficacyPrediction"
if [ ! -d "$topdir" ]; then
    { echo "${topdir} doesn't exists."; exit 1; }
fi
data_dir="${topdir}/data/proteome_data/correlationFiltering"
combinations=('RPP' 'RPR' 'RRP' 'RRR')

for threshold in '0.1' '0.2' '0.3' '0.4' '0.5' '0.6' '0.7' '0.8' '0.9' '1.0'; do

    for method in 'multitaskSVM' 'RF' 'RLR' 'SVM'; do
        maindir="${topdir}/results/threshold${threshold}/${method}"

        for dataset in 'whole' 'selective'; do
            kernel_dir="${topdir}/data/precomputed_multitask_kernels/threshold${threshold}/${dataset}"

            if [ ! -d "$kernel_dir" ]; then
                { echo "${kernel_dir} doesn't exist"; exit 1; }
            fi

            if [ "$method" = 'multitaskSVM' ]; then

                identifier='kernel_matrix'

                for combination in "${combinations[@]}"; do
                    err="runRGSCV.err"
                    out="runRGSCV.out"
                    ana_dir="${maindir}/${dataset}/${combination}"
                    if [ ! -d "$ana_dir" ]; then
                        mkdir -p "$ana_dir" || { echo "mkdir ${ana_dir} failed"; exit 1; }
                    fi
                    cd "$ana_dir" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
                    cp "${topdir}/bin/rgscv.py" . || { echo "cp ${topdir}/bin/rgscv.py . failed"; exit 1; }
                    cp "${topdir}/source/${method}_config.py" . || { echo "cp ${topdir}/source/${method}_config.py . failed"; exit 1; }
                    python -u rgscv.py --analysis-dir "$ana_dir" --data-file "${data_dir}/preprocessed_${dataset}_data_spearman_filtered_threshold${threshold}.csv" --method "$method" --njobs=-1 --kernel-dir "$kernel_dir" --combination "$combination" --kernel-identifier "$identifier" 1> "${out}" 2> "${err}"
                done

            else

                err="runRGSCV.err"
                out="runRGSCV.out"
                ana_dir="${maindir}/${dataset}"
                if [ ! -d "$ana_dir" ]; then
                    mkdir -p "$ana_dir" || { echo "mkdir ${ana_dir} failed"; exit 1; }
                fi
                cd "$ana_dir" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
                cp "${topdir}/bin/rgscv.py" . || { echo "cp ${topdir}/bin/rgscv.py . failed"; exit 1; }
                cp "${topdir}/source/${method}_config.py" . || { echo "cp ${topdir}/source/${method}_config.py . failed"; exit 1; }
                python -u rgscv.py --analysis-dir "${ana_dir}" --data-file "${data_dir}/preprocessed_${dataset}_data_spearman_filtered_threshold${threshold}.csv" --method "${method}" --njobs=-1 1> "${out}" 2> "${err}"
            fi

        done

    done

done
