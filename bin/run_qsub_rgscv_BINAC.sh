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

cd .. || { echo "Couldn't cd into parent directory."; exit 1; }
topdir="$PWD"
cd "${topdir}/bin" || { echo "Couldn't cd into ${topdir}/bin directory."; exit 1; }
data_dir="${topdir}/data/proteome_data/correlationFiltering"
combinations=('RPP' 'RPR' 'RRP' 'RRR' 'SPP' 'SPR' 'SRP' 'SRR')

for threshold in '0.85' '0.9' '0.95' '0.98' '1.0'; do

    for method in 'RLR' 'multitaskSVM' 'RF' 'SVM'; do
        maindir="${topdir}/results/threshold${threshold}/${method}"

        for dataset in 'whole' 'selective'; do
            kernel_dir="${topdir}/data/precomputed_multitask_kernels/threshold${threshold}/${dataset}"
            if [ ! -d "$kernel_dir" ]; then
                { echo "${kernel_dir} doesn't exist"; exit 1; }
            fi

            if [ "$method" = 'multitaskSVM' ]; then

                identifier='kernel_matrix'

                for combination in "${combinations[@]}"; do
                    ana_dir="${maindir}/${dataset}/${combination}"
                    if [ ! -d "$ana_dir" ]; then
                        mkdir -p "$ana_dir" || { echo "mkdir ${ana_dir} failed"; exit 1; }
                    fi
                    cd "$ana_dir" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
                    cp "${topdir}/bin/rgscv.py" . || { echo "cp ${topdir}/bin/rgscv.py . failed"; exit 1; }
                    cp "${topdir}/source/${method}_config.py" . || { echo "cp ${topdir}/source/${method}_config.py . failed"; exit 1; }
                done
                jobname="${threshold}${method}${dataset}"
                ana_dir="${maindir}/${dataset}"
                cd "$ana_dir" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
                cp "${topdir}/bin/run_rgscv_BINAC.sh" . || { echo "cp ${topdir}/bin/run_rgscv_BINAC.sh . failed"; exit 1; }
                qsub -v "ANA_DIR"="${ana_dir}","DATA_DIR"="${data_dir}","DATA_FILE_ID"="preprocessed_${dataset}_data_spearman_filtered_threshold${threshold}","METHOD"="${method}","NJOBS"=28,"KERNEL_DIR"="$kernel_dir","IDENTIFIER"="$identifier" -N "$jobname" -q short -l walltime=48:00:00,mem=125gb,nodes=1:ppn=28 "run_rgscv_BINAC.sh"

            else

                jobname="${threshold}${method}${dataset}"
                ana_dir="${maindir}/${dataset}"
                if [ ! -d "$ana_dir" ]; then
                    mkdir -p "$ana_dir" || { echo "mkdir ${ana_dir} failed"; exit 1; }
                fi
                cd "$ana_dir" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
                cp "${topdir}/bin/rgscv.py" . || { echo "cp ${topdir}/bin/rgscv.py . failed"; exit 1; }
                cp "${topdir}/bin/run_rgscv_BINAC.sh" . || { echo "cp ${topdir}/bin/run_rgscv_BINAC.sh . failed"; exit 1; }
                cp "${topdir}/source/${method}_config.py" . || { echo "cp ${topdir}/source/${method}_config.py . failed"; exit 1; }
                if [ "$method" = 'RLR' ]; then
                    qsub -v "ANA_DIR"="${ana_dir}","DATA_DIR"="${data_dir}","DATA_FILE_ID"="preprocessed_${dataset}_data_spearman_filtered_threshold${threshold}","METHOD"="${method}","NJOBS"=28 -N "$jobname" -q short -l walltime=48:00:00,mem=125gb,nodes=1:ppn=28 "run_rgscv_BINAC.sh"
                else
		    qsub -v "ANA_DIR"="${ana_dir}","DATA_DIR"="${data_dir}","DATA_FILE_ID"="preprocessed_${dataset}_data_spearman_filtered_threshold${threshold}","METHOD"="${method}","NJOBS"=4 -N "$jobname" -q short -l walltime=48:00:00,mem=17gb,nodes=1:ppn=4 "run_rgscv_BINAC.sh"
                fi

            fi

        done

    done

done
