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
# source "/Users/schmidtj/opt/anaconda3/etc/profile.d/conda.sh"
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
kernel_dir="${topdir}/data/precomputed_multitask_kernels/filtered/threshold${threshold}"

for method in 'multitaskSVM' 'RF' 'RLR'; do

    for dataset in 'whole' 'selective'; do
        maindir="${topdir}/results/filtered/threshold${threshold}/${method}/${dataset}"
        if [ ! -d "$maindir" ]; then
            { echo "${maindir} doesn't exists."; exit 1; }
        fi

        for timepoint in 'III14' 'C-1' 'C28'; do

            err="runFeatureEval_${timepoint}.err"
            out="runFeatureEval_${timepoint}.out"

            if [ "$method" = 'multitaskSVM' ]; then

                if [ "$dataset" = 'whole' ]; then
                    if [ "$timepoint" = 'III14' ] || [ "$timepoint" = 'C-1' ]; then
                        rgscv_path="${maindir}/RPR/RGSCV/RepeatedGridSearchCV_results.tsv"
                        ana_dir="${maindir}/RPR/featureEvaluation"
                        kernel_identifier='kernel_matrix_RPR'
                    else
                        rgscv_path="${maindir}/RRR/RGSCV/RepeatedGridSearchCV_results.tsv"
                        ana_dir="${maindir}/RRR/featureEvaluation"
                        kernel_identifier='kernel_matrix_RRR'
                    fi
                else
                    if [ "$timepoint" = 'III14' ] || [ "$timepoint" = 'C-1' ]; then
                        rgscv_path="${maindir}/RPR/RGSCV/RepeatedGridSearchCV_results.tsv"
                        ana_dir="${maindir}/RPR/featureEvaluation"
                        kernel_identifier='kernel_matrix_SelectiveSet_RPR'
                    else
                        rgscv_path="${maindir}/RRR/RGSCV/RepeatedGridSearchCV_results.tsv"
                        ana_dir="${maindir}/RRR/featureEvaluation"
                        kernel_identifier='kernel_matrix_SelectiveSet_RRR'
                    fi
                fi

                if [ ! -d "$ana_dir" ]; then
                    mkdir "$ana_dir" || { echo "mkdir ${ana_dir} failed"; exit 1; }
                fi
                cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
                cp "${topdir}/bin/featureEvalRLR.py" . || { echo "cp ${topdir}/bin/featureEvalRLR.py . failed"; exit 1; }
                python -u featureEval.py --data-dir "$data_dir" --data-file-id "preprocessed_${dataset}_data_spearman_filtered_threshold${threshold}" --rgscv-path "$rgscv_path" --out-dir "$ana_dir" --timepoint "$timepoint" --method "$method" --kernel-dir "$kernel_dir" --kernel-identifier "$kernel_identifier" 1> "${out}" 2> "${err}"

            else

                rgscv_path="${maindir}/RGSCV/RepeatedGridSearchCV_results.tsv"
                ana_dir="${maindir}/featureEvaluation"
                if [ ! -d "$ana_dir" ]; then
                    mkdir "$ana_dir" || { echo "mkdir ${ana_dir} failed"; exit 1; }
                fi
                cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
                cp "${topdir}/bin/featureEvalRLR.py" . || { echo "cp ${topdir}/bin/featureEvalRLR.py . failed"; exit 1; }
                python -u featureEval.py --data-dir "$data_dir" --data-file-id "preprocessed_${dataset}_data_spearman_filtered_threshold${threshold}" --rgscv-path "$rgscv_path" --out-dir "$ana_dir" --timepoint "$timepoint" --method "$method" 1> "${out}" 2> "${err}"

            fi

        done

    done

done
