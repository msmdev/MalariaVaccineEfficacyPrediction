# Machine Learning for Prediction of Malaria Vaccine Efficacy based on Antibody Profiles

This repository contains code and data used in the course of our study to predict malaria vaccine efficacy based on antibody profiles.
Proteome microarrays representing about 91% of the *Plasmodium falciparum* (Pf) proteome have been used to identify Pf-specific antibody profiles of malaria-naive volunteers during immunization with attenuated Pf sporozoites (PfSPZ) under chloroquine chemoprophylaxis, using the PfSPZ Chemoprophylaxis Vaccine (PfSPZ-CVac).
We reused this earlier published [data](https://www.nature.com/articles/nature21060) and compared the ability of three supervised machine learning methods to identify predictive antibody profiles after immunization and before controlled human malaria infection (CHMI).<br>
We adapted a multitask SVM approach to analyze time-dependent Pf-induced antibody profiles in a single prediction model and developed a new explainabilty method, named ESPY, to interpret the impact of Pf-specific antigens based on this non-linear model.

## Layout of the study

The study is structured in three main parts.
The first part is a performance assessment of the new **multitask SVM** approach in comparison to state-of-the-art methods.
In the second part, the new **ESPY (fEature diStance exPlainabilitY)** method is used to quantify informative features from the non-linear multitask SVM model in comparison to state-of-the art methods. The the third and last part we show the ESPY values on simulated data and compare those with the SHAP values from [Lundberg et al.](https://github.com/slundberg/shap).
All executable code can be found in the [bin](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/bin) folder.

## Requirements
**All code should always be run in a customized conda environment.**
Please install the conda environment via
```bash
conda env create -f malaria_environment.yml
```
and activate it with
```bash
conda activate malaria_env
```
before executing any code.

## Installation
You can install the package using pip (note: you must be inside the repository):
```bash
pip install .
```
Please note: The code is designed and tested for Linux or MacOS systems. In the following, we will assume that you have cloned the git repository into your home directory. If you want to locate the repository elsewhere in your filesystem, you will need to change the bash run scripts in [bin](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/bin) accordingly.

### Developer mode
Instead you can install in "develop" or "editable" mode using pip:
```bash
pip install --editable .
```
This puts a link into the python installation to the code, such, that your package is installed but any changes to the source code will immediately take effect.
At the same time, all your client code can import the package the usual way.

## Structure of proteome data
The following data table gives an overview of the data structure of the proteome raw data: <br>
(file format: .csv file)

| Patient     | group | Protection | Dose | TimePointOrder | feature 1 | ... | feature n |
|-------------|-------|------------|------|----------------|-----------|-----|-----------|
| ID-01 III14 | 01    | 1          | 1    | 2              | 300.5     | ... | ...       |
| ID-01 C1    | 01    | 1          | 1    | 3              | 320.0     | ... | ...       |
| ID-01 C28   | 01    | 1          | 1    | 4              | 400.0     | ... | ...       |
| ID-02 III14 | 02    | 0          | 0    | 2              | 1000.5    | ... | ...       |
| ID-02 C1    | 02    | 0          | 0    | 3              | 1200.0    | ... | ...       |
| ID-02 C28   | 02    | 0          | 0    | 4              | 1100.5    | ... | ...       |
| ID-03 III14 | 03    | 1          | 4    | 2              | 600.3     | ... | ...       |
| ID-03 C1    | 03    | 1          | 4    | 3              | 400.3     | ... | ...       |
| ID-03 C28   | 03    | 1          | 4    | 4              | 200.0     | ... | ...       |
| ......      | ....  | ..         | ..   | ..             | ...       | ... | ...       |


The column **group** represents the ID of the patient.
In column **Dose** the integer values represent the respective PfSPZ dose:
{0 : placebo; 1 : 3.2 * 10^3 PfSPZ; 2 : 1.28 * 10^4 PfSPZ; 4 : 5.12 * 10^4 PfSPZ}
The integer values of the column **TimePointOrder** represent the day of blood serum extraction:
{2 : III14 post-immunization; 3 : C-1 pre-CHMI; 4 : C+28 post-CHMI}.

You can find the raw data of the proteome-microarray-based antibody reactivity profile for both the whole set of 7,455 Pf-specific antigen fragments and a set of 1.194 Pf-specific cell surface antigen fragments (selected from the whole set of Pf-specific fragments of the proteome microarray) in [data/proteome_data](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/data/proteome_data).
Both files contain the raw data that has to be baseline and arcsine transformed using the dataPreprocessing.py script. Antibody responses after CHMI (C+28) were excluded from our malaria vaccine efficacy prediction analysis. This results form the fact that controls also underwent CHMI. Thus, after CHMI, there are no samples for the unprotected class anymore and, therefore, applying binary classification models will not be feasible.

### Structure of simulated data
The simulated dataset consists of 500 samples and 1000 features, where 15 features are defined as informative features and the remaining ones as uninformative features. You can find the simulated data in [data/simulated_data](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/data/simulated_data).

### Preprocessing of the Data
We set up the run_dataPreprocessing.sh script to directly execute the data preprocessing on the [whole raw proteome data](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/data/proteome_data/whole_proteomearray_rawdata.csv) and the [pre-selected cell-surface raw proteome data](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/data/proteome_data/selective_proteomearray_rawdata.csv). The processed data is written to [data/proteome_data](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/data/proteome_data)
Additionally, the run_dataPreprocessing.sh scripts executes the grouping of strongly covarying features for a Pearson correlation coefficient from 0.1 to 1.0 in 0.1 steps on the whole proteome data and the pre-selected cell-surface proteome data. The filtered groups per Pearson correlation coefficient are written to [data/proteome_data/correlationFiltering](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/data/proteome_data/correlationFiltering).
The run_dataPreprocessing.sh script has to be executed in the [bin](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/bin) folder.
```bash
# run data preprocessing for whole and pre-selected cell-surface raw proteome data
./run_dataPreprocessing.sh
```


## Prediction performance assesment of the multitask SVM in comparison to state-of-the-art methods
Here we give a short introduction how to run the 10-times repeated nested stratified 5-fold cross-validation for the multitask SVM and three state-of-the-art-methods, namely elastic net regularized logistic regression (RLR), single-task SVM with a RBF kernel and random forest.
However, we <span style="color:orange;">strongly</span> advise to not run the performance assessment, since it is computationally demanding and time-consuming. If you really need to execute it, don't use your laptop but a high performance computing system.<br>

### Requirements
To run the performance assessment you need to install the **NestedCV** package inside of your conda malaria_env environment.
The NestedCV package implements a method to perform repeated nested stratified cross-validation for any estimator that implements the scikit-learn estimator interface.
The NestedCV package can be downloaded from [here](https://github.com/msmdev/NestedCV).
First activate the conda malaria_env environment:
```bash
conda activate malaria_env
```
Afterwards navigate into the NestedCV repository and install it via pip:
```bash
pip install .
```

### Arguments
If you are entirely sure that you want to run the performance assessment (Be sure to know what you are doing!), you can execute it (in [bin](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/bin)) as follows:

```bash
# run performance assessment for multitask-SVM, RLR, single-task-SVM and RF
./run_rncv.sh

```
The output files for the performance assessment of the multitask-SVM and the state-of-the-art methods will be generated automatically and are saved to a 'RNCV' folder, which is stored [here](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/results) in pre-defined subfolders for each Pearson correlation coefficient and for the whole and pre-selective data. 

## Tuning models for feature evaluation
Here we give a short introduction how to optimize the best parameters running a 10-time repeated nested grid search for the multitask-SVM before evaluating informative features.  However, we <span style="color:orange;">strongly</span> advise to not run the performance assessment, since it is computationally demanding and time-consuming. If you really need to execute it, don't use your laptop but a high performance computing system.<br>

### Requirements
To run the performance assessment you need to install the **NestedCV** package inside of your conda malaria_env environment.
The NestedCV package implements a method to perform repeated nested stratified cross-validation for any estimator that implements the scikit-learn estimator interface.
The NestedCV package can be downloaded from [here](https://github.com/msmdev/NestedCV).
First activate the conda malaria_env environment:
```bash
conda activate malaria_env
```
Afterwards navigate into the NestedCV repository and install it via pip:
```bash
pip install .
```

### Arguments
If you are entirely sure that you want to run the grid search for feature evaluation (Be sure to know what you are doing!), you can execute it (in [bin](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/bin)) as follows:

```bash
# run parameter tuning for multitask-SVM
./run_rgscv.sh

```
The output files for the parameter optimization of the multitask-SVM will be generated automatically and are saved to a 'RGSCV' folder, which is stored [here](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/results) in pre-defined subfolders for each Pearson correlation coefficient and for the whole and pre-selective data.

## Apply ESPY
Here we show how to apply the ESPY method to the simulated data and the proteome data sets.

### On simulated data
Here we show how to evaluate informative features using our ESPY method and the [SHAP](https://github.com/slundberg/shap) framework SHapley Additive exPlanations) framework from Lundberg et al..
#### Arguments
To run ESPY on simulated data, you can execute the run_featureEvalSimulated.sh script in the [bin](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/bin) folder as
follows:

```bash
# run ESPY on simulated data
./run_featureEvalSimulated.sh
```
The output files for the informative feature evaluation using ESPY and the SHAP framework will be generated automatically. The output files are stored [here](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/results/simulated)

### On proteome data
To run ESPY on the proteome data sets, you can easily run the shell script run_featureEval.sh in the [bin](https://github.com/msmdev/MalariaVaccineEfficacyPrediction/tree/main/bin) folder via
```bash
./run_featureEval.sh
```
and the output files for the whole and the selective proteome array per time point will be generated automatically.
The output files are stored [here](https://github.com/jacqui20/MalariaVaccineEfficacyPrediction/tree/main/results/threshold0.8/multitaskSVM) in pre-defined subfolders for the whole and selective data and the kernel combination 'RRR'. <br>
