# Non-Discrimination in AI: An Application to Fair Cardiovascular Disease Diagnosis


## About the Project

The aim of this project is to investigate and develop a fairness framework to mitigate bias in deep learning multi-label classification and a binary-classification models for the diagnosis of cardiovascular diseases (CVDs). This model identifies three protected groups, *race, sex, and age,* to evaluate fairness against in the classification of the followings CVDs:

  1. Cardiomyopathies
  2. Myocardial Infarction
  3. Ischemic Heart Disease
  4. Heart Failure
  5. Peripheral Vascular Disease
  6. Cardiac Arrest
  7. Cerebral Infarction
  8. Arrhythmia

## Abstract

The advent of artificial intelligence within medicine continues to make strides in
fast and dependable disease diagnosis, removing the burden off of our clinicians
and medical professionals and allowing them to focus on more nuanced and spe-
cialized tasks. While the potential of these automated frameworks to do good
is expansive, we must criticize the limitations and inherent challenges that such
processes pose in light of higher stakes applications where patients’ lives and liveli-
hoods are directly impacted. At the forefront of these challenges is bias. Because
models must learn from historical data and on a predetermined objective function,
they remain vulnerable to perpetuating bias relics and disregarding demographic
fairness in the name of increased performance. This project aims to mend these
demographic biases and develop a Cardiovascular disease classifier that is fair. We
take a three-pronged approach to our development life cycle– building an effective
machine learning classification model, in-depth bias evaluation, and a framework
of mitigation interventions to result in a model that can effectively classify car-
diovascular disease while remaining fair to our identified protected attributes. We
investigate and implement a variety of preprocessing and postprocessing mitigation
methods to both gradient boosted and deep learning models, successfully manag-
ing the fairness-accuracy tradeoff to ensure the equitable sharing of benefits of AI
for all.
ii


## The Dataset

Data for this project was sourced from the [UKBiobank](https://www.ukbiobank.ac.uk/), a largescale biomedical database with information from participants across the UK. Fields used in this project include physical measures, sociodemographic, lifestyle, environmental, early-life, mental health, and bloody assay factors.

The subset of this dataset used throughout the lifespan of the project consists of approximately 500k records with large representation skews for each of the demographics within the determined protected groups.

## Project Structure

#### Data Preprocessing & EDA (prefix - PRE)
* [Data Cleaning](https://github.com/analiseb/UB-Masters-Thesis/blob/main/PRE-data-preprocessing-alternative.ipynb)
* [Imputation](https://github.com/analiseb/UB-Masters-Thesis/blob/main/PRE_normalization_imputation-alternative.ipynb)
* [Exporatory Data Analysis](https://github.com/analiseb/UB-Masters-Thesis/blob/main/PRE_eda.ipynb)

#### Model Development (prefix - MODEL)

* [XGBoost](https://github.com/analiseb/UB-Masters-Thesis/blob/main/MODEL_Baseline_XGBoost-alternative.ipynb)
* [MLP](https://github.com/analiseb/UB-Masters-Thesis/blob/main/MODEL_mlp-alternative.ipynb)
* [TabNet](https://github.com/analiseb/UB-Masters-Thesis/blob/main/MODEL_tabnet_pytorch-alternative.ipynb)
* [Performance Results](https://github.com/analiseb/UB-Masters-Thesis/blob/main/MODEL_performance_thresholds.ipynb)
* [Saved Models](https://github.com/analiseb/UB-Masters-Thesis/tree/main/saved_models)

#### Bias Evaluation & Mitigation (prefix - BEVAL/MIT)

* [Bias Evaluation](https://github.com/analiseb/UB-Masters-Thesis/blob/main/BEVAL_bias_analysis-alternative.ipynb)
* [Bias Mitigatation - Preprocessing](https://github.com/analiseb/UB-Masters-Thesis/blob/main/MIT_preprocessing_mitigation--XGBoost_methods.ipynb)
* [Bias Mitigation - Postprocessing](https://github.com/analiseb/UB-Masters-Thesis/blob/main/MIT_postprocessing_mitigation--DL_methods_recovered.ipynb)
* [Results Summary](https://github.com/analiseb/UB-Masters-Thesis/blob/main/RESULTS%20.ipynb)

#### Other
* [utility functions (data processing & plotting)](https://github.com/analiseb/UB-Masters-Thesis/blob/main/utilities.py)
* [fairness functions for bias evaluation ](https://github.com/analiseb/UB-Masters-Thesis/blob/main/fairness_helpers.py)
* [variables for data interpretation](https://github.com/analiseb/UB-Masters-Thesis/blob/main/global_variables.py)
