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

The advent of artificial intelligence within medicine continues to make strides infast and dependable disease diagnosis, removing the burden off of our clinicians and medical professionals and allowing them to focus on more nuanced and specialized tasks. While the potential of these automated frameworks to do good is expansive, we must criticize the limitations and inherent challenges that such processes pose in light of higher stakes applications where patients’ lives and livelihoods are directly impacted. At the forefront of these challenges is bias. Because models must learn from historical data and on a predetermined objective function,
they remain vulnerable to perpetuating bias relics and disregarding demographic fairness in the name of increased performance. This project aims to mend these demographic biases and develop a Cardiovascular disease classifier that is fair. We take a three-pronged approach to our development life cycle– building an effective machine learning classification model, in-depth bias evaluation, and a framework of mitigation interventions to result in a model that can effectively classify cardiovascular disease while remaining fair to our identified protected attributes. We investigate and implement a variety of preprocessing and postprocessing mitigation
methods to both gradient boosted and deep learning models, successfully managing the fairness-accuracy tradeoff to ensure the equitable sharing of benefits of AI for all.


## The Dataset

Data for this project was sourced from the [UKBiobank](https://www.ukbiobank.ac.uk/), a largescale biomedical database with information from participants across the UK. Fields used in this project include physical measures, sociodemographic, lifestyle, environmental, early-life, mental health, and bloody assay factors.

The subset of this dataset used throughout the lifespan of the project consists of approximately 500k records with large representation skews for each of the demographics within the determined protected groups.

## Project Structure

#### Data Preprocessing & EDA (prefix - PRE)
* [Data Cleaning](https://github.com/analiseb/UB-Masters-Thesis/blob/main/PRE-data-preprocessing-alternative.ipynb)
* [Imputation](https://github.com/analiseb/UB-Masters-Thesis/blob/main/PRE_missforest_imputation.ipynb)
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
* [Results Summary](https://github.com/analiseb/UB-Masters-Thesis/blob/main/MIT_results_summary.ipynb)

#### Other
* [utility functions (data processing & plotting)](https://github.com/analiseb/UB-Masters-Thesis/blob/main/utilities.py)
* [fairness functions for bias evaluation ](https://github.com/analiseb/UB-Masters-Thesis/blob/main/fairness_helpers.py)
* [variables for data interpretation](https://github.com/analiseb/UB-Masters-Thesis/blob/main/global_variables.py)

## Fairness Metrics

**Average Odds Difference (AOD):** measures the bias by using the false positive rate and the true positive rate
    $$AOD = \frac{1}{2}[(FPR_{D=unprivileged}-FPR_{D=privileged}+TPR_{D=privileged}-TPR_{D=unprivileged}] $$
**Disparate Impact (DI):** compares the proportion of individuals that receive a favorable outcome for two groups, a majority group and a minority group
    $$DI = P(\hat{Y}=1 | A=minority)/P(\hat{Y}=1 | A=majority)$$
    $$\text{where }\hat{Y} \text{ are the model predictions and A is the group of the sensitive attributes}$$
**Equal Opportunity Difference (EOP):** measures the deviation from the equality of opportunity, which means that the same proportion of each population receives the favorable outcome
    $$ EOP = P(\hat{Y}=1 | A=minority)-P(\hat{Y}=1 | A=majority; Y=1)$$
**Statistical Parity Difference (SPD):** measures the difference that the majority and protected classes to receive a favorable outcome
   $$SPD = P(\hat{Y}=1 | A=minority)-P(\hat{Y}=1 | A=majority)$$
**Theil Index:** measures an entropic distance the population is away from the 'ideal' state of everyone having the same outcome
    $$Theil\text{ }Index = \frac{1}{n}\Sigma_{i=1}^{n}\frac{b_{i}}{\mu}ln\frac{b_{i}}{\mu},\text{   where  }b_{i}= \hat{y_{i}}-y_{i}+1$$
