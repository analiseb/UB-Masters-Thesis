# Non-Discrimination in Deep Learning: An Application to trustworthy Cardiovascular Disease Diagnosis

This repository is the majority of my thesis in partial fulfillment of the requirements for the degree of Masters of Science in Fundamental Principles of Data Science from the University of Barcelona

## About the Project

The aim of this project is to investigate and develop a fairness framework to mitigate bias in a deep learning multi-classification model for the diagnosis of cardiovascular diseases (CVDs). This model identifies three protected groups, *race, sex, and age,* to evaluate fairness against in the classification of the followings CVDs:

  1. Cardiomyopathies
  2. Myocardial Infarction
  3. Ischemic Heart Disease
  4. Heart Failure
  5. Peripheral Vascular Disease
  6. Cardiac Arrest
  7. Cerebral Infarction
  8. Arrhythmia

## The Dataset

Data for this project was sourced from the [UKBiobank](https://www.ukbiobank.ac.uk/), a largescale biomedical database with information from participants across the UK. Fields used in this project include physical measures, sociodemographic, lifestyle, environmental, early-life, mental health, and bloody assay factors.

The subset of this dataset used throughout the lifespan of the project consistes of approximately 87k records with large representation skews for each of the demographics within the determined protected groups.