import pandas as pd

data_link ='https://github.com/analiseb/UB-Masters-Thesis/blob/main/data/CVD_data.csv?raw=true'

outcomes = ['outcome_myocardial_infarction','outcome_cardiomyopathies','outcome_ischemic_heart_disease','outcome_heart_failure','outcome_peripheral_vascular_disease','outcome_cardiac_arrest','outcome_cerebral_infarction','outcome_arrhythmia']
prioritized_outcomes = ['outcome_myocardial_infarction', 'outcome_ischemic_heart_disease', 'outcome_heart_failure','outcome_peripheral_vascular_disease']

df = pd.read_csv(data_link)
df.drop('Unnamed: 0', axis=1, inplace=True)

# classifying features by datatype for appropriate use in model
continuous_cols = df.iloc[:,:18].columns.to_list()
numerical_cols = df.iloc[:,18:18+13].columns.to_list()
categorical_cols = df.iloc[:,18+13:18+13+30].columns.to_list() # ordinal encoded
nominal_cats = ['1428-0.0','20117-0.0','2100-0.0','2654-0.0','21000-0.0','1538-0.0','31-0.0','6138-0.0','2090-0.0','1508-0.0','6142-0.0','1468-0.0','1239-0.0','1448-0.0','hypertension']

race_mapping = {1001.0:'British',
        1003.0:'Any other white background',
        1002.0:'Irish',
        3001.0:'Indian',
        6.0:'Other ethnic group',
        4001.0:'Caribbean',
        3002.0 :'Pakistani',
        4002.0 :'African',
        3004.0  :'Any other Asian background',
        1.0:'White',
        5.0:'Chinese',
        2004.0:'Any other mixed background',
        2003.0:'White and Asian',
        2001.0:'White and Black Caribbean',
        3003.0:'Bangladeshi',
        2002.0:'White and Black African',
        2.0:'Mixed',
        3.0:'Asian or Asian British',
        4003.0:'Any other Black background',
        4.0:'Black or Black British'    
}