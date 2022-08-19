import pandas as pd

data_link ='https://github.com/analiseb/UB-Masters-Thesis/blob/main/data/CVD_data.csv?raw=true'
pre_data_link = 'https://github.com/analiseb/UB-Masters-Thesis/blob/main/data/preprocessed-ukb46359.csv?raw=true'
tabnet_data = 'https://github.com/analiseb/UB-Masters-Thesis/blob/main/data/tabnet_preprocessed.csv?raw=true'

outcomes = ['outcome_myocardial_infarction','outcome_cardiomyopathies','outcome_ischemic_heart_disease','outcome_heart_failure','outcome_peripheral_vascular_disease','outcome_cardiac_arrest','outcome_cerebral_infarction','outcome_arrhythmia']
prioritized_outcomes = ['outcome_myocardial_infarction', 'outcome_ischemic_heart_disease', 'outcome_heart_failure','outcome_peripheral_vascular_disease']

df = pd.read_csv(data_link)
df.drop('Unnamed: 0', axis=1, inplace=True)

# classifying features by datatype for appropriate use in model
continuous_cols = df.iloc[:,:18].columns.to_list()
numerical_cols = df.iloc[:,18:18+13].columns.to_list()
categorical_cols = df.iloc[:,18+13:18+13+30].columns.to_list() # ordinal encoded
nominal_cats = ['1428-0.0','20117-0.0','2100-0.0','2654-0.0','21000-0.0','1538-0.0','31-0.0','6138-0.0','2090-0.0','1508-0.0','6142-0.0','1468-0.0','1239-0.0','1448-0.0','hypertension']

protected_attributes = ['31-0.0',  '21000-0.0', '21003-0.0'] # age 21003-0.0

# mappings
sex_mapping = {1.0:'Male', 
       0.0:'Female'
}

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

input_mapping = {'30850-0.0': 'testosterone',
                '30780-0.0': 'LDL',
                '30690-0.0':'cholesterol',
                '30790-0.0':'LP-a',
                '23101-0.0':'whole body fat-free mass',
                '23099-0.0': 'body fat percentage',
                '48-0.0': 'waist circumference',
                '23100-0.0':'whole body fat mass',
                '30710-0.0': 'CRP',
                '30760-0.0': 'HDL',
                '30640-0.0': 'APOB',
                '30750-0.0': 'HbA1c',
                '49-0.0': 'hip circumference',
                '30770-0.0': 'IGF-1',
                '30740-0.0':'glucose',
                '30630-0.0': 'vascular',
                '30870-0.0': 'triglyceride',
                '21001-0.0': 'BMI',
                '1488-0.0': 'tea intake',
                '4079-0.0': 'diastolic blood pressure',
                '1299-0.0': 'raw veg intake',
                '21003-0.0': 'age',
                '1160-0.0': 'sleep duration',
                '1438-0.0':'bread intake',
                '4080-0.0':'systolic blood pressure',
                '1458-0.0':'cereal intake',
                '1528-0.0':'water intake',
                '1319-0.0':'dried fruit intake',
                '845-0.0':'age completed education',
                '1289-0.0': 'cooked veg intake',
                '1309-0.0': 'fresh fruit intake',
                '1418-0.0': 'milk type',
                '1329-0.0': 'oily fish intake',
                '1220-0.0':'narcolepsy',
                '1428-0.0':'spread type',
                '1249-0.0':'past tobacco smoking',
                '1349-0.0': 'processed meat intake',
                '1369-0.0': 'beef intake',
                '20117-0.0': 'alcohol drinker status',
                '2100-0.0':'psychologist anxiety or depression',
                '2654-0.0':'non-butter spread',
                '1339-0.0': 'non-oily fish intake',
                '21000-0.0': 'ethnic background',
                '2050-0.0':'freq depressed mood past 2 weeks',
                '1408-0.0': 'cheese intake',
                '1200-0.0':'insomnia',
                '1538-0.0':'major dietary changes in the last 5 years',
                '31-0.0':'Sex',
                '6138-0.0':'Qualifications',
                '1359-0.0':'poultry intake',
                '1389-0.0':'pork intake',
                '1478-0.0':'salt added to food',
                '2090-0.0':'doctor anxiety or depression',
                '1508-0.0':'coffee type',
                '1379-0.0': 'lamb intake',
                '6142-0.0': 'employment status',
                '1468-0.0': 'cereal type',
                '1548-0.0':'variation in diet',
                '1239-0.0':'current tobacco smoking',
                '1448-0.0': 'bread type',
                'hypertension': 'hypertension'
}
        

# binary mapping of sex attributes

binary_sex = {'Male':1,
              'Female':0
    
}

# remmap race to white/non-white (1:white, 0:non-white)

binary_race =  {'British':1,
        'Any other white background':1,
        'Irish':1,
        'Indian':0,
        'Other ethnic group':0,
        'Caribbean':0,
        'Pakistani':0,
        'African':0,
        'Any other Asian background':0,
        'White':1,
        'Chinese':0,
        'Any other mixed background':0,
        'White and Asian':0,
        'White and Black Caribbean':0,
        'Bangladeshi':0,
        'White and Black African':0,
        'Mixed':0,
        'Asian or Asian British':0,
        'Any other Black background':0,
        'Black or Black British': 0   
}

# white: 0
# black: 1
# asian: 2 
# mixed or other:3

race_groupings_encoded = {'white':0,
                          'black':1,
                          'asian':2,
                          'mixed or other':3

}
alternate_race_groupings =  {'British': 'white',
        'Any other white background':'white',
        'Irish':'white',
        'Indian':'asian',
        'Other ethnic group':'mixed or other',
        'Caribbean':'mixed or other',
        'Pakistani':'asian',
        'African':'black',
        'Any other Asian background':'asian',
        'White':'white',
        'Chinese':'asian',
        'Any other mixed background':'mixed or other',
        'White and Asian':'mixed or other',
        'White and Black Caribbean':'mixed or other',
        'Bangladeshi':'asian',
        'White and Black African':'mixed or other',
        'Mixed':'mixed or other',
        'Asian or Asian British':'asian',
        'Any other Black background':'black',
        'Black or Black British': 'black'   
}

privileged_groups = { 'sex': 1,
                        'race':1
}

unprivileged_groups = { 'sex': 0,
                        'race':0
}