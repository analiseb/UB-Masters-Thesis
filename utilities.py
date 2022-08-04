
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler,MinMaxScaler
from tensorflow.keras.optimizers  import Adam, Adagrad, SGD
from sklearn.model_selection import train_test_split
import global_variables as gv
import pandas as pd

from imblearn.over_sampling import ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import keras
import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, Concatenate
from tensorflow.keras.optimizers  import Adam, Adagrad, SGD


def process_features(data, target, norm_method=StandardScaler(), one_hot=True, val=True):
    
    # split data into features and target 
    X = data.iloc[:,:61]
    y = data[target]
    
    # split into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    
    # scale numerical features
    scaler = norm_method # QuantileTransformer(output_distribution='uniform'), StandardScaler(), MaxMinScaler()
    X_train[gv.continuous_cols+gv.numerical_cols]=scaler.fit_transform(X_train[gv.continuous_cols+gv.numerical_cols])
    X_test[gv.continuous_cols+gv.numerical_cols] = scaler.transform(X_test[gv.continuous_cols+gv.numerical_cols])
    
    # get_dummies on nominal categorical features & drop original cols
    if one_hot:
        join = pd.concat([X_train,X_test],axis=0)
        # dummies = pd.get_dummies(join[gv.nominal_cats], columns=gv.nominal_cats, drop_first=True)
        dummies = pd.get_dummies(join[gv.categorical_cols], columns=gv.nominal_cats, drop_first=True)

        X_train[dummies.columns] = dummies.iloc[:len(X_train),:]
        X_test[dummies.columns] = dummies.iloc[len(X_train):,:]
        X_test = X_test.reindex(columns = X_train.columns, fill_value=0)
        del(dummies)

        X_train.drop(gv.nominal_cats,axis=1,inplace=True)
        X_test.drop(gv.nominal_cats,axis=1,inplace=True)

    if val:
      X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.15, random_state=1)

      return X_train, X_val, X_test, y_train, y_val, y_test

    else:
      return X_train, X_test, y_train, y_test


def resample_data(X_train, y_train, method):
    
    if method=='ADASYN':
        X_temp,y_train= ADASYN().fit_resample(X_train,y_train)
        X_train = pd.DataFrame(X_temp, columns=X_train.columns)
        
    elif method=='over':
        # over sample the majority class in train
        oversample = RandomOverSampler(sampling_strategy='minority',random_state=1)
        X_temp, y_train = oversample.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_temp, columns=X_train.columns)

    elif method=='under':
        # under sample the majority class in train
        undersample = RandomUnderSampler(sampling_strategy='majority',random_state=1)
        X_temp, y_train = undersample.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_temp, columns=X_train.columns)
        
    elif method=='partial_under':
        # under sample the majority class in train, ~2:1 neg/pos ratio
        undersample_uneven = RandomUnderSampler(sampling_strategy=0.5,random_state=1)
        X_temp, y_train = undersample_uneven.fit_resample(X_train, y_train)
        X_train = pd.DataFrame(X_temp, columns=X_temp.columns)
        
    return X_train, y_train

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def mlp_model( X_train, X_val, y_val, y_train, X_test, y_test, epochs, batch, activation='tanh',opt=SGD, lr=0.000001):
    
    # define model
    model = Sequential()
    model.add(Dense(1000, activation=activation , input_shape=(X_train.shape[1],)))
    model.add(Dense(500, activation=activation))
    model.add(Dense(500, activation=activation ))
    model.add(Dense(200, activation=activation ))
    model.add(Dense(1, activation=tf.nn.sigmoid))
    
    # compile model
    model.compile(
        loss="binary_crossentropy",
        optimizer=opt(learning_rate=lr),
        metrics=['acc',f1_m,precision_m, recall_m])

    # fit model on train set
    history = model.fit(
        X_train, y_train,
        batch_size=batch,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        validation_data=(X_val, y_val),
    )
    score = model.evaluate(X_test, y_test, verbose=0)
    return history, score