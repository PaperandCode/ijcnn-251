# -*- coding: utf-8 -*-

# We talk about the dimensions of response and predictors. We assume the
# dimensions of response are N X 1 and the dimensions of predictors are N X d.
# N represents the number of observations and d represents the number of
# features.

# %%
import keras
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import regularizers
import keras.backend as K
import xgboost as xgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
def Logistic_regression_construction(response, predictors):

    from sklearn.linear_model import LogisticRegression

    model_logistic = LogisticRegression(solver='liblinear', max_iter=10000, random_state=0)
    model_logistic.fit(predictors, response)
    return model_logistic

def RandomForest_Classifier_construction(response,predictors):
    from sklearn.ensemble import RandomForestClassifier

    model_random_forest_classifier = RandomForestClassifier(n_estimators=500, random_state=0)
    model_random_forest_classifier.fit(predictors, response)
    return model_random_forest_classifier

def MLP_Classifier_construction(response, predictors):
    tf.random.set_seed(1)
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.optimizers import SGD, Adam
    from keras.callbacks import EarlyStopping
    # Here we let X shape = [observations, number of features]
    layer_reg=0.01
    train_X = predictors
    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(response)
    train_y = np_utils.to_categorical(encoded_y)
    model_mlp = Sequential()
    # input layer
    model_mlp.add(Dense(units=100, kernel_initializer='normal',kernel_regularizer=regularizers.l2(layer_reg), activation='relu', input_dim=train_X.shape[1]))
    # hidden layer
    # model_mlp.add(Dense(units=200, kernel_initializer='normal', kernel_regularizer=regularizers.l2(layer_reg),activation='relu'))
    # model_mlp.add(Dense(units=200, kernel_initializer='normal', kernel_regularizer=regularizers.l2(layer_reg),activation='relu'))
    # model_mlp.add(Dense(units=100, kernel_initializer='normal', kernel_regularizer=regularizers.l2(layer_reg),activation='relu'))
    model_mlp.add(Dense(units=100, kernel_initializer='normal', kernel_regularizer=regularizers.l2(layer_reg),activation='relu'))
    # model_mlp.add(Dense(units=5, kernel_initializer='normal',activation='relu'))
    # output layer
    model_mlp.add(Dense(units=2, activation='softmax'))
    # parameters
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=0, mode='auto')
    opt = Adam(lr=0.001, decay=0.005)
    model_mlp.compile(loss='categorical_crossentropy', optimizer=opt)
    model_mlp.summary()
    # fit model_mlp
    model_mlp.fit(train_X, train_y, epochs=500, batch_size=int(1000),  validation_split=0.2, shuffle=False, callbacks=[earlystop])
    return model_mlp
# def MLP_Classifier_construction(response, predictors):
#     from sklearn.neural_network import MLPClassifier
#     clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(50,50), random_state=3, learning_rate_init=0.0001)
#     clf.fit(predictors, response)
#     return clf

def training_construction(response, predictors,Training_name):
    model = 0
    if Training_name=='LR':
        model=Logistic_regression_construction(response, predictors)
        
    elif Training_name=='RF':
        model=RandomForest_Classifier_construction(response, predictors)
        
    elif Training_name=='MLP':
        model=MLP_Classifier_construction(response, predictors)

    return model
