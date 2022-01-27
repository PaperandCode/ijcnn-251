# -*- coding: utf-8 -*-

# We talk about the dimensions of response and predictors. We assume the
# dimensions of response are N X 1 and the dimensions of predictors are N X d.
# N represents the number of observations and d represents the number of
# features.

# %%
import keras
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras.backend as K
import xgboost as xgb
import numpy as np
import pandas as pd

def Logistic_regression_construction(response, predictors):

    from sklearn.linear_model import LogisticRegression

    # model_logistic = LogisticRegression()
    model_logistic = LogisticRegression(solver='lbfgs', multi_class='ovr')
    model_logistic.fit(predictors, response)
    return model_logistic

def RandomForest_Classifier_construction(response,predictors):
    from sklearn.ensemble import RandomForestClassifier
    
    # model_random_forest_classifier = RandomForestClassifier(max_depth=2, random_state=0)
    model_random_forest_classifier = RandomForestClassifier(n_estimators=100,random_state=0)
    model_random_forest_classifier.fit(predictors, response)
    return model_random_forest_classifier

def MLP_Classifier_construction(response, predictors):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier()
    clf.fit(predictors, response)
    return clf

def training_construction(response, predictors,Training_name):
    model = 0
    if Training_name=='LR':
        model=Logistic_regression_construction(response, predictors)
        
    elif Training_name=='RF':
        model=RandomForest_Classifier_construction(response, predictors)
        
    elif Training_name=='MLP':
        model=MLP_Classifier_construction(response, predictors)

    return model
