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
from sklearn.model_selection import GridSearchCV
def RandomForest_construction(response,predictors):
        
    from sklearn.ensemble import RandomForestRegressor
    # param_grid = {
    #     'bootstrap': [True],
    #     'max_depth': [10, 100],
    #     'max_features': [3, 10],
    #     'min_samples_leaf': [3, 10],
    #     'min_samples_split': [0.1, 0.3],
    #     'n_estimators': [30, 100, 200]
    # }
    # rf = RandomForestRegressor()
    # model_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
    # model_rf.fit(predictors, response)
    # model_rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
    model_rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
    # model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=0)
    model_rf.fit(predictors, response)
    
    return model_rf


def LASSO_construction(response, predictors, cross_validation=5,
                       tuning_parameters={'alpha': [1e-10, 1e-5, 1e-3, 1e-2, 0.1, 1]}):
    from sklearn.linear_model import Lasso
    import numpy as np

    lasso = Lasso(max_iter=10000)
    model_lasso = GridSearchCV(lasso, tuning_parameters,
                               scoring='neg_mean_squared_error', cv=cross_validation)
    model_lasso.fit(predictors, response)

    return model_lasso

def MLP_construction(response, predictors):
    tf.random.set_seed(1)
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.optimizers import SGD, Adam
    from keras.callbacks import EarlyStopping
    # Here we let X shape = [observations, number of features]
    layer_reg=0.01
    train_X = predictors
    train_y = response
    model_mlp = Sequential()
    # input layer
    model_mlp.add(Dense(units=200, kernel_initializer='normal',kernel_regularizer=regularizers.l2(layer_reg), activation='relu', input_dim=train_X.shape[1]))
    # hidden layer
    model_mlp.add(Dense(units=200, kernel_initializer='normal', kernel_regularizer=regularizers.l2(layer_reg),activation='relu'))
    model_mlp.add(Dense(units=200, kernel_initializer='normal', kernel_regularizer=regularizers.l2(layer_reg),activation='relu'))
    model_mlp.add(Dense(units=100, kernel_initializer='normal', kernel_regularizer=regularizers.l2(layer_reg),activation='relu'))
    model_mlp.add(Dense(units=100, kernel_initializer='normal', kernel_regularizer=regularizers.l2(layer_reg),activation='relu'))
    # model_mlp.add(Dense(units=5, kernel_initializer='normal',activation='relu'))
    # output layer
    model_mlp.add(Dense(units=1))
    # parameters
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, verbose=0, mode='auto')
    opt = Adam(lr=0.001, decay=0.005)
    model_mlp.compile(loss='mse', optimizer=opt)
    model_mlp.summary()
    # fit model_mlp
    model_mlp.fit(train_X, train_y, epochs=300, batch_size=int(64),  validation_split=0.2, shuffle=False, callbacks=[earlystop])
    return model_mlp

# %%
def training_construction(response, predictors,Training_name):
    model = 0
    if Training_name=='RF':
        model=RandomForest_construction(response, predictors)

    elif Training_name=='LASSO':
        cross_validation_LASSO = 5
        tuning_parameters_LASSO = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
        model=LASSO_construction(response, predictors,
                           cross_validation_LASSO, tuning_parameters_LASSO)

    elif Training_name=='MLP':
        model=MLP_construction(response, predictors)

    return model
