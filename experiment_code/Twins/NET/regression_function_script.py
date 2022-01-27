import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, TerminateOnNaN
from keras.optimizers import Adam, SGD
import keras.backend as K
from keras.metrics import binary_accuracy
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Dropout
from keras import regularizers
import tensorflow as tf
import numpy as np
import pandas as pd
from distance import *
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
def D_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    loss_D = tf.reduce_mean(K.binary_crossentropy(t_true, t_pred))

    return loss_D


def g_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    y_pre = t_true * y1_pred + (1 - t_true) * y0_pred
    '''factual loss of g and m'''
    g_factual_loss = tf.reduce_mean(K.binary_crossentropy(y_true, y_pre))
    return g_factual_loss



def D_accuracy(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)

def plus_ortho_loss(concat_true, concat_pred):
    y_true = tf.reshape(concat_true[:, 0], [-1, 1])
    d_true = tf.reshape(concat_true[:, 1], [-1, 1])
    d_pre = tf.reshape(concat_pred[:, 2], [-1, 1])
    y0_pre = tf.reshape(concat_pred[:, 0], [-1, 1])
    y1_pre = tf.reshape(concat_pred[:, 1], [-1, 1])
    prob_1_pre = d_pre
    y_pre = d_true * y1_pre + (1 - d_true) * y0_pre
    return tf.reduce_mean((y_true - y_pre) + (d_true - prob_1_pre))

def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))
class EpsilonLayer(Layer):

    def __init__(self):
        super(EpsilonLayer, self).__init__()

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, 1],
                                       initializer='RandomNormal',
                                       #  initializer='ones',
                                       trainable=True)
        super(EpsilonLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]
def net_loss(imb_fun, orthogonal_opt, TOR_type, epsilon, gm_ratio, imb_ratio, orthogonal_ratio, TOR_ratio):
    def final_loss(concat_true, concat_pred):
        '''concat_true is 2-dim [yf,d_true]; concat_pred is 4-dim [y0,y1,prob,epsilon]'''
        y_true = tf.reshape(concat_true[:, 0], [-1, 1])
        d_true = tf.reshape(concat_true[:, 1], [-1, 1])
        y0_pre = tf.reshape(concat_pred[:, 0], [-1, 1])
        y1_pre = tf.reshape(concat_pred[:, 1], [-1, 1])
        d_pre = tf.reshape(concat_pred[:, 2], [-1, 1])
        epsilons = tf.reshape(concat_pred[:, 3], [-1, 1])
        h_rep = concat_pred[:, 4:]
        y_pre = d_true*y1_pre + (1-d_true)*y0_pre
        '''factual loss of g and m'''
        g_factual_loss = g_loss(concat_true, concat_pred)
        m_loss = D_loss(concat_true, concat_pred)
        '''imbalanced loss'''
        if imb_fun:
            p_ipm = tf.reduce_mean(d_true)
            print('imblance function is ' + imb_fun)
            imb_loss = imb_error(imb_fun, h_rep, d_true, p_ipm)
        else:
            imb_loss = 0
        '''orthogonal regularization'''
        d_pre = (d_pre + 0.01) / 1.02
        prob_1_pre = d_pre
        prob_0_pre = 1-prob_1_pre
        if orthogonal_opt == '+':
            orthogonal_loss = tf.reduce_mean((y_true - y_pre)+(d_true - prob_1_pre))
        elif orthogonal_opt == '*':
            orthogonal_loss = tf.reduce_mean((y_true - y_pre) * (d_true - prob_1_pre))
        else:
            orthogonal_loss = 0
        '''treatment-outcome regularization (TOR)'''
        if TOR_type == 'IP':
            h = d_true / prob_1_pre - (1 - d_true) / prob_0_pre
        elif TOR_type == 'IPR':
            h = 1 - (d_true / prob_1_pre)
        elif TOR_type == 'PR':
            h = d_true - prob_1_pre
        else:
            h = 0
        if epsilon == True:
            epsilon_para = epsilons
        else:
            epsilon_para = epsilon
        y_tilda_pre = y_pre + epsilon_para * h
        TOR_loss = tf.reduce_mean(tf.square(y_true - y_tilda_pre))
        '''final loss'''
        loss = g_factual_loss + gm_ratio*m_loss + TOR_ratio*TOR_loss + imb_ratio*imb_loss + orthogonal_ratio*orthogonal_loss
        return loss
    return final_loss
def imb_error(imb_fun, h_rep_norm, d, p_ipm=0.5, rbf_sigma=0.1, r_alpha=1e-4):
    # imbalance loss
    if imb_fun == 'mmd2_rbf':
        imb_dist = mmd2_rbf(h_rep_norm, d, p_ipm, rbf_sigma)
        imb_error = r_alpha * imb_dist
    elif imb_fun == 'mmd2_lin':
        imb_dist = mmd2_lin(h_rep_norm, d, p_ipm)
        imb_error = r_alpha * imb_dist
    elif imb_fun == 'mmd_rbf':
        imb_dist = tf.abs(mmd2_rbf(h_rep_norm, d, p_ipm, rbf_sigma))
        imb_error = safe_sqrt(tf.square(r_alpha) * imb_dist)
    elif imb_fun == 'mmd_lin':
        imb_dist = mmd2_lin(h_rep_norm, d, p_ipm)
        imb_error = safe_sqrt(tf.square(r_alpha) * imb_dist)
    elif imb_fun == 'wass':
        imb_dist, imb_mat = wasserstein(h_rep_norm, d, p_ipm, lam=10.0, its=10, sq=False, backpropT=1)
        imb_error = r_alpha * imb_dist
    elif imb_fun == 'wass2':
        imb_dist, imb_mat = wasserstein(h_rep_norm, d, p_ipm, lam=10.0, its=10, sq=True, backpropT=1)
        imb_error = r_alpha * imb_dist
    else:
        imb_dist = lindisc(h_rep_norm, p_ipm, d)
        imb_error = r_alpha * imb_dist
    return imb_error
def make_tarnet(predictors, response, layer_reg=0.01, imb_fun='wass', orthogonal_opt='+', TOR_type='IP', epsilon=True, gm_ratio=1, imb_ratio=1, orthogonal_ratio=0.1, TOR_ratio=0.01):
    tf.random.set_seed(0)
    input_dim = predictors.shape[1]
    inputs = Input(shape=(input_dim,), name='input')
    # representation
    z = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    # z = Dropout(0.5)(z)
    z = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(z)
    # z = Dropout(0.5)(z)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(z)

    # m_head
    # d_pre = Dense(units=20)(inputs)
    # d_pre = Dense(units=10)(d_pre)
    d_pre = Dense(units=1, activation='sigmoid')(inputs)

    # g_head
    # phi = Dropout(0.5)(phi)
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(layer_reg))(phi)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(layer_reg))(phi)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(layer_reg))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(layer_reg))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(layer_reg), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(layer_reg), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(d_pre, name='epsilon')
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, d_pre, epsilons, phi])
    model = Model(inputs=inputs, outputs=concat_pred)

    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=5, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=True, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0)
    ]
    '''initial optimizer'''
    opt = Adam(lr=0.001, decay=0.001)
    # opt = SGD(lr=1e-5, momentum=0.9, nesterov=True)
    '''compile model_Y'''
    model.compile(loss=net_loss(imb_fun=imb_fun, orthogonal_opt=orthogonal_opt, TOR_type=TOR_type, epsilon=epsilon,
                                gm_ratio=gm_ratio,
                                imb_ratio=imb_ratio, orthogonal_ratio=orthogonal_ratio, TOR_ratio=TOR_ratio),
                  optimizer=opt,
                  metrics=[g_loss, D_loss, D_accuracy, track_epsilon,plus_ortho_loss])
    model.summary()
    # fit model
    model.fit(predictors, response,
              epochs=300, batch_size=int(1000), validation_split=0.2, shuffle=False, callbacks=callbacks)
    return model
def make_dragonnet(predictors, response, layer_reg=0.01, imb_fun='wass', orthogonal_opt='+', TOR_type='IP', epsilon=True, gm_ratio=1, imb_ratio=1, orthogonal_ratio=0.1, TOR_ratio=0.01):
    tf.random.set_seed(0)
    input_dim = predictors.shape[1]
    inputs = Input(shape=(input_dim,), name='input')
    # representation
    z = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    z = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(z)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(z)

    # m_head
    # d_pre = Dense(units=20)(phi)
    # d_pre = Dense(units=10)(d_pre)
    d_pre = Dense(units=1, activation='sigmoid')(phi)


    # g_head
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(layer_reg))(phi)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(layer_reg))(phi)

    # second layer
    y0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(layer_reg))(y0_hidden)
    y1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(layer_reg))(y1_hidden)

    # third
    y0_predictions = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(layer_reg), name='y0_predictions')(
        y0_hidden)
    y1_predictions = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(layer_reg), name='y1_predictions')(
        y1_hidden)

    dl = EpsilonLayer()
    epsilons = dl(d_pre, name='epsilon')
    concat_pred = Concatenate(1)([y0_predictions, y1_predictions, d_pre, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=5, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=True, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0)
    ]
    '''initial optimizer'''
    opt = Adam(lr=0.001, decay=0.001)
    # opt = SGD(lr=1e-5, momentum=0.9, nesterov=True)
    '''compile model_Y'''
    model.compile(loss=net_loss(imb_fun=imb_fun, orthogonal_opt=orthogonal_opt, TOR_type=TOR_type, epsilon=epsilon, gm_ratio=gm_ratio,
                  imb_ratio=imb_ratio, orthogonal_ratio=orthogonal_ratio, TOR_ratio=TOR_ratio), optimizer=opt, metrics=[g_loss, D_loss, D_accuracy, track_epsilon, plus_ortho_loss])
    model.summary()
    # fit model
    model.fit(predictors, response,
                epochs=300, batch_size=int(1000), validation_split=0.2, shuffle=False, callbacks=callbacks)
    return model
def make_net(predictors, response, layer_reg, imb_fun, orthogonal_opt, TOR_type, epsilon, gm_ratio, imb_ratio, orthogonal_ratio, TOR_ratio, model_name):
    model = 0
    if model_name in ['tarnet', 'tarnet_mmd_lin', 'tarnet_wass']:
        model = make_tarnet(predictors, response, layer_reg, imb_fun, orthogonal_opt, TOR_type, epsilon, gm_ratio, imb_ratio, orthogonal_ratio, TOR_ratio)
    if model_name in ['dragonnet', 'dragonnet_mmd_lin', 'dragonnet_wass']:
        model = make_dragonnet(predictors, response, layer_reg, imb_fun, orthogonal_opt, TOR_type, epsilon, gm_ratio, imb_ratio, orthogonal_ratio, TOR_ratio)
    return model