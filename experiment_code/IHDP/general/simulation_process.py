import numpy as np
from sklearn.preprocessing import LabelEncoder
import math
import pandas as pd
import os
from scipy.special import erfinv
from scipy.stats import norm
import sys
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import joblib
from scipy import stats
from random import sample
from sklearn.model_selection import train_test_split

def prob_to_value(prob_list):
    return np.random.multinomial(n=1, pvals=prob_list)
def sum_exp(z, num_z, num_sum):
    zlist = list(range(0, num_z))
    sum_list=sample(zlist, num_sum)
    sum = 0
    for i in sum_list:
        sum = sum+np.exp(0.2*z[:, i])
    sum = sum.reshape(-1, 1)
    return sum
def build_nonlinear_d(z, num_z, confounding_ratio):
    confounding_size = int(confounding_ratio*num_z)
    coef_1 = np.random.uniform(low=-0.1, high=0.1, size=(confounding_size, 1))
    coef_2 = np.random.uniform(low=-0.1, high=0.1, size=(confounding_size, 1))
    coef_3 = np.random.uniform(low=-0.1, high=0.1, size=(confounding_size, 1))
    z_for_d = z[:, 0:confounding_size]
    combine_z1 = np.dot(z_for_d, coef_1)
    combine_z2 = np.dot(z_for_d, coef_2)
    combine_z3 = np.dot(z_for_d, coef_3)
    true_prob_0 = np.exp(combine_z1) / (np.exp(combine_z1)+np.exp(combine_z2)+np.exp(combine_z3))
    true_prob_1 = np.exp(combine_z2) / (np.exp(combine_z1)+np.exp(combine_z2)+np.exp(combine_z3))
    true_prob_2 = np.exp(combine_z3) / (np.exp(combine_z1)+np.exp(combine_z2)+np.exp(combine_z3))
    prob = np.hstack((true_prob_0, true_prob_1, true_prob_2))
    d_onehot = np.array(list(map(lambda x: np.random.multinomial(n=1, pvals=x), prob)))
    d = np.squeeze(np.array(list(map(lambda x: np.argwhere(x==1), d_onehot)))).reshape(-1, 1)
    return d, true_prob_0, true_prob_1, true_prob_2
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def nonlinear_dataset(M0, M=100, N=1000, outputpath='', seed=0, confounding_ratio=1):
    if not os.path.isdir('./dataset'):
        os.mkdir('./dataset')
    confounding_ratio = confounding_ratio
    num_z = 10  # number of confounder variables Z
    z_loc, z_scale = 0, 1
    np.random.seed(seed=3)
    z = np.random.normal(loc=z_loc, scale=z_scale, size=(N, num_z))
    # construct d
    d_factual_index, true_prob_0, true_prob_1, true_prob_2 = build_nonlinear_d(z, num_z, confounding_ratio)
    indicator_0 = np.zeros((N, 1))
    indicator_1 = np.zeros((N, 1))
    indicator_2 = np.zeros((N, 1))
    idx_0 = np.where(d_factual_index == 0)
    idx_1 = np.where(d_factual_index == 1)
    idx_2 = np.where(d_factual_index == 2)
    indicator_0[idx_0, 0] = 1
    indicator_1[idx_1, 0] = 1
    indicator_2[idx_2, 0] = 1
    nu0 = indicator_0 - true_prob_0
    nu1 = indicator_1 - true_prob_1
    nu2 = indicator_2 - true_prob_2
    for count in range(M0, M):
        # construct y
        np.random.seed(seed=None)
        coef_z = np.random.uniform(low=1, high=1.5, size=(num_z, 1))
        coef_z1 = np.random.uniform(low=0.1, high=0.5, size=(num_z, 1))
        coef_z2 = np.random.uniform(low=0.1, high=0.5, size=(num_z, 1))
        coef_z3 = np.random.uniform(low=0.1, high=0.5, size=(num_z, 1))
        com_z1 = np.dot(z, coef_z1)
        com_z2 = np.dot(z, coef_z2)
        com_z3 = np.dot(z, coef_z3)
        g1 = np.power(com_z1+1, 2)*np.exp(np.power(0.1, 0.5))
        g2 = np.power(com_z2+1, 2)*np.exp(np.power(0.5, 0.5))
        g3 = np.power(com_z3+1, 2)*np.exp(np.power(1, 0.5))
        sigma_1 = np.power(9, 0.5)
        sigma_2 = np.power(4, 0.5)
        sigma_3 = np.power(1, 0.5)
        xi_1 = np.random.normal(loc=0, scale=sigma_1, size=(N, 1))
        xi_2 = np.random.normal(loc=0, scale=sigma_2, size=(N, 1))
        xi_3 = np.random.normal(loc=0, scale=sigma_3, size=(N, 1))
        y1 = g1 + xi_1
        y2 = g2 + xi_2
        y3 = g3 + xi_3
        y_factual = []
        for i in range(0, len(d_factual_index)):
            if d_factual_index[i] == 0:
                y_factual.append(y1[i])
            if d_factual_index[i] == 1:
                y_factual.append(y2[i])
            if d_factual_index[i] == 2:
                y_factual.append(y3[i])
        y_factual = np.array(y_factual).reshape(-1, 1)
        itrain, itest = train_test_split(np.arange(len(d_factual_index)), test_size=0.3, shuffle=False)
        joblib.dump([z, d_factual_index, y_factual, y1, y2, y3, itrain, itest, g1, g2, g3, true_prob_0, true_prob_1, true_prob_2], outputpath + '/' + str(N) + '_experiment_' + str(count))

if __name__ == '__main__':
    nonlinear_dataset(M0=0, M=1, outputpath='./dataset', N=40000, seed=5, confounding_ratio=1)