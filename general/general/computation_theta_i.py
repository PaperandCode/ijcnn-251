# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math

def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)
def compute_bq(blist, noise_term_vector, r, k, q):
    br_bar = blist[r-1]
    value = -br_bar*nCr(r, q) * noise_term_vector[r-q-1]
    for u in range(k-1-q, 0, -1):
        value = value - blist[q+u-1]*nCr(q+u, q)*noise_term_vector[u-1]
    return value
def Algorithm_coeff(noise_term_vector, r, k):
    '''noise_term_vector is a list: [E_nu, E_nu^2, ..., E_nu^r]'''
    blist = np.zeros(r).tolist()
    blist[r-1] = 1/noise_term_vector[r-1] # compute b_r
    for q in range(k-1, 0, -1):
        bq = compute_bq(blist, noise_term_vector, r, k, q)
        blist[q-1] = bq
    return blist

def mult_moments_computation(blist, nu_residual, E_res_m_list, k, r):
    br_bar = blist[r-1]
    value = 1/br_bar * nu_residual ** r
    for q in range(1, k):
        value = value + blist[q-1] * (nu_residual ** q - E_res_m_list[q-1])
    return value
def compute_dml_theta(g_pre, yf_i, g_pre_i, Prob_i):
    theta = np.mean(g_pre) + 1/g_pre.shape[0]*np.sum((yf_i - g_pre_i) / Prob_i)
    Prob_i_trim = Prob_i.copy()
    Prob_i_trim[Prob_i_trim<0.01] = 0.01
    Prob_i_trim[Prob_i_trim > 0.99] = 0.99
    theta_trim = np.mean(g_pre) + 1/g_pre.shape[0]*np.sum((yf_i - g_pre_i) / Prob_i_trim)
    return theta, theta_trim
def compute_0order_theta(g_pre):
    theta = np.mean(g_pre)
    return theta
def compute_IPW_theta(g_pre, yf_i, g_pre_i, Prob_i):
    IPW_theta = 1/g_pre.shape[0]*np.sum(yf_i/Prob_i)
    AIPW_theta = IPW_theta + np.mean(g_pre) - 1/g_pre.shape[0]*np.sum(g_pre_i/Prob_i)
    return IPW_theta, AIPW_theta
def compute_highorder_theta(g_pre, res_m, yi_minus_gi_given_z_list, r, k, E_res_m_list, E_nu_r_list, method):
    '''E_res_m_list is estimate E_nu, E_nur_r_list is true E_nu'''
    if k == 1:
        mult_m = 0
        if method == 'estimate':
            mult_m = 1 / E_res_m_list[r-1] * (res_m ** r)
        elif method == 'true':
            mult_m = 1 / E_nu_r_list[r - 1] * (res_m ** r)
        theta_DR_order = np.mean(g_pre)
        theta_reg_list = np.dot(yi_minus_gi_given_z_list.T, mult_m)/mult_m.shape[0]
        theta = theta_DR_order + np.mean(theta_reg_list)
        return theta
    else:
        mult_m = 0
        if method == 'estimate':
            blist_unknown = Algorithm_coeff(noise_term_vector=E_res_m_list, r=r, k=k)
            mult_m = mult_moments_computation(blist=blist_unknown, nu_residual=res_m, E_res_m_list=E_res_m_list, k=k, r=r)
        elif method == 'true':
            blist_known = Algorithm_coeff(noise_term_vector=E_nu_r_list, r=r, k=k)
            mult_m = mult_moments_computation(blist=blist_known, nu_residual=res_m, E_res_m_list=E_nu_r_list, k=k, r=r)
        theta_DR_order = np.mean(g_pre)
        theta_reg_list = np.dot(yi_minus_gi_given_z_list.T, mult_m)/mult_m.shape[0]
        theta = theta_DR_order + np.mean(theta_reg_list)
        return theta


def compute_ATE(ATE_df, exp_df, num_D, max_r, model_g_methods, model_m_methods):
    '''compute ATE'''
    '''compute true ATE'''
    for i in range(0, num_D):
        for j in range(i + 1, num_D):
            for estimator in ['DR', 'IPW', 'AIPW', 'DML', 'trim_DML']:
                ATE_df.loc['true', estimator + '_' + str(i + 1) + ',' + str(j + 1)] = exp_df.loc[
                                                                                       'true', estimator + '_' + str(
                                                                                           i + 1)] - exp_df.loc[
                                                                                       'true', estimator + '_' + str(
                                                                                           j + 1)]
            for r in range(2, max_r + 1):
                for k in range(1, r + 1):
                    ATE_df.loc['true', str(r) + '&' + str(k) + '_' + str(i + 1) + ',' + str(j + 1)] = \
                        exp_df.loc['true', str(r) + '&' + str(k) + '_' + str(i + 1)] - exp_df.loc[
                            'true', str(r) + '&' + str(k) + '_' + str(j + 1)]

            for training_id_g in range(0, len(model_g_methods)):
                for training_id_m in range(0, len(model_m_methods)):
                    training_name_g = model_g_methods[training_id_g]
                    training_name_m = model_m_methods[training_id_m]
                    for estimator in ['DR', 'IPW', 'AIPW', 'DML', 'trim_DML']:
                        '''compute DR, IPW, DML ATE'''
                        ATE_df.loc[training_name_g + ',' + training_name_m, estimator + '_' + str(i + 1) + ',' + str(j + 1)] = \
                            exp_df.loc[training_name_g + ',' + training_name_m, estimator + '_' + str(i + 1)] - exp_df.loc[
                                training_name_g + ',' + training_name_m, estimator + '_' + str(j + 1)]
                        ATE_df.loc[training_name_g + ',' + training_name_m + '_e', estimator + '_' + str(i + 1) + ',' + str(j + 1)] = \
                            np.abs(
                                ATE_df.loc[training_name_g + ',' + training_name_m, estimator + '_' + str(i + 1) + ',' + str(j + 1)] /
                                ATE_df.loc['true', estimator + '_' + str(i + 1) + ',' + str(j + 1)] - 1)

                    '''compute high-order ATE'''
                    for r in range(2, max_r + 1):
                        for k in range(1, r + 1):
                            ATE_df.loc[training_name_g + ',' + training_name_m, str(r) + '&' + str(k) + '_' + str(
                                i + 1) + ',' + str(j + 1)] = \
                                exp_df.loc[
                                    training_name_g + ',' + training_name_m, str(r) + '&' + str(k) + '_' + str(i + 1)] - \
                                exp_df.loc[
                                    training_name_g + ',' + training_name_m, str(r) + '&' + str(k) + '_' + str(j + 1)]
                            ATE_df.loc[
                                training_name_g + ',' + training_name_m + '_e', str(r) + '&' + str(k) + '_' + str(
                                    i + 1) + ',' + str(j + 1)] = \
                                np.abs(ATE_df.loc[
                                           training_name_g + ',' + training_name_m, str(r) + '&' + str(k) + '_' + str(
                                               i + 1) + ',' + str(j + 1)] / ATE_df.loc[
                                           'true', 'DR_' + str(i + 1) + ',' + str(j + 1)] - 1)

    return ATE_df
