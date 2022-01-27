# -*- coding: utf-8 -*-
import keras
import numpy as np
import time
import os
import pandas as pd
import regression_function_script as rs
import classification_function_script as cf
import computation_theta_i
from sklearn.preprocessing import StandardScaler
from utils import *
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]='0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.1
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
def make_model_dict(model_net_methods, imb_funs=None):
    model_dict = {}
    for model_name in model_net_methods:
        if imb_funs == None:
            model_dict[model_name] = 0
        else:
            for imb_fun in imb_funs:
                model_dict[model_name+'_'+imb_fun] = 0
    return model_dict
def run_ihdp(M0, M, dataset_path):
    method='estimate'
    M=M # number of simulated experiment
    orthogonal_option = '+'
    TOR_type = 'IP'
    epsilon = True
    gm_ratio = 1
    imb_ratio = 0
    orthogonal_ratio = 0
    TOR_ratio = 0
    num_D=2
    max_r=2
    sample_times=1000
    t = time.time()
    imb_funs = ['wass']
    model_net_methods = ['tarnet', 'dragonnet']
    # model_net_methods = ['dragonnet']
    model_net_dict = make_model_dict(model_net_methods)
    # model_net_dict = {'tarnet_mmd2_rbf':0,'tarnet_mmd2_lin':0,'tarnet_mmd_rbf':0,'tarnet_mmd_lin':0,'tarnet_wass':0,'tarnet_wass2':0}
    for m in range(M0, M):
        z, y_all, yf, D, mu0, mu1, itrain, itest = get_ihdp_data(dataset_path, m)
        N = z.shape[0]
        z_train, D_train, yf_train, mu0_train, mu1_train = z[itrain], D[itrain], yf[itrain], mu0[itrain], mu1[itrain]
        z_test, D_test, yf_test, mu0_test, mu1_test = z[itest], D[itest], yf[itest], mu0[itest], mu1[itest]
        yfd_true_train = np.concatenate([yf[itrain], D_train], 1)
        true_g_train = [mu0_train, mu1_train]
        true_g_test = [mu0_test, mu1_test]
        for model_name in model_net_dict.keys():
            if imb_ratio==0:
                model_net_dict[model_name] = \
                    rs.make_net(predictors=z_train.copy(), response=yfd_true_train.copy(), layer_reg=0.01,
                                imb_fun=False, orthogonal_opt=orthogonal_option, TOR_type=TOR_type, epsilon=epsilon, gm_ratio=gm_ratio,
                                imb_ratio=imb_ratio, orthogonal_ratio=orthogonal_ratio, TOR_ratio=TOR_ratio, model_name=model_name)
            else:
                for imb_fun in imb_funs:
                    model_net_dict[model_name] = \
                        rs.make_net(predictors=z_train.copy(), response=yfd_true_train.copy(), layer_reg=0.01, imb_fun=imb_fun, orthogonal_opt=orthogonal_option,
                                    TOR_type=TOR_type, epsilon=epsilon,
                                    gm_ratio=gm_ratio, imb_ratio=imb_ratio, orthogonal_ratio=orthogonal_ratio, TOR_ratio=TOR_ratio, model_name=model_name)
        if not os.path.isdir('./ATE'):
            os.mkdir('./ATE')
        if not os.path.isdir('./exp'):
            os.mkdir('./exp')
        if not os.path.isdir('./ATTE'):
            os.mkdir('./ATTE')
        if not os.path.isdir('./cond_exp'):
            os.mkdir('./cond_exp')
        if not os.path.isdir('./ITE'):
            os.mkdir('./ITE')
        exp_headings = make_exp_headings(max_r, num_D)
        ATE_headings = make_ATE_headings(max_r, num_D)
        # saving_list = make_saving_index_for_NET(model_net_methods)
        exp_df = pd.DataFrame(columns=exp_headings)
        exp_df_test = pd.DataFrame(columns=exp_headings)
        ATE_df = pd.DataFrame(columns=ATE_headings)
        ATE_df_test = pd.DataFrame(columns=ATE_headings)
        for model_name in model_net_dict.keys():
            model_net = model_net_dict[model_name]
            yprob1_pre_train = model_net.predict(z_train)
            y0_pre_train = yprob1_pre_train[:, 0].reshape(-1, 1)
            y1_pre_train = yprob1_pre_train[:, 1].reshape(-1, 1)
            prob1_pre_train = yprob1_pre_train[:, 2].reshape(-1, 1)
            prob0_pre_train = 1 - prob1_pre_train
            g_pre_train_list = [y0_pre_train, y1_pre_train]
            prob_pre_train_list = [prob0_pre_train, prob1_pre_train]
            yprob1_pre_test = model_net.predict(z_test)
            y0_pre_test = yprob1_pre_test[:, 0].reshape(-1, 1)
            y1_pre_test = yprob1_pre_test[:, 1].reshape(-1, 1)
            prob1_pre_test = yprob1_pre_test[:, 2].reshape(-1, 1)
            prob0_pre_test = 1 - prob1_pre_test
            g_pre_test_list = [y0_pre_test, y1_pre_test]
            prob_pre_test_list = [prob0_pre_test, prob1_pre_test]
            '''compute theta^i'''
            for i in range(0, num_D):
                idx_i_train = np.where(D_train == i)[0]
                idx_i_test = np.where(D_test == i)[0]
                '''true theta^i'''
                for estimator in ['DR', 'IPW', 'AIPW', 'DML', 'trim_DML']:
                    exp_df.loc['true', estimator + '_' + str(i + 1)] = np.mean(true_g_train[i])
                    exp_df_test.loc['true', estimator + '_' + str(i + 1)] = np.mean(true_g_test[i])
                for r in range(2, max_r + 1):
                    for k in range(1, r + 1):
                        exp_df.loc['true', str(r) + '&' + str(k) + '_' + str(i + 1)] = np.mean(true_g_train[i])
                        exp_df_test.loc['true', str(r) + '&' + str(k) + '_' + str(i + 1)] = np.mean(true_g_test[i])
                '''prepare data to compute theta'''
                g_i_pre_given_z_train, g_i_pre_given_zi_train, yf_i_train, prob_i_given_zi_train, yi_minus_gi_given_z_list_train, res_nu_i_train, E_res_nu_i_list_train =\
                    prepare_data_for_NET(g_pre_train_list[i], idx_i_train, yf_train, prob_pre_train_list[i], max_r, sample_times)
                g_i_pre_given_z_test, g_i_pre_given_zi_test, yf_i_test, prob_i_given_zi_test, yi_minus_gi_given_z_list_test, res_nu_i_test, E_res_nu_i_list_test =\
                    prepare_data_for_NET(g_pre_test_list[i], idx_i_test, yf_test, prob_pre_test_list[i], max_r, sample_times)
                '''compute DR theta'''
                exp_df.loc[model_name, 'DR_' + str(i + 1)] = \
                    computation_theta_i.compute_0order_theta(g_i_pre_given_z_train)
                exp_df.loc[model_name+'_e', 'DR_' + str(i + 1)] = \
                    np.abs(exp_df.loc[model_name, 'DR_' + str(i + 1)]/exp_df.loc['true','DR_' + str(i + 1)]-1)
                exp_df_test.loc[model_name, 'DR_' + str(i + 1)] = \
                    computation_theta_i.compute_0order_theta(g_i_pre_given_z_test)
                exp_df_test.loc[model_name +'_e', 'DR_' + str(i + 1)] = \
                    np.abs(exp_df_test.loc[model_name, 'DR_' + str(i + 1)] / exp_df_test.loc[
                        'true', 'DR_' + str(i + 1)] - 1)
                '''compute IPW theta'''
                exp_df.loc[model_name, 'IPW_' + str(i + 1)], exp_df.loc[model_name, 'AIPW_' + str(i + 1)] = \
                    computation_theta_i.compute_IPW_theta(g_i_pre_given_z_train, yf_i_train, g_i_pre_given_zi_train, prob_i_given_zi_train)
                exp_df.loc[model_name + '_e', 'IPW_' + str(i + 1)] = np.abs(exp_df.loc[model_name, 'IPW_' + str(i + 1)] / exp_df.loc['true', 'DR_' + str(i + 1)] - 1)
                exp_df.loc[model_name + '_e', 'AIPW_' + str(i + 1)] = np.abs(exp_df.loc[model_name, 'AIPW_' + str(i + 1)] / exp_df.loc['true', 'DR_' + str(i + 1)] - 1)

                exp_df_test.loc[model_name, 'IPW_' + str(i + 1)], exp_df_test.loc[
                    model_name, 'AIPW_' + str(i + 1)] = computation_theta_i.compute_IPW_theta(g_i_pre_given_z_test, yf_i_test, g_i_pre_given_zi_test, prob_i_given_zi_test)
                exp_df_test.loc[model_name + '_e', 'IPW_' + str(i + 1)] = np.abs(exp_df_test.loc[model_name, 'IPW_' + str(i + 1)] / exp_df_test.loc[
                        'true', 'DR_' + str(i + 1)] - 1)
                exp_df_test.loc[model_name + '_e', 'AIPW_' + str(i + 1)] = np.abs(exp_df_test.loc[model_name, 'AIPW_' + str(i + 1)] / exp_df_test.loc[
                        'true', 'DR_' + str(i + 1)] - 1)
                '''compute dml theta'''
                exp_df.loc[model_name, 'DML_' + str(i + 1)], exp_df.loc[model_name, 'trim_DML_' + str(i + 1)] = \
                    computation_theta_i.compute_dml_theta(g_i_pre_given_z_train, yf_i_train, g_i_pre_given_zi_train, prob_i_given_zi_train)
                exp_df.loc[model_name + '_e', 'DML_' + str(i + 1)] = \
                    np.abs(exp_df.loc[model_name, 'DML_' + str(i + 1)]/exp_df.loc['true','DR_' + str(i + 1)]-1)
                exp_df.loc[model_name + '_e', 'trim_DML_' + str(i + 1)] = \
                    np.abs(exp_df.loc[model_name, 'trim_DML_' + str(i + 1)]/exp_df.loc['true','DR_' + str(i + 1)]-1)

                exp_df_test.loc[model_name, 'DML_' + str(i + 1)], exp_df_test.loc[model_name, 'trim_DML_' + str(i + 1)] = \
                    computation_theta_i.compute_dml_theta(g_i_pre_given_z_test, yf_i_test, g_i_pre_given_zi_test, prob_i_given_zi_test)
                exp_df_test.loc[model_name + '_e', 'DML_' + str(i + 1)] = \
                    np.abs(exp_df_test.loc[model_name, 'DML_' + str(i + 1)]/exp_df_test.loc['true', 'DR_' + str(i + 1)]-1)
                exp_df_test.loc[model_name + '_e', 'trim_DML_' + str(i + 1)] = \
                    np.abs(exp_df_test.loc[model_name, 'trim_DML_' + str(i + 1)]/exp_df_test.loc['true', 'DR_' + str(i + 1)]-1)
                '''compute rcl theta'''
                for r in range(2, max_r+1):
                    for k in range(1, r+1):
                        exp_df.loc[model_name, str(r)+'&'+str(k)+'_'+ str(i + 1)] = \
                            computation_theta_i.compute_highorder_theta(g_i_pre_given_z_train, res_nu_i_train, yi_minus_gi_given_z_list_train,
                                                                  r, k, E_res_nu_i_list_train, None, method)
                        exp_df.loc[model_name + '_e', str(r) + '&' + str(k) + '_' + str(i + 1)] = \
                            np.abs(exp_df.loc[model_name, str(r) + '&' + str(k) + '_' + str(i + 1)]/exp_df.loc['true', 'DR_' + str(i + 1)]-1)
                        exp_df_test.loc[model_name, str(r)+'&'+str(k)+'_'+ str(i + 1)] = \
                            computation_theta_i.compute_highorder_theta(g_i_pre_given_z_test, res_nu_i_test, yi_minus_gi_given_z_list_test,
                                                                  r, k, E_res_nu_i_list_test, None, method)
                        exp_df_test.loc[model_name + '_e', str(r) + '&' + str(k) + '_' + str(i + 1)] = \
                            np.abs(exp_df_test.loc[model_name, str(r) + '&' + str(k) + '_' + str(i + 1)] / exp_df_test.loc['true', 'DR_' + str(i + 1)] - 1)
        exp_df.to_csv('./exp/'+str(N)+"_exp_" + str(m)+'_train'+ '.csv')
        exp_df_test.to_csv('./exp/' + str(N)+"_exp_" + str(m)+'_test'+'.csv')
        '''ATE results'''
        ATE_df = computation_theta_i.compute_ATE(ATE_df, exp_df, num_D, max_r, model_net_methods)
        ATE_df_test = computation_theta_i.compute_ATE(ATE_df_test, exp_df_test, num_D, max_r, model_net_methods)
        ATE_df.to_csv('./ATE/'+str(N)+"_ATE_" + str(m)+'_train'+'.csv')
        ATE_df_test.to_csv('./ATE/' + str(N)+"_ATE_" + str(m)+'_test'+ '.csv')
        keras.backend.clear_session()
    elapsed = time.time() - t
    print('The time taken is ' + str(elapsed))
if __name__ == '__main__':
    run_ihdp(0, 1000, r'./dataset_1-1000')