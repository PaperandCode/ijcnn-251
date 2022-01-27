import keras
import numpy as np
import time
import os
import pandas as pd
import regression_function_script as rs
import classification_function_script as cf
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import simulation_process as simu
import compute_results
import computation_theta_i
from utils import *
import tensorflow as tf
def run_ihdp(M0, M, dataset_path):
    method = 'estimate'
    sample_times = 1000 # construct xi_cf times
    num_D = 2
    M=M # number of simulated experiment
    # model_g_methods = ['RF']
    # model_m_methods = ['RF']
    model_g_methods = ['LASSO', 'RF', 'MLP']
    model_m_methods = ['LR', 'RF', 'MLP']
    model_g_dict = {'LASSO': [0] * num_D, 'RF': [0] * num_D, 'MLP': [0] * num_D}
    model_m_dict = {'LR': 0, 'RF': 0, 'MLP': 0}
    t = time.time()
    for m in range(M0, M):
        z, y_all, yf, D, mu0, mu1, itrain, itest = get_ihdp_data(dataset_path, m)
        N = z.shape[0]
        z_train, D_train, yf_train= z[itrain], D[itrain], yf[itrain]
        z_test, D_test, yf_test = z[itest], D[itest], yf[itest]
        true_g = [mu0, mu1]
        true_prob = [np.zeros((N,1)), np.zeros((N,1)), np.zeros((N,1))]
        # Train the function m0 (treatment)
        for model_m_name in model_m_methods:
            model_m_dict[model_m_name] = cf.training_construction(D_train, z_train, model_m_name)            
        for i in range(0, num_D):
            idx_i_train = np.where(D_train == i)[0]
            zi_train, yf_i_train = z_train[idx_i_train], yf_train[idx_i_train]
            # Train the function g0(di,.)
            for training_sequence in range(0, len(model_g_methods)):
                Training_name=model_g_methods[training_sequence]
                model_g_hat = model_g_dict[Training_name]
                model_g_hat[i] = rs.training_construction\
                    (yf_i_train, zi_train, Training_name)
        if not os.path.isdir('./ATE'):
            os.mkdir('./ATE')
        if not os.path.isdir('./exp'):
            os.mkdir('./exp')
        if not os.path.isdir('./ATTE'):
            os.mkdir('./ATTE')
        if not os.path.isdir('./cond_exp'):
            os.mkdir('./cond_exp')
        exp_headings = make_exp_headings(max_r, num_D)
        ATE_headings = make_ATE_headings(max_r, num_D)
        saving_list = make_saving_index(model_g_methods, model_m_methods)
        exp_df = pd.DataFrame(columns=exp_headings, index=saving_list)
        exp_df_test = pd.DataFrame(columns=exp_headings, index=saving_list)
        ATE_df = pd.DataFrame(columns=ATE_headings, index=saving_list)
        ATE_df_test = pd.DataFrame(columns=ATE_headings, index=saving_list)
        for i in range(0, num_D):
            idx_i_train = np.where(D_train == i)[0]
            zi_train, yf_i_train = z_train[idx_i_train], yf_train[idx_i_train]
            idx_i_test = np.where(D_test == i)[0]
            zi_test, yf_i_test = z_test[idx_i_test], yf_test[idx_i_test]
            '''compute theta^i'''
            '''true theta^i'''
            for estimator in ['DR', 'IPW', 'AIPW', 'DML', 'trim_DML']:
                exp_df.loc['true', estimator + '_' + str(i + 1)] = np.mean(true_g[i][itrain])
                exp_df_test.loc['true', estimator + '_' + str(i + 1)] = np.mean(true_g[i][itest])
            for r in range(2, max_r + 1):
                for k in range(1, r + 1):
                    exp_df.loc['true', str(r) + '&' + str(k) + '_' + str(i + 1)] = np.mean(true_g[i][itrain])
                    exp_df_test.loc['true', str(r) + '&' + str(k) + '_' + str(i + 1)] = np.mean(true_g[i][itest])
            for training_id_g in range(0, len(model_g_methods)):
                training_name_g = model_g_methods[training_id_g]
                model_g_i = model_g_dict[training_name_g][i]
                g_i_pre_given_z_train, g_i_pre_given_zi_train, yi_minus_gi_given_z_list_train = prepare_g_related(
                    model_g_i, yf_train, z_train, idx_i_train, sample_times)
                g_i_pre_given_z_test, g_i_pre_given_zi_test, yi_minus_gi_given_z_list_test = prepare_g_related(
                    model_g_i, yf_test, z_test, idx_i_test, sample_times)
                for training_id_m in range(0, len(model_m_methods)):
                    training_name_m = model_m_methods[training_id_m]
                    model_m = model_m_dict[training_name_m]
                    '''prepare propensity score related data'''
                    prob_all_train = model_m.predict_proba(z_train)
                    prob_all_test = model_m.predict_proba(z_test)
                    prob_i_given_zi_train = prob_all_train[idx_i_train, i].reshape(-1, 1)
                    prob_i_given_zi_test = prob_all_test[idx_i_test, i].reshape(-1, 1)
                    res_nu_i_train, E_res_nu_i_list_train, E_nu_i_list_train = \
                        prepare_m_related(prob_all_train, true_prob[i][itrain], idx_i_train, i, z_train.shape[0], max_r, method)
                    res_nu_i_test, E_res_nu_i_list_test, E_nu_i_list_test = \
                        prepare_m_related(prob_all_test, true_prob[i][itest], idx_i_test, i, z_test.shape[0], max_r, method)
                    '''compute DR theta'''
                    exp_df.loc[training_name_g + ',' + training_name_m, 'DR_' + str(i + 1)] = \
                        computation_theta_i.compute_0order_theta(g_i_pre_given_z_train)
                    exp_df.loc[training_name_g + ',' + training_name_m+'_e', 'DR_' + str(i + 1)] = \
                        np.abs(exp_df.loc[training_name_g + ',' + training_name_m, 'DR_' + str(i + 1)]/exp_df.loc['true','DR_' + str(i + 1)]-1)
                    exp_df_test.loc[training_name_g + ',' + training_name_m, 'DR_' + str(i + 1)] = \
                        computation_theta_i.compute_0order_theta(g_i_pre_given_z_test)
                    exp_df_test.loc[training_name_g + ',' + training_name_m +'_e', 'DR_' + str(i + 1)] = \
                        np.abs(exp_df_test.loc[training_name_g + ',' + training_name_m, 'DR_' + str(i + 1)] / exp_df_test.loc[
                            'true', 'DR_' + str(i + 1)] - 1)
                    '''compute IPW theta'''
                    exp_df.loc[training_name_g + ',' + training_name_m, 'IPW_' + str(i + 1)], exp_df.loc[training_name_g + ',' + training_name_m, 'AIPW_' + str(i + 1)] = \
                        computation_theta_i.compute_IPW_theta(g_i_pre_given_z_train, yf_i_train, g_i_pre_given_zi_train, prob_i_given_zi_train)
                    exp_df.loc[training_name_g + ',' + training_name_m + '_e', 'IPW_' + str(i + 1)] = np.abs(exp_df.loc[training_name_g + ',' + training_name_m, 'IPW_' + str(i + 1)] / exp_df.loc['true', 'DR_' + str(i + 1)] - 1)
                    exp_df.loc[training_name_g + ',' + training_name_m + '_e', 'AIPW_' + str(i + 1)] = np.abs(exp_df.loc[training_name_g + ',' + training_name_m, 'AIPW_' + str(i + 1)] / exp_df.loc['true', 'DR_' + str(i + 1)] - 1)

                    exp_df_test.loc[training_name_g + ',' + training_name_m, 'IPW_' + str(i + 1)], exp_df_test.loc[
                        training_name_g + ',' + training_name_m, 'AIPW_' + str(i + 1)] = computation_theta_i.compute_IPW_theta(g_i_pre_given_z_test, yf_i_test, g_i_pre_given_zi_test, prob_i_given_zi_test)
                    exp_df_test.loc[training_name_g + ',' + training_name_m + '_e', 'IPW_' + str(i + 1)] = np.abs(exp_df_test.loc[training_name_g + ',' + training_name_m, 'IPW_' + str(i + 1)] / exp_df_test.loc[
                            'true', 'DR_' + str(i + 1)] - 1)
                    exp_df_test.loc[training_name_g + ',' + training_name_m + '_e', 'AIPW_' + str(i + 1)] = np.abs(exp_df_test.loc[training_name_g + ',' + training_name_m, 'AIPW_' + str(i + 1)] / exp_df_test.loc[
                            'true', 'DR_' + str(i + 1)] - 1)
                    '''compute dml theta'''
                    exp_df.loc[training_name_g + ',' + training_name_m, 'DML_' + str(i + 1)], exp_df.loc[training_name_g + ',' + training_name_m, 'trim_DML_' + str(i + 1)] = \
                        computation_theta_i.compute_dml_theta(g_i_pre_given_z_train, yf_i_train, g_i_pre_given_zi_train, prob_i_given_zi_train)                    
                    exp_df.loc[training_name_g + ',' + training_name_m + '_e', 'DML_' + str(i + 1)] = \
                        np.abs(exp_df.loc[training_name_g + ',' + training_name_m, 'DML_' + str(i + 1)]/exp_df.loc['true','DR_' + str(i + 1)]-1)                    
                    exp_df.loc[training_name_g + ',' + training_name_m + '_e', 'trim_DML_' + str(i + 1)] = \
                        np.abs(exp_df.loc[training_name_g + ',' + training_name_m, 'trim_DML_' + str(i + 1)]/exp_df.loc['true','DR_' + str(i + 1)]-1)
                    
                    exp_df_test.loc[training_name_g + ',' + training_name_m, 'DML_' + str(i + 1)], exp_df_test.loc[training_name_g + ',' + training_name_m, 'trim_DML_' + str(i + 1)] = \
                        computation_theta_i.compute_dml_theta(g_i_pre_given_z_test, yf_i_test, g_i_pre_given_zi_test, prob_i_given_zi_test)                    
                    exp_df_test.loc[training_name_g + ',' + training_name_m + '_e', 'DML_' + str(i + 1)] = \
                        np.abs(exp_df_test.loc[training_name_g + ',' + training_name_m, 'DML_' + str(i + 1)]/exp_df_test.loc['true', 'DR_' + str(i + 1)]-1)
                    exp_df_test.loc[training_name_g + ',' + training_name_m + '_e', 'trim_DML_' + str(i + 1)] = \
                        np.abs(exp_df_test.loc[training_name_g + ',' + training_name_m, 'trim_DML_' + str(i + 1)]/exp_df_test.loc['true', 'DR_' + str(i + 1)]-1)
                    '''compute rcl theta'''
                    for r in range(2, max_r+1):
                        for k in range(1, r+1):
                            exp_df.loc[training_name_g + ',' + training_name_m, str(r)+'&'+str(k)+'_'+ str(i + 1)] = \
                                computation_theta_i.compute_highorder_theta(g_i_pre_given_z_train, res_nu_i_train, yi_minus_gi_given_z_list_train,
                                                                      r, k, E_res_nu_i_list_train, E_nu_i_list_train, method)
                            exp_df.loc[training_name_g + ',' + training_name_m + '_e', str(r) + '&' + str(k) + '_' + str(i + 1)] = \
                                np.abs(exp_df.loc[training_name_g + ',' + training_name_m, str(r) + '&' + str(k) + '_' + str(i + 1)]/exp_df.loc['true', 'DR_' + str(i + 1)]-1)
                            exp_df_test.loc[training_name_g + ',' + training_name_m, str(r)+'&'+str(k)+'_'+ str(i + 1)] = \
                                computation_theta_i.compute_highorder_theta(g_i_pre_given_z_test, res_nu_i_test, yi_minus_gi_given_z_list_test,
                                                                      r, k, E_res_nu_i_list_test, E_nu_i_list_test, method)
                            exp_df_test.loc[training_name_g + ',' + training_name_m + '_e', str(r) + '&' + str(k) + '_' + str(i + 1)] = \
                                np.abs(exp_df_test.loc[training_name_g + ',' + training_name_m, str(r) + '&' + str(k) + '_' + str(i + 1)] / exp_df_test.loc['true', 'DR_' + str(i + 1)] - 1)
        exp_df.to_csv('./exp/'+str(N)+"_exp_" + str(m)+'_train'+ '.csv')
        exp_df_test.to_csv('./exp/' + str(N)+"_exp_" + str(m)+'_test'+'.csv')
        '''ATE results'''
        ATE_df = computation_theta_i.compute_ATE(ATE_df, exp_df, num_D, max_r, model_g_methods, model_m_methods)
        ATE_df_test = computation_theta_i.compute_ATE(ATE_df_test, exp_df_test, num_D, max_r, model_g_methods, model_m_methods)
        ATE_df.to_csv('./ATE/'+str(N)+"_ATE_" + str(m)+'_train'+'.csv')
        ATE_df_test.to_csv('./ATE/' + str(N)+"_ATE_" + str(m)+'_test'+ '.csv')
        keras.backend.clear_session()
    elapsed = time.time() - t
    print('The time taken is ' + str(elapsed))
if __name__ == '__main__':
    max_r = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    N=747
    M0=0
    M=1000
    run_ihdp(M0=M0, M=M, dataset_path='./dataset_1-1000')