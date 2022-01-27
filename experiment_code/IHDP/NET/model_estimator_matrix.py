import numpy as np
import os
import pandas as pd
import scipy.stats as stats
def make_model_estimator_matrix(filename, model_list, estimator_list, error_name='weight_rel_error_avg'):
    matrix_df = pd.DataFrame(index=model_list)
    df_raw = pd.read_csv('./'+filename+'.csv', index_col=0)
    for model_name in model_list:
        for estimator_name in estimator_list:
            matrix_df.loc[model_name, estimator_name] = df_raw.loc[model_name + '_' + estimator_name, error_name]
            # if estimator_name not in ['0', '1']:
            #     matrix_df.loc[model_name, estimator_name+'-0'] = 1-matrix_df.loc[model_name, estimator_name]/matrix_df.loc[model_name, '0']
            #     matrix_df.loc[model_name, estimator_name + '-1'] = 1 - matrix_df.loc[model_name, estimator_name] / matrix_df.loc[model_name, '1']
    matrix_df.to_csv(filename+'_'+error_name+'_model_vs_estimator.csv')
def combine_general_net(file1, file2, output):
    df1 = pd.read_csv(file1, index_col=0)
    df2 = pd.read_csv(file2, index_col=0)
    df3 = pd.concat([df1, df2], axis=0)
    df3.to_csv(output)
if __name__ == '__main__':
    model_list = ['tarnet','dragonnet']
    # estimator_list = ['0', '1', '2&1', '2&2', '3&1', '3&2', '3&3', '4&1', '4&2', '4&3','4&4', '5&1', '5&2', '5&3', '5&4', '5&5', '6&1', '6&2', '6&3', '6&4', '6&5', '6&6',  '7&1', '7&2', '7&3', '7&4', '7&5', '7&6', '7&7']
    estimator_list = ['DR', 'IPW', 'AIPW', 'DML', 'trim_DML', '2&1', '2&2']

    # combine_general_net(file1='./general_result/ATE_summary_error_stats_1000_test_all.csv',
    #                     file2='./NET_result/ATE_summary_error_stats_1000_test_all.csv',
    #                     output='./ihdp_all.csv')
    for TE_name in ['ATE']:
        make_model_estimator_matrix(filename=TE_name+'_summary_error_stats_runninglist_test', model_list=model_list, estimator_list=estimator_list, error_name='weight_rel_error_avg')
        make_model_estimator_matrix(filename=TE_name+'_summary_error_stats_runninglist_test', model_list=model_list, estimator_list=estimator_list,
                                    error_name='weight_rel_error_std')