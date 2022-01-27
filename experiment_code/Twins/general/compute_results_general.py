import numpy as np
import os
import pandas as pd
import scipy.stats as stats

# model_g_methods = ['LASSO', 'RF', 'MLP']
# model_m_methods = ['LR', 'RF', 'MLP']
# num_D = 2
# max_r = 5
def make_ATE_list(num_D):
    list = []
    for i in range(1, num_D + 1):
        for j in range(i+1, num_D + 1):
            list.append(str(i)+','+str(j))
    return list
def make_ATTE_list(num_D):
    list = []
    for j in range(1, num_D + 1):
        for i in range(1, num_D + 1):
            for l in range(i + 1, num_D + 1):
                list.append(str(i) + ',' + str(l) + '|' + str(j))
    return list
def compute_ATE_df(result_path, M0, M, drawset):
    if not os.path.isdir(result_path+'/stats_result'):
        os.mkdir(result_path+'/stats_result')
    ATE_path = result_path + './ATE'
    error_df = pd.DataFrame()
    rel_error_df = pd.DataFrame()
    true_df = pd.DataFrame()
    for m in range(M0, M):
        df = pd.read_csv(ATE_path + '/' + '11400_ATE_' + str(m) + '_' + drawset + '.csv', index_col=0)
        for i in range(1, num_D + 1):
            for j in range(i + 1, num_D + 1):
                true_df.loc[m, 'true_' + str(i) + ',' + str(j)] = df.loc['true', 'DR_' + str(i) + ',' + str(j)]
                true_df.loc[m, 'true_' + str(i) + ',' + str(j)] = df.loc['true', 'DR_' + str(i) + ',' + str(j)]
        for g_name in model_g_methods:
            for m_name in model_m_methods:
                for col in df.columns.tolist():
                    error_df.loc[m, g_name + ',' + m_name + '_' + col] = df.loc[g_name + ',' + m_name, col] - df.loc['true', col]
                    rel_error_df.loc[m, g_name + ',' + m_name + '_' + col] = np.abs(df.loc[g_name + ',' + m_name, col] / df.loc['true', col] - 1)
    true_df.to_csv(result_path+'./stats_result/ATE_true_list_' + str(M0) + '-' + str(M) + '_' + drawset + '.csv')
    error_df.to_csv(result_path+'./stats_result/ATE_error_list_' + str(M0) + '-' + str(M) + '_' + drawset + '.csv')
    rel_error_df.to_csv(result_path+'./stats_result/ATE_rel_error_list_' + str(M0) + '-' + str(M) + '_' + drawset + '.csv')
def compute_TE_df(TE_name, result_path, M0, M, drawset):
    telist = 0
    if TE_name == 'ATE':
        telist = make_ATE_list(num_D)
    if TE_name == 'ATTE':
        telist = make_ATTE_list(num_D)
    if not os.path.isdir(result_path+'/stats_result'):
        os.mkdir(result_path+'/stats_result')
    path = result_path + './'+TE_name
    error_df = pd.DataFrame()
    rel_error_df = pd.DataFrame()
    true_df = pd.DataFrame()
    for m in range(M0, M):
        df = pd.read_csv(path + '/' + '11400_'+TE_name+'_' + str(m) + '_' + drawset + '.csv', index_col=0)
        for te_name in telist:
            true_df.loc[m, 'true_' + te_name] = df.loc['true', 'DR_' + te_name]
            true_df.loc[m, 'true_' + te_name] = df.loc['true', 'DR_' + te_name]
        for g_name in model_g_methods:
            for m_name in model_m_methods:
                for col in df.columns.tolist():
                    error_df.loc[m, g_name + ',' + m_name + '_' + col] = df.loc[g_name + ',' + m_name, col] - df.loc['true', col]
                    rel_error_df.loc[m, g_name + ',' + m_name + '_' + col] = np.abs(df.loc[g_name + ',' + m_name, col] / df.loc['true', col] - 1)
    true_df.to_csv(result_path+'./stats_result/'+TE_name+'_true_list_' + str(M0) + '-' + str(M) + '_' + drawset + '.csv')
    error_df.to_csv(result_path+'./stats_result/'+TE_name+'_error_list_' + str(M0) + '-' + str(M) + '_' + drawset + '.csv')
    rel_error_df.to_csv(result_path+'./stats_result/'+TE_name+'_rel_error_list_' + str(M0) + '-' + str(M) + '_' + drawset + '.csv')
def compute_inf_idx(TE_name, result_path, M0, M, drawset):
    error_df = pd.read_csv(result_path+'./stats_result/'+TE_name+'_error_list_' + str(M0) + '-' + str(M) + '_' + drawset + '.csv', index_col=0)
    error_df[np.abs(error_df) > 1000] = np.nan
    idx_nan = np.unique(np.where(np.isnan(error_df))[0])
    return idx_nan
def compute_TE_result(TE_name, result_path, M0, M, drawset, delete_name, idx_nan):
    if not os.path.isdir(result_path+'./stats_result'):
        os.mkdir(result_path+'./stats_result')
    error_df = pd.read_csv(result_path+'./stats_result/'+TE_name+'_error_list_' + str(M0) + '-' + str(M) + '_' + drawset + '.csv', index_col=0)
    true_df = pd.read_csv(result_path+'./stats_result/'+TE_name+'_true_list_' + str(M0) + '-' + str(M) + '_' + drawset + '.csv', index_col=0)
    rel_error_df = pd.read_csv(result_path+'./stats_result/'+TE_name+'_rel_error_list_' + str(M0) + '-' + str(M) + '_' + drawset + '.csv', index_col=0)
    if delete_name=='del_inf':
        true_df = true_df.drop(idx_nan).reset_index(drop=True)
        error_df = error_df.drop(idx_nan).reset_index(drop=True)
        rel_error_df = rel_error_df.drop(idx_nan).reset_index(drop=True)
    '''stats dataframe'''
    error_df_stats = ['mean_abs', 'mean_square', 'mean', 'variance', 'skewness', 'kurtosis']
    rel_error_df_stats = ['mean', 'variance', 'skewness', 'kurtosis']
    summary_error_df_stats = ['weight_abs_error_avg', 'mean_abs_error_avg', 'weight_rel_error_avg', 'mean_rel_error_avg']
    error_stats_df = pd.DataFrame(index=error_df_stats, columns=error_df.columns)
    rel_error_stats_df = pd.DataFrame(index=rel_error_df_stats, columns=rel_error_df.columns)
    summary_error_stats_df = pd.DataFrame()
    for col in error_df.columns.tolist():
        error_stats_df.loc['mean',col] = np.mean(error_df[col])
        error_stats_df.loc['variance',col] = np.var(error_df[col])
        error_stats_df.loc['skewness',col] = stats.skew(error_df[col])
        error_stats_df.loc['kurtosis',col] = stats.kurtosis(error_df[col])
        error_stats_df.loc['mean_abs',col] = np.mean(np.abs(error_df[col])) # MAE
        error_stats_df.loc['mean_square',col] = np.mean(np.power(error_df[col], 2))  # MAE
    for col in rel_error_df.columns.tolist():
        rel_error_stats_df.loc['mean'][col] = np.mean(rel_error_df[col])
        rel_error_stats_df.loc['variance'][col] = np.var(rel_error_df[col])
        rel_error_stats_df.loc['skewness'][col] = stats.skew(rel_error_df[col])
        rel_error_stats_df.loc['kurtosis'][col] = stats.kurtosis(rel_error_df[col])
    telist = 0
    if TE_name == 'ATE':
        telist = make_ATE_list(num_D)
    if TE_name == 'ATTE':
        telist = make_ATTE_list(num_D)
    for g_name in model_g_methods:
        for m_name in model_m_methods:
            '''compute 0 and 1 order result'''
            for estimator in ['DR', 'IPW', 'AIPW', 'DML', 'trim_DML']:
                abs_sum_array = 0
                rel_sum_array = 0
                sum_abs_array = 0
                sum_rel_array = 0
                weight_sum = 0
                count = 0
                for te_name in telist:
                    weight_sum = weight_sum + np.abs(true_df['true_'+te_name])
                    abs_sum_array = abs_sum_array + np.abs(true_df['true_'+te_name])*np.abs(error_df[g_name + ',' + m_name+'_'+estimator+'_'+te_name])
                    rel_sum_array = rel_sum_array + np.abs(true_df['true_' + te_name]) * rel_error_df[g_name + ',' + m_name + '_' + estimator + '_' + te_name]
                    sum_abs_array = sum_abs_array + np.abs(error_df[g_name + ',' + m_name+'_'+estimator+'_'+te_name])
                    sum_rel_array = sum_rel_array + rel_error_df[g_name + ',' + m_name + '_' + estimator + '_' + te_name]
                    count = count + 1
                abs_weight_sum_array = abs_sum_array/weight_sum
                rel_weight_sum_array = rel_sum_array/weight_sum
                abs_mean_sum_array = sum_abs_array/count
                rel_mean_sum_array = sum_rel_array/count
                summary_error_stats_df.loc[g_name + ',' + m_name + '_' + estimator, 'weight_abs_error_avg'] = np.mean(
                    abs_weight_sum_array)
                summary_error_stats_df.loc[g_name + ',' + m_name + '_' + estimator, 'weight_abs_error_std'] = np.std(
                    abs_weight_sum_array, ddof=1)
                summary_error_stats_df.loc[g_name + ',' + m_name + '_' + estimator, 'mean_abs_error_avg'] = np.mean(
                    abs_mean_sum_array)
                summary_error_stats_df.loc[g_name + ',' + m_name + '_' + estimator, 'mean_abs_error_std'] = np.std(
                    abs_mean_sum_array, ddof=1)
                summary_error_stats_df.loc[g_name + ',' + m_name + '_' + estimator, 'weight_rel_error_avg'] = np.mean(
                    rel_weight_sum_array)
                summary_error_stats_df.loc[g_name + ',' + m_name + '_' + estimator, 'weight_rel_error_std'] = np.std(
                    rel_weight_sum_array, ddof=1)
                summary_error_stats_df.loc[g_name + ',' + m_name + '_' + estimator, 'mean_rel_error_avg'] = np.mean(
                    rel_mean_sum_array)
                summary_error_stats_df.loc[g_name + ',' + m_name + '_' + estimator, 'mean_rel_error_std'] = np.std(
                    rel_mean_sum_array, ddof=1)
            '''compute high order result'''
            for r in range(2, max_r+1):
                for k in range(1, r+1):
                    abs_sum_array = 0
                    rel_sum_array = 0
                    sum_abs_array = 0
                    sum_rel_array = 0
                    weight_sum = 0
                    count = 0
                    for te_name in telist:
                        weight_sum = weight_sum + np.abs(true_df['true_'+te_name])
                        abs_sum_array = abs_sum_array + np.abs(true_df['true_'+te_name])*np.abs(error_df[g_name + ',' + m_name+'_'+str(r)+'&'+str(k)+'_'+te_name])
                        rel_sum_array = rel_sum_array + np.abs(true_df['true_' + te_name]) * rel_error_df[g_name + ',' + m_name + '_' + str(r)+'&'+str(k) + '_' + te_name]
                        sum_abs_array = sum_abs_array + np.abs(error_df[g_name + ',' + m_name+'_'+str(r)+'&'+str(k)+'_'+te_name])
                        sum_rel_array = sum_rel_array + rel_error_df[g_name + ',' + m_name + '_' + str(r)+'&'+str(k) + '_' + te_name]
                        count = count + 1
                    abs_weight_sum_array = abs_sum_array/weight_sum
                    rel_weight_sum_array = rel_sum_array/weight_sum
                    abs_mean_sum_array = sum_abs_array/count
                    rel_mean_sum_array = sum_rel_array/count
                    summary_error_stats_df.loc[g_name + ',' + m_name + '_' + str(r)+'&'+str(k), 'weight_abs_error_avg'] = np.mean(
                        abs_weight_sum_array)
                    summary_error_stats_df.loc[g_name + ',' + m_name + '_' + str(r)+'&'+str(k), 'weight_abs_error_std'] = np.std(
                        abs_weight_sum_array, ddof=1)
                    summary_error_stats_df.loc[g_name + ',' + m_name + '_' + str(r)+'&'+str(k), 'mean_abs_error_avg'] = np.mean(
                        abs_mean_sum_array)
                    summary_error_stats_df.loc[g_name + ',' + m_name + '_' + str(r)+'&'+str(k), 'mean_abs_error_std'] = np.std(
                        abs_mean_sum_array, ddof=1)
                    summary_error_stats_df.loc[g_name + ',' + m_name + '_' + str(r)+'&'+str(k), 'weight_rel_error_avg'] = np.mean(
                        rel_weight_sum_array)
                    summary_error_stats_df.loc[g_name + ',' + m_name + '_' + str(r)+'&'+str(k), 'weight_rel_error_std'] = np.std(
                        rel_weight_sum_array, ddof=1)
                    summary_error_stats_df.loc[g_name + ',' + m_name + '_' + str(r)+'&'+str(k), 'mean_rel_error_avg'] = np.mean(
                        rel_mean_sum_array)
                    summary_error_stats_df.loc[g_name + ',' + m_name + '_' + str(r)+'&'+str(k), 'mean_rel_error_std'] = np.std(
                        rel_mean_sum_array, ddof=1)

        summary_error_stats_df.to_csv(result_path+'./'+TE_name+''+'_summary_error_stats_'+str(error_df.shape[0])+'_'+drawset+'_'+delete_name+'.csv')
        error_stats_df.to_csv(result_path+'./stats_result/'+TE_name+'_error_stats_'+str(error_df.shape[0])+'_'+drawset+'_'+delete_name+'.csv')
        rel_error_stats_df.to_csv(result_path+'./stats_result/'+TE_name+'_ATE_rel_error_stats_'+str(error_df.shape[0])+'_'+drawset+'_'+delete_name+'.csv')

if __name__ == '__main__':
    model_g_methods = ['LR', 'RF', 'MLP']
    model_m_methods = ['LR', 'RF', 'MLP']
    # model_g_methods = ['LR', 'RF']
    # model_m_methods = ['MLP']
    # model_NET_methods = ['tarnet', 'dragonnet']
    num_D = 2
    max_r = 2
    M0=0
    M=100
    for drawset in ['test']:
        '''compute error df'''
        compute_TE_df(TE_name='ATE', result_path='./', M0=M0, M=M, drawset=drawset)
        ATE_general_inf_idx = compute_inf_idx(TE_name='ATE', result_path='./', M0=M0,
                                                                  M=M, drawset=drawset)
        for delete_name in ['all', 'del_inf']:
            compute_TE_result \
                (TE_name='ATE', result_path='./', M0=M0, M=M, drawset=drawset, delete_name=delete_name,
                 idx_nan=ATE_general_inf_idx)
