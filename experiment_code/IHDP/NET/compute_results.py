import numpy as np
import os
import pandas as pd
import scipy.stats as stats
from utils import *
def make_model_dict(model_net_methods, imb_funs=None):
    model_dict = {}
    for model_name in model_net_methods:
        if imb_funs == None:
            model_dict[model_name] = 0
        else:
            for imb_fun in imb_funs:
                model_dict[model_name+'_'+imb_fun] = 0
    return model_dict
def make_ATE_list(num_D):
    list = []
    for i in range(1, num_D + 1):
        for j in range(i+1, num_D + 1):
            list.append(str(i)+','+str(j))
    return list
def make_exp_list(num_D):
    list = []
    for i in range(1, num_D+1):
        list.append(str(i))
    return list

def compute_exp_result(result_path, runninglist):
    if not os.path.isdir(result_path+'/stats_result'):
        os.mkdir(result_path+'/stats_result')
    exp_path = result_path+'/exp'
    for drawset in ['train', 'test']:
        error_df = pd.DataFrame()
        rel_error_df = pd.DataFrame()
        true_df = pd.DataFrame()
        for m in runninglist:
            df = pd.read_csv(exp_path+'/'+str(N) + '_exp_'+str(m)+'_' + drawset+'.csv', index_col=0)
            for i in range(1, num_D+1):
                true_df.loc[m, 'true_' + str(i)] = df.loc['true', 'DR_' + str(i)]
            for model_name in model_net_dict.keys():
                for col in df.columns.tolist():
                    error_df.loc[m, model_name+'_'+col] = df.loc[model_name, col]-df.loc['true', col]
                    rel_error_df.loc[m, model_name+'_'+col] = np.abs(df.loc[model_name, col]/df.loc['true', col]-1)
        true_df.to_csv(result_path+'/stats_result/exp_true_list_' + 'runninglist' + '_' + drawset + '.csv')
        error_df.to_csv(result_path+'/stats_result/exp_error_list_'+'runninglist'+'_'+drawset+'.csv')
        rel_error_df.to_csv(result_path+'/stats_result/exp_rel_error_list_'+'runninglist'+'_'+drawset+'.csv')
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
        atelist = make_exp_list(num_D)
        for model_name in model_net_dict.keys():
            for estimator in ['DR', 'IPW', 'AIPW', 'DML', 'trim_DML']:
                abs_sum_array = 0
                rel_sum_array = 0
                sum_abs_array = 0
                sum_rel_array = 0
                weight_sum = 0
                count = 0
                for ate_name in atelist:
                    weight_sum = weight_sum + np.abs(true_df['true_'+ate_name])
                    abs_sum_array = abs_sum_array + np.abs(true_df['true_'+ate_name])*np.abs(error_df[model_name+'_'+estimator+'_'+ate_name])
                    rel_sum_array = rel_sum_array + np.abs(true_df['true_' + ate_name]) * rel_error_df[model_name + '_' + estimator + '_' + ate_name]
                    sum_abs_array = sum_abs_array + np.abs(error_df[model_name+'_'+estimator+'_'+ate_name])
                    sum_rel_array = sum_rel_array + rel_error_df[model_name + '_' + estimator + '_' + ate_name]
                    count = count + 1
                abs_weight_sum_array = abs_sum_array/weight_sum
                rel_weight_sum_array = rel_sum_array/weight_sum
                abs_mean_sum_array = sum_abs_array/count
                rel_mean_sum_array = sum_rel_array/count
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'weight_abs_error_avg'] = np.mean(
                    abs_weight_sum_array)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'weight_abs_error_std'] = np.std(
                    abs_weight_sum_array, ddof=1)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'mean_abs_error_avg'] = np.mean(
                    abs_mean_sum_array)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'mean_abs_error_std'] = np.std(
                    abs_mean_sum_array, ddof=1)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'weight_rel_error_avg'] = np.mean(
                    rel_weight_sum_array)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'weight_rel_error_std'] = np.std(
                    rel_weight_sum_array, ddof=1)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'mean_rel_error_avg'] = np.mean(
                    rel_mean_sum_array)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'mean_rel_error_std'] = np.std(
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
                    for ate_name in atelist:
                        weight_sum = weight_sum + np.abs(true_df['true_'+ate_name])
                        abs_sum_array = abs_sum_array + np.abs(true_df['true_'+ate_name])*np.abs(error_df[model_name+'_'+str(r)+'&'+str(k)+'_'+ate_name])
                        rel_sum_array = rel_sum_array + np.abs(true_df['true_' + ate_name]) * rel_error_df[model_name + '_' + str(r)+'&'+str(k) + '_' + ate_name]
                        sum_abs_array = sum_abs_array + np.abs(error_df[model_name+'_'+str(r)+'&'+str(k)+'_'+ate_name])
                        sum_rel_array = sum_rel_array + rel_error_df[model_name + '_' + str(r)+'&'+str(k) + '_' + ate_name]
                        count = count + 1
                    abs_weight_sum_array = abs_sum_array/weight_sum
                    rel_weight_sum_array = rel_sum_array/weight_sum
                    abs_mean_sum_array = sum_abs_array/count
                    rel_mean_sum_array = sum_rel_array/count
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'weight_abs_error_avg'] = np.mean(
                        abs_weight_sum_array)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'weight_abs_error_std'] = np.std(
                        abs_weight_sum_array, ddof=1)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'mean_abs_error_avg'] = np.mean(
                        abs_mean_sum_array)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'mean_abs_error_std'] = np.std(
                        abs_mean_sum_array, ddof=1)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'weight_rel_error_avg'] = np.mean(
                        rel_weight_sum_array)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'weight_rel_error_std'] = np.std(
                        rel_weight_sum_array, ddof=1)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'mean_rel_error_avg'] = np.mean(
                        rel_mean_sum_array)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'mean_rel_error_std'] = np.std(
                        rel_mean_sum_array, ddof=1)

        summary_error_stats_df.to_csv(result_path+'/exp_'+'summary_error_stats_'+'runninglist'+'_'+drawset+'.csv')
        error_stats_df.to_csv(result_path+'/stats_result/'+'exp_'+'error_stats_'+'runninglist'+'_'+drawset+'.csv')
        rel_error_stats_df.to_csv(result_path+'/stats_result/'+'exp_'+'rel_error_stats_'+'runninglist'+'_'+drawset+'.csv')
def compute_ATE_result(result_path, runninglist):
    if not os.path.isdir(result_path+'/stats_result'):
        os.mkdir(result_path+'/stats_result')
    ATE_path = result_path+'/ATE'
    for drawset in ['train', 'test']:
        error_df = pd.DataFrame()
        rel_error_df = pd.DataFrame()
        true_df = pd.DataFrame()
        for m in runninglist:
            df = pd.read_csv(ATE_path+'/'+str(N) + '_ATE_'+str(m)+'_' + drawset+'.csv', index_col=0)
            for i in range(1, num_D+1):
                for j in range(i+1, num_D+1):
                    true_df.loc[m, 'true_' + str(i) + ',' + str(j)] = df.loc['true', 'DR_' + str(i) + ',' + str(j)]
            for model_name in model_net_dict.keys():
                for col in df.columns.tolist():
                    error_df.loc[m, model_name+'_'+col] = df.loc[model_name, col]-df.loc['true', col]
                    rel_error_df.loc[m, model_name+'_'+col] = np.abs(df.loc[model_name, col]/df.loc['true', col]-1)
        true_df.to_csv(result_path+'/stats_result/ATE_true_list_' + 'runninglist' + '_' + drawset + '.csv')
        error_df.to_csv(result_path+'/stats_result/ATE_error_list_'+'runninglist'+'_'+drawset+'.csv')
        rel_error_df.to_csv(result_path+'/stats_result/ATE_rel_error_list_'+'runninglist'+'_'+drawset+'.csv')
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
        atelist = make_ATE_list(num_D)
        for model_name in model_net_dict.keys():
            '''compute 0 and 1 order result'''
            for estimator in ['DR', 'IPW', 'AIPW', 'DML', 'trim_DML']:
                abs_sum_array = 0
                rel_sum_array = 0
                sum_abs_array = 0
                sum_rel_array = 0
                weight_sum = 0
                count = 0
                for ate_name in atelist:
                    weight_sum = weight_sum + np.abs(true_df['true_'+ate_name])
                    abs_sum_array = abs_sum_array + np.abs(true_df['true_'+ate_name])*np.abs(error_df[model_name+'_'+estimator+'_'+ate_name])
                    rel_sum_array = rel_sum_array + np.abs(true_df['true_' + ate_name]) * rel_error_df[model_name + '_' + estimator + '_' + ate_name]
                    sum_abs_array = sum_abs_array + np.abs(error_df[model_name+'_'+estimator+'_'+ate_name])
                    sum_rel_array = sum_rel_array + rel_error_df[model_name + '_' + estimator + '_' + ate_name]
                    count = count + 1
                abs_weight_sum_array = abs_sum_array/weight_sum
                rel_weight_sum_array = rel_sum_array/weight_sum
                abs_mean_sum_array = sum_abs_array/count
                rel_mean_sum_array = sum_rel_array/count
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'weight_abs_error_avg'] = np.mean(
                    abs_weight_sum_array)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'weight_abs_error_std'] = np.std(
                    abs_weight_sum_array, ddof=1)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'mean_abs_error_avg'] = np.mean(
                    abs_mean_sum_array)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'mean_abs_error_std'] = np.std(
                    abs_mean_sum_array, ddof=1)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'weight_rel_error_avg'] = np.mean(
                    rel_weight_sum_array)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'weight_rel_error_std'] = np.std(
                    rel_weight_sum_array, ddof=1)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'mean_rel_error_avg'] = np.mean(
                    rel_mean_sum_array)
                summary_error_stats_df.loc[
                    model_name + '_' + estimator, 'mean_rel_error_std'] = np.std(
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
                    for ate_name in atelist:
                        weight_sum = weight_sum + np.abs(true_df['true_'+ate_name])
                        abs_sum_array = abs_sum_array + np.abs(true_df['true_'+ate_name])*np.abs(error_df[model_name+'_'+str(r)+'&'+str(k)+'_'+ate_name])
                        rel_sum_array = rel_sum_array + np.abs(true_df['true_' + ate_name]) * rel_error_df[model_name + '_' + str(r)+'&'+str(k) + '_' + ate_name]
                        sum_abs_array = sum_abs_array + np.abs(error_df[model_name+'_'+str(r)+'&'+str(k)+'_'+ate_name])
                        sum_rel_array = sum_rel_array + rel_error_df[model_name + '_' + str(r)+'&'+str(k) + '_' + ate_name]
                        count = count + 1
                    abs_weight_sum_array = abs_sum_array/weight_sum
                    rel_weight_sum_array = rel_sum_array/weight_sum
                    abs_mean_sum_array = sum_abs_array/count
                    rel_mean_sum_array = sum_rel_array/count
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'weight_abs_error_avg'] = np.mean(
                        abs_weight_sum_array)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'weight_abs_error_std'] = np.std(
                        abs_weight_sum_array, ddof=1)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'mean_abs_error_avg'] = np.mean(
                        abs_mean_sum_array)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'mean_abs_error_std'] = np.std(
                        abs_mean_sum_array, ddof=1)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'weight_rel_error_avg'] = np.mean(
                        rel_weight_sum_array)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'weight_rel_error_std'] = np.std(
                        rel_weight_sum_array, ddof=1)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'mean_rel_error_avg'] = np.mean(
                        rel_mean_sum_array)
                    summary_error_stats_df.loc[model_name + '_' + str(r)+'&'+str(k), 'mean_rel_error_std'] = np.std(
                        rel_mean_sum_array, ddof=1)

        summary_error_stats_df.to_csv(result_path+'/ATE_'+'summary_error_stats_'+'runninglist'+'_'+drawset+'.csv')
        error_stats_df.to_csv(result_path+'/stats_result/'+'ATE_'+'error_stats_'+'runninglist'+'_'+drawset+'.csv')
        rel_error_stats_df.to_csv(result_path+'/stats_result/'+'ATE_'+'rel_error_stats_'+'runninglist'+'_'+drawset+'.csv')


if __name__ == '__main__':
    imb_funs = ['wass']
    model_net_methods = ['tarnet', 'dragonnet']
    model_net_dict = make_model_dict(model_net_methods)
    N=747
    num_D = 2
    max_r = 2
    runninglist =list(range(0, 1000))
    pathname=''
    # compute_ITE_result(result_path='./' + pathname, runninglist=runninglist)
    # compute_exp_result(result_path='./' + pathname, runninglist=runninglist)
    # compute_cond_exp_result(result_path='./' + pathname, runninglist=runninglist)
    compute_ATE_result(result_path='./' + pathname, runninglist=runninglist)
    # compute_ATTE_result(result_path='./' + pathname, runninglist=runninglist)