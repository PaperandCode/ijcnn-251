import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from functools import partial
def make_exp_headings(max_r, num_D):
    expectation_headings = []
    for i in range(0, num_D):
        for estimator in ['DR', 'IPW', 'AIPW', 'DML', 'trim_DML']:
            expectation_headings.append(estimator + '_' + str(i + 1))
        for r in range(2, max_r+1):
            for k in range(1, r+1):
                expectation_headings.append(str(r)+'&'+str(k)+'_'+str(i+1))
    return expectation_headings

def make_ATE_headings(max_r, num_D):
    ATE_headings = []
    for i in range(0, num_D):
        for j in range(i+1, num_D):
            for estimator in ['DR', 'IPW', 'AIPW', 'DML', 'trim_DML']:
                ATE_headings.append(estimator + '_' + str(i + 1) + ',' + str(j + 1))
            for r in range(2, max_r + 1):
                for k in range(1, r + 1):
                    ATE_headings.append(str(r)+'&'+str(k)+'_'+str(i+1)+','+str(j+1))
    return ATE_headings

def make_saving_index(model_g_methods, model_m_methods):
    savinglist = ['true']
    for g_name in model_g_methods:
        for m_name in model_m_methods:
                savinglist.append(g_name+','+m_name)
                savinglist.append(g_name+','+m_name+'_e')
    return savinglist
def make_saving_index_for_NET(model_net_methods):
    savinglist = ['true']
    for name in model_net_methods:
        savinglist.append(name)
        savinglist.append(name+'_e')
    return savinglist
def get_simulation_data(N, experiment_index):
    data_load = joblib.load('./dataset/'+str(N)+'_experiment_'+str(experiment_index))
    z = data_load[0]
    d = data_load[1]
    y_f = data_load[2]
    itrain, itest = data_load[6], data_load[7]
    g1, g2, g3 = data_load[8], data_load[9], data_load[10]
    prob1, prob2, prob3 = data_load[11], data_load[12], data_load[13]
    return z, d, y_f, g1, g2, g3, itrain, itest, prob1, prob2, prob3
def get_acic_data(experiment_index):
    loader = joblib.load('./dataset/'+'experiment_'+str(experiment_index))
    true_ATE, df = loader[0], loader[1]
    t = df['z'].values
    y = df[['y0','y1']].values
    y_f = df['y'].values
    x = df.iloc[:, 5:].values
    itrain, itest = train_test_split(np.arange(len(t)), test_size=0.3, shuffle=False)
    return x, y, y_f.reshape(-1, 1), t.reshape(-1, 1), itrain, itest
def get_ihdp_data(dataset_path, experiment_index):
    train_file_path = dataset_path+'/ihdp_npci_1-1000.train.npz'
    test_file_path = dataset_path+'/ihdp_npci_1-1000.test.npz'

    index = experiment_index  # np.random.RandomState(seed).randint(0, 1000)
    train_set = np.load(train_file_path)
    test_set = np.load(test_file_path)
    def get_field(name):
        return np.concatenate([train_set[name][..., index],
                               test_set[name][..., index]], axis=0)
    n_train, n_test = train_set["x"].shape[0], test_set["x"].shape[0]
    train_indices = np.arange(0, n_train)
    test_indices = np.arange(n_train, n_train + n_test)
    y_f, y_cf = get_field("yf"), get_field("ycf")
    t = get_field("t").astype(int)
    y = np.zeros((n_train + n_test, 2))
    y[t == 0, 0] = y_f[t == 0]
    y[t == 0, 1] = y_cf[t == 0]
    y[t == 1, 1] = y_f[t == 1]
    y[t == 1, 0] = y_cf[t == 1]

    mu0, mu1 = get_field("mu0"), get_field("mu1")
    x = get_field("x")
    x = np.column_stack([np.expand_dims(np.arange(n_train + n_test) + 1, axis=-1), x])
    x = x[:, 1:]
    return x, y, y_f.reshape(-1, 1), t.reshape(-1, 1), mu0.reshape(-1, 1), mu1.reshape(-1, 1), train_indices, test_indices

def prepare_set_A(idx_i, yf, g_i_pre_given_z, N, sample_times):
    yi_minus_gi_given_z = yf - g_i_pre_given_z
    yi_minus_gi_given_zi = yi_minus_gi_given_z[idx_i]
    idx_not_i = list(set(list(range(0, len(yi_minus_gi_given_z)))) - set(idx_i.tolist()))
    newfun = partial(make_yi_minus_gi_given_z, np.squeeze(yi_minus_gi_given_z.copy()), yi_minus_gi_given_zi, idx_not_i)
    sampletimes = range(0, sample_times)
    yi_minus_gi_given_z_list = np.array(list(map(newfun, sampletimes))).T
    return yi_minus_gi_given_z_list

def make_yi_minus_gi_given_z(yi_minus_gi_given_z, yi_minus_gi_given_zi, idx_not_i, sampletime):
    np.random.seed(sampletime)
    cf_xi = np.random.choice(np.squeeze(yi_minus_gi_given_zi.copy()), len(idx_not_i))
    temp = yi_minus_gi_given_z.copy()
    temp[idx_not_i] = cf_xi
    return temp

def prepare_g_related(model_g_i, yf, z, idx_i, sample_times):
    g_i_pre_given_z = model_g_i.predict_proba(z)[:, 1].reshape(-1, 1)
    # g_i_pre_given_z = model_g_i.predict(z).reshape(-1, 1)
    g_i_pre_given_zi = g_i_pre_given_z[idx_i]
    yi_minus_gi_given_z_list = prepare_set_A(idx_i, yf, g_i_pre_given_z, z.shape[0], sample_times)
    return g_i_pre_given_z, g_i_pre_given_zi, yi_minus_gi_given_z_list

def prepare_m_related(prob_all, true_prob_i, idx_i, i, N, max_r, method):
    prob_i_given_z = prob_all[:, i].reshape(-1, 1)
    indicator_for_i = np.zeros((N, 1))
    indicator_for_i[idx_i] = 1
    res_m = indicator_for_i - prob_i_given_z
    E_res_m_list = compute_E_max_r(res_m, max_r)
    if method=='true':
        true_res_m = indicator_for_i - true_prob_i
        E_nu_list = compute_E_max_r(true_res_m, max_r)
        return res_m, E_res_m_list, E_nu_list
    else:
        return res_m, E_res_m_list, 0

def compute_E_max_r(x, max_r):
    '''moment_1 to moment_max_r'''
    value = []
    for r in range(0, max_r):
        value.append(np.mean(x**(r+1)))
    return value
def get_twins_data(dataset_path, experiment_index):
    loader = joblib.load(dataset_path+'/twins_' + str(experiment_index))
    return loader[0],loader[1],loader[2],loader[3],loader[4],loader[5],loader[6],loader[7]
if __name__ == '__main__':
    get_acic_data(28)