import numpy as np
from scipy.special import expit
import os
import joblib
def generate_twins(raw_dataset_path, new_dataset_path, train_rate=0.8, M0=0, M=1, seed=0):
    np.random.seed(seed)
    # Load original data (11400 patients, 30 features, 2 dimensional potential outcomes)
    ori_data = np.loadtxt(raw_dataset_path + "/Twin_data.csv", delimiter=",", skiprows=1)

    # Define features
    x = ori_data[:, :30]
    no, dim = x.shape

    # Define potential outcomes
    potential_y = ori_data[:, 30:]
    # Die within 1 year = 1, otherwise = 0
    potential_y = np.array(potential_y < 9999, dtype=float)

    ## Assign treatment
    coef = np.random.uniform(-0.01, 0.01, size=[dim, 1])
    prob_temp = 1/(1+np.exp(-np.matmul(x, coef)))
    # prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.01, size=[no, 1]))

    # prob_t = prob_temp / (2 * np.mean(prob_temp))
    # prob_t[prob_t > 1] = 1

    t = np.random.binomial(1, prob_temp, [no, 1])
    t = t.reshape([no, ])

    # indicator = np.zeros((1, len(t)))
    # idx = np.where(t==1)[0]
    # indicator[0, idx] = 1
    # indicator = indicator.reshape(-1, 1)
    # list = np.mean(indicator - prob_t)
    ## Define observable outcomes
    y = np.transpose(t) * potential_y[:, 1] + np.transpose(1 - t) * potential_y[:, 0]
    y = np.reshape(np.transpose(y), [no, ])
    for m in range(M0, M):
        # np.random.seed(m)
        ## Train/test division
        idx = np.random.permutation(no)
        train_idx = idx[:int(train_rate * no)]
        test_idx = idx[int(train_rate * no):]

        y = y.reshape(-1, 1)
        t = t.reshape(-1, 1)
        mu0 = potential_y[:, 0].reshape(-1, 1)
        mu1 = potential_y[:, 1].reshape(-1, 1)
        savelist = [x, potential_y, y, t, mu0, mu1, train_idx, test_idx]
        joblib.dump(savelist, new_dataset_path+'/twins_'+str(m))
if __name__ == '__main__':
    target_twins_dataset_path = r'./raw_twins_data'
    if not os.path.isdir('./twins_dataset'):
        os.mkdir('./twins_dataset')
    new_dataset_path = './twins_dataset'
    generate_twins(raw_dataset_path=target_twins_dataset_path, new_dataset_path=new_dataset_path, train_rate=0.8, M0=0, M=100, seed=1)