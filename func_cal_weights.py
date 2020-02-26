import numpy as np

def calc_wd(dis_u, model_names, sigmaS = X):
        wd = np.empty((len(model_names)))
        S = np.exp( - ((dis_u / sigmaS2) ** 2))
        for j in range(len(model_names)):
            S_tmp = np.copy(S[j, :]) 
            S_tmp[j] = 0
            Ru = 1 + (np.sum(S_tmp))
            wd[j] = 1 / Ru
            del(S_tmp)
            del(Ru)
    return wd



def calc_wn(delta_q, model_names, sigmaD2 = X):
        wn = np.empty((len(model_names)))
        for j in range(len(model_names)):
            wn[j] =  np.exp( - ((delta_q[j] / sigmaD2) ** 2))

    return wn



def calc_weights(wu, wq, model_names):
    w_prod = np.empty((len(wq)))
    weights = {}
    for j in range(len(model_names)):
        tmp_wu = wu[j]
        w_prod[j] = wq[j] * tmp_wu

    wu_wq_sum = np.nansum(w_prod)
    for j in range(len(model_names)):
        ref = model_names[j]
        if wu_wq_sum != 0.0:
            weights[ref] =  w_prod[j] / wu_wq_sum
        else:
            weights[ref] = w_prod[j] * 0.0

    return weights
