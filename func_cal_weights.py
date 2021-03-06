# The weighting method is based on some previous study. Knutti et al. 2017, GRL, Lorenz et al. 2017, JGR and Brunner et al. 2019, ERL.
# We refer to the published code, more detail can be found from https://github.com/lukasbrunner/ClimWIP and https://github.com/ruthlorenz/weighting_CMIP
# here show a example
import numpy as np

def calc_wu(dis_u, model_names, sigmaS = X):
        wu = np.empty((len(model_names)))
        S = np.exp( - ((dis_u / sigmaS2) ** 2))
        for j in range(len(model_names)):
            S_tmp = np.copy(S[j, :]) 
            S_tmp[j] = 0
            Ru = 1 + (np.sum(S_tmp))
            wu[j] = 1 / Ru
            del(S_tmp)
            del(Ru)
    return wu



def calc_wq(delta_q, model_names, sigmaD2 = X):
        wq = np.empty((len(model_names)))
        for j in range(len(model_names)):
            wq[j] =  np.exp( - ((delta_q[j] / sigmaD2) ** 2))

    return wq



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
