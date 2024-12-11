# -----------------------------------------------------------------
# Librairies
# -----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.optimization import (MM_KL_IPL, 
                          MM_KL_new_w, 
                          MM_KL_new_w_regularized,
                          MM_LS_IPL,
                          MM_LS_new_w,
                          MM_LS_new_w_regularized)
from src.generation import (sampledistributionchoice)
from src.covariance_matrix import (SCM,
                                corr_phase,
                                bandw,
                                regul_linear)

# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------
def oneMonteCarlo(m, k, p, n, trueCov, sampledist, iter_max_MM, number_of_trials):
    
    Y = sampledistributionchoice(trueCov, p+k+m, n, sampledist)
    Y = np.asarray(Y, dtype = np.complex128)
    S_all = corr_phase(Y)
    Sigma_tilde = S_all # SCM
    Psi_tilde = np.abs(Sigma_tilde) 

    # past 1 => size = p
    Sigma_tilde_past1 = Sigma_tilde[0:p, 0:p] # dim : p, p
    new_past_Sigma_tilde1 = Sigma_tilde[p:p+k, 0:p] # dim : k, p
    new_Sigma_tilde1 = Sigma_tilde[p:p+k, p:p+k] # dim : k, k
    Psi_tilde_past1 = Psi_tilde[0:p, 0:p] # dim : p, p
    new_past_Psi_tilde1 = Psi_tilde[p:p+k, 0:p] # dim : k, p
    new_Psi_tilde1 = Psi_tilde[p:p+k, p:p+k] # dim : k, k

    # past 2 (of size = p+k)
    Sigma_tilde_past2 = Sigma_tilde[0:p+k, 0:p+k] # dim : p+k, p+k
    new_past_Sigma_tilde2 = Sigma_tilde[p+k:p+k+m, 0:p+k] # dim : m, p+k
    new_Sigma_tilde2 = Sigma_tilde[p+k:p+k+m, p+k:p+k+m] # dim : m, m
    Psi_tilde_past2 = Psi_tilde[0:p+k, 0:p+k] # dim : p+k, p+k
    new_past_Psi_tilde2 = Psi_tilde[p+k:p+k+m, 0:p+k] # dim : m, p+k
    new_Psi_tilde2 = Psi_tilde[p+k:p+k+m, p+k:p+k+m] # dim : m, m

    # ones
    w_new_ones1 = np.ones((k, 1) ,dtype=complex) 
    w_new_ones2 = np.ones((m, 1) ,dtype=complex) 

    # MM offline on past 1 (size = p)
    phases_offline_past1 = MM_LS_IPL(Sigma_tilde_past1, iter_max_MM)
    phases_offline_past1_0 = np.angle(phases_offline_past1[0])
    delta_phases_offline_past1 = np.angle(phases_offline_past1)- phases_offline_past1_0
    w_past1 = phases_offline_past1 # size = p
    delta_phases_offline_past1 = delta_phases_offline_past1.reshape((delta_phases_offline_past1.shape[0],)) # size = p
    delta_phases_offline_past1 = (delta_phases_offline_past1 + np.pi) % (2 * np.pi) - np.pi

    # MM offline LS on past 2 (size = p+k)
    phases_offline_past2 = MM_LS_IPL(Sigma_tilde_past2, iter_max_MM) 
    phases_offline_past2_0 = np.angle(phases_offline_past2[0])
    delta_phases_offline_past2 = np.angle(phases_offline_past2)- phases_offline_past2_0
    # delta_phases_offline_past2_p = np.angle(phases_offline_past2)
    w_past2_offline = phases_offline_past2 # size = p+k
    delta_phases_offline_past2 = delta_phases_offline_past2.reshape((delta_phases_offline_past2.shape[0],)) # size = p+k
    delta_phases_offline_past2 = (delta_phases_offline_past2 + np.pi) % (2 * np.pi) - np.pi

    # offline LS 
    phases_offline = MM_LS_IPL(Sigma_tilde, iter_max_MM) 
    phases_offline_0 = np.angle(phases_offline[0])
    delta_phases_offline = np.angle(phases_offline)- phases_offline_0
    delta_phases_offline = delta_phases_offline.reshape((delta_phases_offline.shape[0],)) # size = p+k+m
    delta_phases_offline = (delta_phases_offline + np.pi) % (2 * np.pi) - np.pi

    # sequential LS on bloc 1  (offline past1) # size = k
    phases_seq_MM_new_past1 = MM_LS_new_w(w_new_ones1, w_past1, new_past_Psi_tilde1, new_past_Sigma_tilde1, new_Psi_tilde1, new_Sigma_tilde1, iter_max_MM)
    delta_phases_seq_MM_new_past1 = np.angle(phases_seq_MM_new_past1) - phases_offline_past1_0
    delta_phases_seq_MM_new_past1 = delta_phases_seq_MM_new_past1.reshape((delta_phases_seq_MM_new_past1.shape[0],))
    delta_phases_seq_MM_new_past1 = (delta_phases_seq_MM_new_past1 + np.pi) % (2 * np.pi) - np.pi

    w_past2 = np.concatenate((w_past1, phases_seq_MM_new_past1)) # size = p+k

    # sequential LS on bloc 2 (concatenation past2)
    phases_seq_MM_new_past2 = MM_LS_new_w(w_new_ones2, w_past2, new_past_Psi_tilde2, new_past_Sigma_tilde2, new_Psi_tilde2, new_Sigma_tilde2, iter_max_MM)
    delta_phases_seq_MM_new_past2 = np.angle(phases_seq_MM_new_past2) - phases_offline_past1_0
    delta_phases_seq_MM_new_past2 = delta_phases_seq_MM_new_past2.reshape((delta_phases_seq_MM_new_past2.shape[0],))
    delta_phases_seq_MM_new_past2 = (delta_phases_seq_MM_new_past2 + np.pi) % (2 * np.pi) - np.pi

    # sequential LS on bloc 2 (offline past2)
    phases_seq_MM_new_past2_offline = MM_LS_new_w(w_new_ones2, w_past2_offline, new_past_Psi_tilde2, new_past_Sigma_tilde2, new_Psi_tilde2, new_Sigma_tilde2, iter_max_MM)
    delta_phases_seq_MM_new_past2_offline = np.angle(phases_seq_MM_new_past2_offline) - phases_offline_past2_0
    delta_phases_seq_MM_new_past2_offline = delta_phases_seq_MM_new_past2_offline.reshape((delta_phases_seq_MM_new_past2_offline.shape[0],))
    delta_phases_seq_MM_new_past2_offline = (delta_phases_seq_MM_new_past2_offline + np.pi) % (2 * np.pi) - np.pi

    return delta_phases_seq_MM_new_past1, delta_phases_seq_MM_new_past2, delta_phases_seq_MM_new_past2_offline, delta_phases_offline

def parallel_Monte_Carlo(m, k, p, n, trueCov, sampledist, number_of_trials, number_of_threads, iter_max_MM, Multi=True):
    if Multi:
        delta_phases_MM_seq_past1 = []
        delta_phases_MM_seq_past2 = []
        delta_phases_MM_seq_past2_offline = []
        delta_phases_MM_offline = []
        parallel_results = Parallel(n_jobs=number_of_threads)(delayed(oneMonteCarlo)(m, k, p, n, trueCov, sampledist, iter_max_MM, iMC) for iMC in tqdm(range(number_of_trials)))
        # paralle_results : tuple
        for i in range(number_of_trials):
            delta_phases_MM_seq_past1.append(parallel_results[i][0])
            delta_phases_MM_seq_past2.append(parallel_results[i][1])
            delta_phases_MM_seq_past2_offline.append(parallel_results[i][2])
            delta_phases_MM_offline.append(parallel_results[i][3])
        return delta_phases_MM_seq_past1, delta_phases_MM_seq_past2, delta_phases_MM_seq_past2_offline, delta_phases_MM_offline
    else:
        results = [] # results container
        delta_phases_MM_seq_past1 = []
        delta_phases_MM_seq_past2 = []
        delta_phases_MM_seq_past2_offline = []
        delta_phases_MM_offline = []
        for iMC in tqdm(range(number_of_trials)):
            results.append(oneMonteCarlo(m, k, p, n, trueCov, sampledist, iter_max_MM, iMC))
            delta_phases_MM_seq_past1.append( np.array(results[i][0]))
            delta_phases_MM_seq_past2.append( np.array(results[i][1]))
            delta_phases_MM_seq_past2_offline.append( np.array(results[i][2]))
            delta_phases_MM_offline.append( np.array(results[i][3]))
        return delta_phases_MM_seq_past1, delta_phases_MM_seq_past2, delta_phases_MM_seq_past2_offline, delta_phases_MM_offline
