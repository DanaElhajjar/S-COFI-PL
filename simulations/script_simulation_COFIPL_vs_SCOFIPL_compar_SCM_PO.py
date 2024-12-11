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

def oneMonteCarlo(n, plugin, distance, trueCov, parameters, number_of_trials):
    """
    A function that performs a single Monte Carlo trial for phase recovery using different plug-in methods
    and optimization distances (KL or LS).

    Inputs:
        * n (int): Number of samples for covariance matrix estimation.
        * plugin (str): Method for covariance matrix estimation. 
            Options include: 'SCM', 'PO', 'SKSCM', 'SKPO', 'BWSCM', 'BWPO'.
        * distance (str): Distance type for optimization 
            Options include: 'KL', 'LS'
        * trueCov (numpy.ndarray): True covariance matrix for data generation.
        * parameters (tuple): Additional parameters:
            - k (int): Number of new dimensions.
            - p (int): Number of past dimensions.
            - b (float): Bandwidth parameter for tapered matrices.
            - beta (float): Regularization parameter.
            - iter_max_MM (int): Max iterations for MM algorithm.
            - sampledist (str): Distribution type for sample generation.
        * number_of_trials (int): Number of Monte Carlo trials.

    Outputs:
        * tuple: (delta_phase_MM_seq_ones, delta_theta_MM):
           - delta_phase_MM_seq_ones (numpy.ndarray): Sequential MM phase differences.
           - delta_theta_MM (numpy.ndarray): Offline MM phase differences.
    """

    k, p, b, beta, iter_max_MM, sampledist = parameters

    # initialization
    w_new_ones = np.ones((k, 1) ,dtype=complex) 
    
    X = sampledistributionchoice(trueCov, p+k, n, sampledist)
    X = np.asarray(X, dtype = np.complex128)

    w_new_ones = np.ones((k, 1) ,dtype=complex) 
    w_new_ones_PO = np.ones((k, 1) ,dtype=complex) 

    # PO
    S_all_PO = corr_phase(X)
    Sigma_tilde_PO = S_all_PO
    Sigma_tilde_past_PO = Sigma_tilde_PO[0:p, 0:p]
    new_past_Sigma_tilde_PO = Sigma_tilde_PO[p:p+k, 0:p]
    new_Sigma_tilde_PO = Sigma_tilde_PO[p:p+k, p:p+k]
    Psi_tilde_PO = np.abs(Sigma_tilde_PO)
    Psi_tilde_past_PO = Psi_tilde_PO[0:p, 0:p]
    new_past_Psi_tilde_PO = Psi_tilde_PO[p:p+k, 0:p]
    new_Psi_tilde_PO = Psi_tilde_PO[p:p+k, p:p+k]

    # SCM
    S_all = SCM(X)
    Sigma_tilde = S_all
    Sigma_tilde_past = Sigma_tilde[0:p, 0:p]
    new_past_Sigma_tilde = Sigma_tilde[p:p+k, 0:p]
    new_Sigma_tilde = Sigma_tilde[p:p+k, p:p+k]
    Psi_tilde = np.abs(Sigma_tilde)
    Psi_tilde_past = Psi_tilde[0:p, 0:p]
    new_past_Psi_tilde = Psi_tilde[p:p+k, 0:p]
    new_Psi_tilde = Psi_tilde[p:p+k, p:p+k]

    if distance == 'KL':
        Psi_tilde_past_inv = np.linalg.inv(Psi_tilde_past)
        Psi_tilde_past_inv_PO = np.linalg.inv(Psi_tilde_past_PO)
        # MM SCM en offline sur p+k
        theta_temp = MM_KL_IPL(Sigma_tilde, iter_max_MM)
        theta_temp_0 = np.angle(theta_temp[0])
        delta_theta_MM = np.angle(theta_temp) - theta_temp_0
        w_past_MM = theta_temp[0:p]
        delta_theta_MM = delta_theta_MM.reshape((delta_theta_MM.shape[0],))

        # sequential sur k new data
        # offline sur past p
        Sigma_tilde_past_off = Sigma_tilde[0:p, 0:p]
        theta_temp_past = MM_KL_IPL(Sigma_tilde_past_off, iter_max_MM)
        theta_temp_0_past = np.angle(theta_temp_past[0])
        w_past_MM = theta_temp_past 
        # sequential MM                             
        phase_MM_seq_ones = MM_KL_new_w(w_new_ones, w_past_MM, Psi_tilde_past_inv, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_MM)
        delta_phase_MM_seq_ones = np.angle(phase_MM_seq_ones) - theta_temp_0_past
        delta_phase_MM_seq_ones = delta_phase_MM_seq_ones.reshape((delta_phase_MM_seq_ones.shape[0],))

        # MM en offline PO en offline sur p+k
        theta_temp_PO = MM_KL_IPL(Sigma_tilde_PO, iter_max_MM)
        theta_temp_0_PO = np.angle(theta_temp_PO[0])
        delta_theta_MM_PO= np.angle(theta_temp_PO) - theta_temp_0_PO
        delta_theta_MM_PO = delta_theta_MM_PO.reshape((delta_theta_MM_PO.shape[0],))

        # sequential sur k new data
        # offline sur past p
        theta_temp_PO_past = MM_KL_IPL(Sigma_tilde_past_PO, iter_max_MM)
        theta_temp_0_PO_past = np.angle(theta_temp_PO_past[0])
        w_past_MM_PO = theta_temp_PO_past[0:p]

        # sequential PO
        phase_MM_seq_ones_PO = MM_KL_new_w(w_new_ones_PO, w_past_MM_PO, Psi_tilde_past_inv_PO, new_past_Psi_tilde_PO, new_past_Sigma_tilde_PO, new_Psi_tilde_PO, new_Sigma_tilde_PO, iter_max_MM)
        delta_phase_MM_seq_ones_PO = np.angle(phase_MM_seq_ones_PO) - theta_temp_0_PO_past
        delta_phase_MM_seq_ones_PO = delta_phase_MM_seq_ones_PO.reshape((delta_phase_MM_seq_ones_PO.shape[0],))

    if distance == 'LS':
        # MM SCM en offline sur p+k
        theta_temp = MM_LS_IPL(Sigma_tilde, iter_max_MM)
        theta_temp_0 = np.angle(theta_temp[0])
        delta_theta_MM = np.angle(theta_temp) - theta_temp_0
        w_past_MM = theta_temp[0:p]
        delta_theta_MM = delta_theta_MM.reshape((delta_theta_MM.shape[0],))

        # sequential sur k new data
        # offline sur past p
        Sigma_tilde_past_off = Sigma_tilde[0:p, 0:p]
        theta_temp_past = MM_LS_IPL(Sigma_tilde_past_off, iter_max_MM)
        theta_temp_0_past = np.angle(theta_temp_past[0])
        w_past_MM = theta_temp_past 
        # sequential MM                             
        phase_MM_seq_ones = MM_LS_new_w(w_new_ones, w_past_MM, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_MM)
        # MM_LS_new_w(w_new_ones, w_past_MM, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_MM)
        # MM_LS_new_w_regularized(w_new_ones, w_past_MM, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, Sigma_tilde, iter_max_MM, beta)
        delta_phase_MM_seq_ones = np.angle(phase_MM_seq_ones) - theta_temp_0_past
        delta_phase_MM_seq_ones = delta_phase_MM_seq_ones.reshape((delta_phase_MM_seq_ones.shape[0],))

        # MM en offline PO en offline sur p+k
        theta_temp_PO = MM_LS_IPL(Sigma_tilde_PO, iter_max_MM)
        theta_temp_0_PO = np.angle(theta_temp_PO[0])
        delta_theta_MM_PO= np.angle(theta_temp_PO) - theta_temp_0_PO
        delta_theta_MM_PO = delta_theta_MM_PO.reshape((delta_theta_MM_PO.shape[0],))

        # sequential sur k new data
        # offline sur past p
        theta_temp_PO_past = MM_LS_IPL(Sigma_tilde_past_PO, iter_max_MM)
        theta_temp_0_PO_past = np.angle(theta_temp_PO_past[0])
        w_past_MM_PO = theta_temp_PO_past[0:p]

        # PO
        phase_MM_seq_ones_PO = MM_LS_new_w(w_new_ones_PO, w_past_MM_PO, new_past_Psi_tilde_PO, new_past_Sigma_tilde_PO, new_Psi_tilde_PO, new_Sigma_tilde_PO, iter_max_MM)
        delta_phase_MM_seq_ones_PO = np.angle(phase_MM_seq_ones_PO) - theta_temp_0_PO_past
        delta_phase_MM_seq_ones_PO = delta_phase_MM_seq_ones_PO.reshape((delta_phase_MM_seq_ones_PO.shape[0],))

    return (delta_phase_MM_seq_ones, 
            delta_phase_MM_seq_ones_PO,
            delta_theta_MM,
            delta_theta_MM_PO)

def parallel_Monte_Carlo(p, k, n, trueCov, parameters, number_of_trials, number_of_threads, Multi=True):
    if Multi:
        delta_phase_MM_seq_ones = []
        delta_theta_MM = []
        delta_phase_MM_seq_ones_PO = []
        delta_theta_MM_PO = []
        parallel_results = Parallel(n_jobs=number_of_threads)(delayed(oneMonteCarlo)(k, p, n, trueCov, parameters, iMC) for iMC in tqdm(range(number_of_trials)))
        # paralle_results : tuple
        for i in range(number_of_trials):
            delta_phase_MM_seq_ones.append(parallel_results[i][0])
            delta_phase_MM_seq_ones_PO.append(parallel_results[i][1])
            delta_theta_MM.append(np.array(parallel_results[i][2]))
            delta_theta_MM_PO.append(np.array(parallel_results[i][3]))
        return delta_phase_MM_seq_ones, delta_phase_MM_seq_ones_PO, delta_theta_MM, delta_theta_MM_PO
    else:
        results = [] # results container
        delta_phase_MM_seq_ones = []
        delta_theta_MM = []
        for iMC in tqdm(range(number_of_trials)):
            results.append(oneMonteCarlo(p, k, n, trueCov, parameters, iMC))
            delta_phase_MM_seq_ones.append(results[i][0])
            delta_theta_MM.append(np.array(results[i][1]))
        return (delta_phase_MM_seq_ones, 
                delta_theta_MM)