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

    if plugin == 'SCM':
        Sigma_tilde = SCM(X)
    if plugin == 'PO':
        Sigma_tilde = corr_phase(X)
    if plugin == 'SKSCM':
        S_all = SCM(X)
        Sigma_tilde = regul_linear(S_all, beta)
        Sigma_tilde_past_off = regul_linear(S_all[0:p, 0:p], beta)
    if plugin == 'SKPO':
        S_all = corr_phase(X)
        Sigma_tilde = regul_linear(S_all, beta)
        Sigma_tilde_past_off = regul_linear(S_all[0:p, 0:p], beta)
    if plugin == 'BWSCM':
        Sigma_tilde = bandw(SCM(X), b)
    if plugin == 'BWPO':
        Sigma_tilde = bandw(corr_phase(X), b)


    Sigma_tilde_past = Sigma_tilde[0:p, 0:p]
    new_past_Sigma_tilde = Sigma_tilde[p:p+k, 0:p]
    new_Sigma_tilde = Sigma_tilde[p:p+k, p:p+k]
    Psi_tilde = np.abs(Sigma_tilde)
    Psi_tilde_past = Psi_tilde[0:p, 0:p]
    new_past_Psi_tilde = Psi_tilde[p:p+k, 0:p]
    new_Psi_tilde = Psi_tilde[p:p+k, p:p+k]

    if distance == 'KL':
        Psi_tilde_past_inv = np.linalg.inv(Psi_tilde_past)
        # MM en offline sur p+k
        theta_temp = MM_KL_IPL(Sigma_tilde, iter_max_MM)
        theta_temp_0 = np.angle(theta_temp[0])
        delta_theta_MM = np.angle(theta_temp) - theta_temp_0
        delta_theta_MM = delta_theta_MM.reshape((delta_theta_MM.shape[0],))
        delta_theta_MM = (delta_theta_MM + np.pi) % (2 * np.pi) - np.pi

        # offline sur past p
        if plugin == 'SKSCM':
            Sigma_tilde_past_off = regul_linear(SCM(X)[0:p, 0:p], beta)
        elif plugin == 'SKPO':
            Sigma_tilde_past_off = regul_linear(corr_phase(X)[0:p, 0:p], beta)
        else:
            Sigma_tilde_past_off = Sigma_tilde_past
        theta_temp_past = MM_KL_IPL(Sigma_tilde_past_off, iter_max_MM)
        theta_temp_0_past = np.angle(theta_temp_past[0])
        w_past_MM = theta_temp_past

        # sequential MM                             
        phase_MM_seq_ones = MM_KL_new_w(w_new_ones, w_past_MM, Psi_tilde_past_inv, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_MM)
        delta_phase_MM_seq_ones = np.angle(phase_MM_seq_ones) - theta_temp_0_past
        delta_phase_MM_seq_ones = delta_phase_MM_seq_ones.reshape((delta_phase_MM_seq_ones.shape[0],))
        delta_phase_MM_seq_ones = (delta_phase_MM_seq_ones + np.pi) % (2 * np.pi) - np.pi

    if distance == 'LS':
        # MM en offline sur p+k
        theta_temp = MM_LS_IPL(Sigma_tilde, iter_max_MM)
        theta_temp_0 = np.angle(theta_temp[0])
        delta_theta_MM = np.angle(theta_temp)- theta_temp_0
        delta_theta_MM = delta_theta_MM.reshape((delta_theta_MM.shape[0],))
        delta_theta_MM = (delta_theta_MM + np.pi) % (2 * np.pi) - np.pi

        # MM en offline sur p
        if plugin == 'SKSCM':
            Sigma_tilde_past_off = regul_linear(SCM(X)[0:p, 0:p], beta)
        elif plugin == 'SKPO':
            Sigma_tilde_past_off = regul_linear(corr_phase(X)[0:p, 0:p], beta)
        else:
            Sigma_tilde_past_off = Sigma_tilde_past
        theta_temp_past = MM_LS_IPL(Sigma_tilde_past, iter_max_MM)
        theta_temp_0_past = np.angle(theta_temp_past[0])
        w_past_MM = theta_temp_past

        # sequential MM 
        phase_MM_seq_ones = MM_LS_new_w(w_new_ones, w_past_MM, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_MM)
        delta_phase_MM_seq_ones = np.angle(phase_MM_seq_ones) - theta_temp_0_past
        delta_phase_MM_seq_ones = delta_phase_MM_seq_ones.reshape((delta_phase_MM_seq_ones.shape[0],))
        delta_phase_MM_seq_ones = (delta_phase_MM_seq_ones + np.pi) % (2 * np.pi) - np.pi

    return (delta_phase_MM_seq_ones, 
            delta_theta_MM)

def parallel_Monte_Carlo(p, k, n, trueCov, parameters, number_of_trials, number_of_threads, Multi=True):
    if Multi:
        delta_phase_MM_seq_ones = []
        delta_theta_MM = []
        parallel_results = Parallel(n_jobs=number_of_threads)(delayed(oneMonteCarlo)(k, p, n, trueCov, parameters, iMC) for iMC in tqdm(range(number_of_trials)))
        # paralle_results : tuple
        for i in range(number_of_trials):
            delta_phase_MM_seq_ones.append(parallel_results[i][0])
            delta_theta_MM.append(np.array(parallel_results[i][1]))
        return delta_phase_MM_seq_ones, delta_theta_MM
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
