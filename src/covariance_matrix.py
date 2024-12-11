# -----------------------------------------------------------------
# Librairies
# -----------------------------------------------------------------
import numpy as np


# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------
def SCM(X):  
    """ A function that computes the ML Estimator for covariance matrix estimation for gaussian data
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
        Outputs:
            * sigma_mle = the ML estimate"""
    p = X.shape[0]
    n = X.shape[1] 

    # initialization
    sigma_mle = np.zeros((p, p)) 

    sigma_mle = (X@X.conj().T) / n 
    
    return sigma_mle


def SCM_LR(X,r):
    (p,n) = X.shape
    Sigma = SCM(X)
    u,s,vh = np.linalg.svd(Sigma)
    u_signal = u[:,:r]
    u_noise = u[:,r:]
    sigma = np.mean(s[r:])
    Sigma = u_signal @ np.diag(s[:r])@u_signal.conj().T + sigma * u_noise@u_noise.conj().T
    return Sigma
    
def corr_phase(X):
    """ A function that computes phase only Estimator for covariance matrix estimation
        Inputs:
            * X = a matrix of size p*N with each observation along column dimension

        Outputs:
            * PO estimator 
    """
    (p,n) = X.shape
    X = X/np.sqrt(abs(X)**2)
    return(np.dot(X,X.conj().T)/n)


def regul_linear(Sigma, beta):
    """
    A function that performs linear regularization of a covariance matrix by interpolating 
    between the input covariance matrix and a scaled identity matrix. 
    This method helps stabilize computations, particularly for ill-conditioned 
    or singular covariance matrices.

        Inputs:
            * Sigma : numpy.ndarray
                A square covariance matrix of shape (p, p) to be regularized.
            * beta : float
                Regularization coefficient in the range [0, 1].
                - beta = 1: The output matrix is the original `Sigma`.
                - beta = 0: The output matrix is a scaled identity matrix based on 
                the normalized trace of `Sigma`.
        Outputs:
            * numpy.ndarray
                The regularized covariance matrix of shape (p, p).
    
    """
    p = Sigma.shape[0]
    I_p = np.eye(p)
    Sigma = beta * Sigma + (1-beta) * I_p *(np.trace(Sigma)/p) 
    return Sigma

def bandw(Sigma,band):
    """ A function that computes covariance matrix tapering for covariance matrix estimation
        Inputs:
            * Sigma covariance estimator
            * band : bandwidth parameter

        Outputs:
            * covariance matrix tapering estimator
    """
    N = Sigma.shape[0]
    transform = np.eye(N,N)
    
    for i in range(N-1):
        transform[i,(i):(i+1+band)] = 1 
        transform[(i):(i+1+band),i] = 1 
    Sigma = np.multiply(Sigma,transform)
    return(Sigma)