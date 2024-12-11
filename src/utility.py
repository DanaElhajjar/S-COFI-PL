# -----------------------------------------------------------------
# Librairies
# -----------------------------------------------------------------
import numpy as np
import scipy as sp
import autograd.numpy as np_a

# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------
def inv(M):
    """ A function that computes the inverse matrix.
            Inputs:
                * M = matrix
            Outputs:
                * Out = inverse matrix"""
    
    eigvals, eigvects = np_a.linalg.eigh(M)
    eigvals = np_a.diag(1/eigvals)
    Out = np_a.dot(
        np_a.dot(eigvects, eigvals), np_a.conjugate(eigvects.T))
    return Out

def ToeplitzMatrix(rho, p):
    """ A function that computes a Hermitian semi-positive matrix.
            Inputs:
                * rho = a scalar
                * p = size of matrix
            Outputs:
                * the matrix """

    return sp.linalg.toeplitz(np.power(rho, np.arange(0, p)))

def calculateMSE(phasedifference, deltathetasim,n_MC,vecL):
    """ A function to calculate the MSE of the phase difernces
    Inputs : 
        * phasedifference : vector of estimated difference
        * deltathetasim : vector of true values of phase differences
    Outputs : 
        * Vector of MSE (size = size of the vector n_samples) """
    return np.array([np.sum(abs(list(phasedifference)[L] - deltathetasim), axis = 0) /n_MC for L in range(len(vecL))])


