# -----------------------------------------------------------------
# Librairies
# -----------------------------------------------------------------
import numpy as np
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64",True)

# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------

## Frobenius norm
#################

def gradient_new_w_LS(w_new, w_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde):
    """ A function that computes the gradient for the least squares (LS) optimization problem.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
        
        Output :
            * gradient : gradient vector
    """
    gradient = -4 * (w_past.conj().T@np.multiply(new_past_Psi_tilde.T, new_past_Sigma_tilde.conj().T)).conj().T \
                - 4 * (w_new.conj().T@np.multiply(new_Psi_tilde, new_Sigma_tilde)).conj().T
    return gradient

def gradient_new_w_LS_jax(w_new, w_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde):
    """ A function that computes the gradient for the least squares (LS) optimization problem.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
        
        Output :
            * gradient : gradient vector
    """
    gradient = -4 * (w_past.conj().T@jnp.multiply(new_past_Psi_tilde.T, new_past_Sigma_tilde.conj().T)).conj().T \
                - 4 * ( w_new.conj().T@jnp.multiply(new_Psi_tilde, new_Sigma_tilde)).conj().T
    return gradient


def gradient_new_w_LS_regularized(w_new, w_past, Sigma_tilde, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta):
    """ A function that computes the gradient for the least squares (LS) optimization problem with a shrinkage to identity covariance matrix.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Sigma_tilde_past : Covariance matrix of the past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * beta : regularization parameter
        
        Output :
            * gradient : gradient vector
    """
    k = w_new.shape[0]
    p = w_past.shape[0]
    I_k = np.eye(k)
    element2 = (1 - beta) * np.trace(Sigma_tilde) / (p+k)
    new_Sigma_tilde_bloc = beta * new_Sigma_tilde + element2 * I_k
    gradient = -4 * beta * (w_past.conj().T@np.multiply(new_past_Psi_tilde.T, new_past_Sigma_tilde.conj().T)).conj().T \
                - 4 * (w_new.conj().T@np.multiply(new_Psi_tilde, new_Sigma_tilde_bloc)).conj().T
    return gradient

def gradient_new_w_LS_regularized_jax(w_new, w_past, Sigma_tilde, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta):
    """ A function that computes the gradient for the least squares (LS) optimization problem with a shrinkage to identity covariance matrix.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Sigma_tilde_past : Covariance matrix of the past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * beta : regularization parameter
        
        Output :
            * gradient : gradient vector
    """
    k = w_new.shape[0]
    p = w_past.shape[0]
    I_k = jnp.eye(k)
    element2 = (1 - beta) * jnp.trace(Sigma_tilde) / (p+k)
    new_Sigma_tilde_bloc = beta * new_Sigma_tilde + element2 * I_k
    gradient = -4 * beta * (w_past.conj().T@jnp.multiply(new_past_Psi_tilde.T, new_past_Sigma_tilde.conj().T)).conj().T \
                - 4 * (w_new.conj().T@jnp.multiply(new_Psi_tilde, new_Sigma_tilde_bloc)).conj().T
    return gradient

def grad_costfunction_LS(Sigma_tilde, w_theta):
    M = -np.abs(Sigma_tilde)*Sigma_tilde
    return(2*M@w_theta)

# def gradient_new_w_LS_ShrinkageToTapering(w_new, w_past, tapering_matrix, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta):
#     """ A function that computes the gradient for the least squares (LS) optimization problem with a shrinkage to tapering covariance matrix.
#          Inputs :
#             * w_new : phases vector of new data
#             * w_past : phases vector of past data
#             * Sigma_tilde_past : Covariance matrix of the past data
#             * new_past_Psi_tilde : Coherence vector between past and new data
#             * new_past_Sigma_tilde : Covariance vector between past and new data
#             * new_Psi_tilde : Coherence matrix of the new data
#             * new_Sigma_tilde : Covariance matrix of the new data
#             * beta : regularization parameter
        
#         Output :
#             * gradient : gradient vector
#     """
#     k = w_new.shape[0]
#     p = w_past.shape[0]
#     new_past_tapering_upper = tapering_matrix[0:p, p:p+k]
#     new_tapering = tapering_matrix[p:p+k, p:p+k]
#     new_past_Sigma_tilde_bloc_upper = beta * new_past_Sigma_tilde.conj().T + (1 - beta) * np.multiply(new_past_tapering_upper, new_past_Sigma_tilde.conj().T)
#     new_Sigma_tilde_bloc = beta * new_Sigma_tilde + (1 - beta) * np.multiply(new_tapering, new_Sigma_tilde)
#     gradient = - 4 * (w_past.conj().T@np.multiply(new_past_Psi_tilde.T, new_past_Sigma_tilde_bloc_upper)).conj().T \
#                 - 4 * (w_new.conj().T@np.multiply(new_Psi_tilde, new_Sigma_tilde_bloc)).conj().T
#     return gradient

# def gradient_new_w_LS_ShrinkageToTapering_jax(w_new, w_past, tapering_matrix, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta):
#     """ A function that computes the gradient for the least squares (LS) optimization problem with a shrinkage to tapering covariance matrix.
#          Inputs :
#             * w_new : phases vector of new data
#             * w_past : phases vector of past data
#             * Sigma_tilde_past : Covariance matrix of the past data
#             * new_past_Psi_tilde : Coherence vector between past and new data
#             * new_past_Sigma_tilde : Covariance vector between past and new data
#             * new_Psi_tilde : Coherence matrix of the new data
#             * new_Sigma_tilde : Covariance matrix of the new data
#             * beta : regularization parameter
        
#         Output :
#             * gradient : gradient vector
#     """
#     k = w_new.shape[0]
#     p = w_past.shape[0]
#     new_past_tapering_upper = tapering_matrix[0:p, p:p+k]
#     new_tapering = tapering_matrix[p:p+k, p:p+k]
#     new_past_Sigma_tilde_bloc_upper = beta * new_past_Sigma_tilde.conj().T + (1 - beta) * jnp.multiply(new_past_tapering_upper, new_past_Sigma_tilde.conj().T)
#     new_Sigma_tilde_bloc = beta * new_Sigma_tilde + (1 - beta) * jnp.multiply(new_tapering, new_Sigma_tilde)
#     gradient = - 4 * (w_past.conj().T@jnp.multiply(new_past_Psi_tilde.T, new_past_Sigma_tilde_bloc_upper)).conj().T \
#                 - 4 * (w_new.conj().T@jnp.multiply(new_Psi_tilde, new_Sigma_tilde_bloc)).conj().T
#     return gradient

## Kullback Leibler divergence
##############################

def gradient_new_w_KL(w_new, w_past, Psi_tilde_past_inv, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde):
    """ A function that computes the gradient for the KullBack-Leibler (KL) optimization problem.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Psi_tilde_past : Coherence matrix of the past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
        
        Output :
            * gradient : gradient vector
    """

    # Psi_tilde_past_inv = np.linalg.inv(Psi_tilde_past)
    C2 = np.linalg.inv(new_Psi_tilde - new_past_Psi_tilde@Psi_tilde_past_inv@new_past_Psi_tilde.T)
    bloc12 = np.multiply(- Psi_tilde_past_inv@new_past_Psi_tilde.T@C2, new_past_Sigma_tilde.conj().T)
    bloc22 = np.multiply(C2, new_Sigma_tilde)
    gradient = 2 * (w_past.conj().T@bloc12 + w_new.conj().T@bloc22).conj().T
    return gradient

def gradient_new_w_KL_jax(w_new, w_past, Psi_tilde_past_inv, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde):
    """ A function that computes the gradient for the KullBack-Leibler (KL) optimization problem.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Psi_tilde_past : Coherence matrix of the past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
        
        Output :
            * gradient : gradient vector
    """
    # Psi_tilde_past_inv = jnp.linalg.inv(Psi_tilde_past)
    C2 = jnp.linalg.inv(new_Psi_tilde - new_past_Psi_tilde@Psi_tilde_past_inv@new_past_Psi_tilde.T)
    bloc12 = jnp.multiply(- Psi_tilde_past_inv@new_past_Psi_tilde.T@C2, new_past_Sigma_tilde.conj().T)
    bloc22 = jnp.multiply(C2, new_Sigma_tilde)
    gradient = 2 * (w_past.conj().T@bloc12 + w_new.conj().T@bloc22).conj().T
    return gradient

def gradient_new_w_KL_regularized(w_new, w_past, Sigma_tilde, Psi_tilde_past_inv, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta):
    """ A function that computes the gradient for the KullBack-Leibler (KL) optimization problem with a shrinkage to identity covariance matrix.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Sigma_tilde : Covariance matrix of all data
            * Psi_tilde_past : Coherence matrix of the past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * beta : regularization parameter
        
        Output :
            * gradient : gradient vector
    """
    k = w_new.shape[0]
    p = w_past.shape[0]
    I_k = np.eye(k)
    # Psi_tilde_past_inv = np.linalg.inv(Psi_tilde_past)
    element2 = (1 - beta) * np.trace(Sigma_tilde) / (p+k)
    new_Sigma_tilde_bloc = beta * new_Sigma_tilde + element2 * I_k
    C2 = np.linalg.inv(new_Psi_tilde - new_past_Psi_tilde@Psi_tilde_past_inv@new_past_Psi_tilde.T)
    bloc12 = np.multiply(- Psi_tilde_past_inv@new_past_Psi_tilde.T@C2, new_past_Sigma_tilde.conj().T)
    bloc22 = np.multiply(C2, new_Sigma_tilde_bloc)
    gradient = 2 * ( beta * w_past.conj().T@bloc12 + w_new.conj().T@bloc22).conj().T
    return gradient

def gradient_new_w_KL_regularized_jax(w_new, w_past, Sigma_tilde, Psi_tilde_past_inv, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta):
    """ A function that computes the gradient for the KullBack-Leibler (KL) optimization problem with a shrinkage to identity covariance matrix.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Sigma_tilde : Covariance matrix of all data
            * Psi_tilde_past : Coherence matrix of the past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * beta : regularization parameter
        
        Output :
            * gradient : gradient vector
    """
    k = w_new.shape[0]
    p = w_past.shape[0]
    I_k = jnp.eye(k)
    # Psi_tilde_past_inv = jnp.linalg.inv(Psi_tilde_past)
    element2 = (1 - beta) * jnp.trace(Sigma_tilde) / (p+k)
    new_Sigma_tilde_bloc = beta * new_Sigma_tilde + element2 * I_k
    C2 = jnp.linalg.inv(new_Psi_tilde - new_past_Psi_tilde@Psi_tilde_past_inv@new_past_Psi_tilde.T)
    bloc12 = jnp.multiply(- Psi_tilde_past_inv@new_past_Psi_tilde.T@C2, new_past_Sigma_tilde.conj().T)
    bloc22 = jnp.multiply(C2, new_Sigma_tilde_bloc)
    gradient = 2 * ( beta * w_past.conj().T@bloc12 + w_new.conj().T@bloc22).conj().T
    return gradient

def grad_costfunction_KL(w_theta, Sigma_tilde):
    M = np.linalg.inv(np.abs(Sigma_tilde))*Sigma_tilde
    return(2*M@w_theta)
