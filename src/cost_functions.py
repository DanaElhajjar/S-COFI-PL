# -----------------------------------------------------------------
# Librairies
# -----------------------------------------------------------------
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
import time as t
from jax.config import config
config.update("jax_enable_x64",True)

# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------

## Frobenius norm
#################

def cost_fct_LS(w_new, w_past, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde):
    """
     A function computes the cost for the least squares (LS) optimization problem.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Psi_tilde_past : Coherence matrix of past data
            * Sigma_tilde_past : Covariance matrix of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
        
        Output :
            * cost : the cost value
    """
    cost = -2 * w_past.conj().T@np.multiply(Psi_tilde_past, Sigma_tilde_past)@w_past \
            - 2 * w_past.conj().T@np.multiply(new_past_Psi_tilde.T, new_past_Sigma_tilde.conj().T)@w_new \
            - 2 * w_new.conj().T@np.multiply(new_past_Psi_tilde, new_past_Sigma_tilde)@w_past \
            - 2 * w_new.conj().T@np.multiply(new_Psi_tilde, new_Sigma_tilde)@w_new
    return np.real(cost.squeeze())

def cost_fct_LS_jax(w_new, w_past, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde):
    """
     A function computes the cost for the least squares (LS) optimization problem.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Psi_tilde_past : Coherence matrix of past data
            * Sigma_tilde_past : Covariance matrix of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
        
        Output :
            * cost : the cost value
    """
    cost = -2 * w_past.conj().T@jnp.multiply(Psi_tilde_past, Sigma_tilde_past)@w_past \
            - 2 * w_past.conj().T@jnp.multiply(new_past_Psi_tilde.T, new_past_Sigma_tilde.conj().T)@w_new \
            - 2 * w_new.conj().T@jnp.multiply(new_past_Psi_tilde, new_past_Sigma_tilde)@w_past \
            - 2 * w_new.conj().T@jnp.multiply(new_Psi_tilde, new_Sigma_tilde)@w_new
    return jnp.real(cost.squeeze())

def cost_fct_LS_regularized(w_new, w_past, Sigma_tilde, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta):
    """
     A function computes the cost for the least squares (LS) optimization problem with a regularized covariance matrix.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Sigma_tilde : Covariance matrix of all data
            * Psi_tilde_past : Coherence matrix of past data
            * Sigma_tilde_past : Covariance matrix of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * beta : regularization parameter
        
        Output :
            * cost : the cost value
    """
    k = w_new.shape[0]
    p = w_past.shape[0]
    I_k = np.eye(k, dtype=np.complex128)
    I_p = np.eye(p, dtype=np.complex128)
    element2 = (1 - beta) * np.trace(Sigma_tilde) / (p+k)
    Sigma_tilde_past_bloc = beta * Sigma_tilde_past + element2 * I_p
    new_Sigma_tilde_bloc = beta * new_Sigma_tilde + element2 * I_k
    cost = -2 * w_past.conj().T@np.multiply(Psi_tilde_past, Sigma_tilde_past_bloc)@w_past \
            - 2 * w_past.conj().T@np.multiply(new_past_Psi_tilde.T, beta*new_past_Sigma_tilde.conj().T)@w_new \
            - 2 * w_new.conj().T@np.multiply(new_past_Psi_tilde, beta*new_past_Sigma_tilde)@w_past \
            - 2 * w_new.conj().T@np.multiply(new_Psi_tilde, new_Sigma_tilde_bloc)@w_new
    return np.real(cost.squeeze())

def cost_fct_LS_regularized_jax(w_new, w_past, Sigma_tilde, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta):
    """
     A function computes the cost for the least squares (LS) optimization problem with a regularized covariance matrix.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Sigma_tilde : Covariance matrix of all data
            * Psi_tilde_past : Coherence matrix of past data
            * Sigma_tilde_past : Covariance matrix of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * beta : regularization parameter
        
        Output :
            * cost : the cost value
    """
    k = w_new.shape[0]
    p = w_past.shape[0]
    I_k = jnp.eye(k, dtype=np.complex128)
    I_p = jnp.eye(p, dtype=np.complex128)
    element2 = (1 - beta) * jnp.trace(Sigma_tilde) / (p+k)
    Sigma_tilde_past_bloc = beta * Sigma_tilde_past + element2 * I_p
    new_Sigma_tilde_bloc = beta * new_Sigma_tilde + element2 * I_k
    cost = -2 * w_past.conj().T@jnp.multiply(Psi_tilde_past, Sigma_tilde_past_bloc)@w_past \
            - 2 * w_past.conj().T@jnp.multiply(new_past_Psi_tilde.T, beta*new_past_Sigma_tilde.conj().T)@w_new \
            - 2 * w_new.conj().T@jnp.multiply(new_past_Psi_tilde, beta*new_past_Sigma_tilde)@w_past \
            - 2 * w_new.conj().T@jnp.multiply(new_Psi_tilde, new_Sigma_tilde_bloc)@w_new
    return np.real(cost.squeeze())

def costfunction_LS(w_theta, Sigma_tilde, Psi_tilde):
    cost = -w_theta.conj().T@((Psi_tilde*Sigma_tilde)@w_theta)
    return np.squeeze(np.real(cost))

def costfunction_LS_jax(Sigma_tilde, w_theta, Psi_tilde):
    cost = -w_theta.conj().T@((Psi_tilde*Sigma_tilde)@w_theta)
    return jnp.squeeze(jnp.real(cost))

# def cost_fct_LS_ShrinkageToTapering(w_new, w_past, tapering_matrix, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta):
#     """
#      A function computes the cost for the least squares (LS) optimization problem with a shrinkage to tapering covariance matrix.
#          Inputs :
#             * w_new : phases vector of new data
#             * w_past : phases vector of past data
#             * Sigma_tilde : Covariance matrix of all data
#             * Psi_tilde_past : Coherence matrix of past data
#             * Sigma_tilde_past : Covariance matrix of past data
#             * new_past_Psi_tilde : Coherence vector between past and new data
#             * new_past_Sigma_tilde : Covariance vector between past and new data
#             * new_Psi_tilde : Coherence matrix of the new data
#             * new_Sigma_tilde : Covariance matrix of the new data
#             * beta : regularization parameter
        
#         Output :
#             * cost : the cost value
#     """
#     k = w_new.shape[0]
#     p = w_past.shape[0]
#     past_tapering = tapering_matrix[0:p, 0:p]
#     new_past_tapering_lower= tapering_matrix[p:p+k, 0:p]
#     new_past_tapering_upper = tapering_matrix[0:p, p:p+k]
#     new_tapering = tapering_matrix[p:p+k, p:p+k]
#     Sigma_tilde_past_bloc = beta * Sigma_tilde_past + (1 - beta) * np.multiply(past_tapering, Sigma_tilde_past)
#     new_past_Sigma_tilde_bloc_upper = beta * new_past_Sigma_tilde.conj().T + (1 - beta) * np.multiply(new_past_tapering_upper, new_past_Sigma_tilde.conj().T)
#     new_past_Sigma_tilde_bloc_lower = beta * new_past_Sigma_tilde + (1 - beta) * np.multiply(new_past_tapering_lower, new_past_Sigma_tilde)
#     new_Sigma_tilde_bloc = beta * new_Sigma_tilde + (1 - beta) * np.multiply(new_tapering, new_Sigma_tilde)
#     cost = -2 * w_past.conj().T@np.multiply(Psi_tilde_past, Sigma_tilde_past_bloc)@w_past \
#             - 2 * w_past.conj().T@np.multiply(new_past_Psi_tilde.T, new_past_Sigma_tilde_bloc_upper)@w_new \
#             - 2 * w_new.conj().T@np.multiply(new_past_Psi_tilde, new_past_Sigma_tilde_bloc_lower)@w_past \
#             - 2 * w_new.conj().T@np.multiply(new_Psi_tilde, new_Sigma_tilde_bloc)@w_new
#     return np.real(cost.squeeze())

# def cost_fct_LS_ShrinkageToTapering_jax(w_new, w_past, tapering_matrix, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta):
#     """
#      A function computes the cost for the least squares (LS) optimization problem with a shrinkage to tapering covariance matrix.
#          Inputs :
#             * w_new : phases vector of new data
#             * w_past : phases vector of past data
#             * Sigma_tilde : Covariance matrix of all data
#             * Psi_tilde_past : Coherence matrix of past data
#             * Sigma_tilde_past : Covariance matrix of past data
#             * new_past_Psi_tilde : Coherence vector between past and new data
#             * new_past_Sigma_tilde : Covariance vector between past and new data
#             * new_Psi_tilde : Coherence matrix of the new data
#             * new_Sigma_tilde : Covariance matrix of the new data
#             * beta : regularization parameter
        
#         Output :
#             * cost : the cost value
#     """
#     k = w_new.shape[0]
#     p = w_past.shape[0]
#     past_tapering = tapering_matrix[0:p, 0:p]
#     new_past_tapering_lower = tapering_matrix[p:p+k, 0:p]
#     new_past_tapering_upper = tapering_matrix[0:p, p:p+k]
#     new_tapering = tapering_matrix[p:p+k, p:p+k]
#     Sigma_tilde_past_bloc = beta * Sigma_tilde_past + (1 - beta) * jnp.multiply(past_tapering, Sigma_tilde_past)
#     new_past_Sigma_tilde_bloc_upper = beta * new_past_Sigma_tilde.conj().T + (1 - beta) * jnp.multiply(new_past_tapering_upper, new_past_Sigma_tilde.conj().T)
#     new_past_Sigma_tilde_bloc_lower = beta * new_past_Sigma_tilde + (1 - beta) * jnp.multiply(new_past_tapering_lower, new_past_Sigma_tilde)
#     new_Sigma_tilde_bloc = beta * new_Sigma_tilde + (1 - beta) * jnp.multiply(new_tapering, new_Sigma_tilde)
#     cost = -2 * w_past.conj().T@jnp.multiply(Psi_tilde_past, Sigma_tilde_past_bloc)@w_past \
#             - 2 * w_past.conj().T@jnp.multiply(new_past_Psi_tilde.T, new_past_Sigma_tilde_bloc_upper)@w_new \
#             - 2 * w_new.conj().T@jnp.multiply(new_past_Psi_tilde, new_past_Sigma_tilde_bloc_lower)@w_past \
#             - 2 * w_new.conj().T@jnp.multiply(new_Psi_tilde, new_Sigma_tilde_bloc)@w_new
#     return jnp.real(cost.squeeze())


## Kullback Leibler divergence
##############################

def cost_fct_KL(w_new, w_past, Psi_tilde_past, Psi_tilde_past_inv, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde):
    """
     A function computes the cost for the KullBack-Leibler (KL) optimization problem.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Psi_tilde_past : Coherence matrix of past data
            * Sigma_tilde_past : Covariance matrix of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
        
        Output :
            * cost : the cost value
    """
    # Psi_tilde_past_inv = np.linalg.inv(Psi_tilde_past)
    new_Psi_tilde_inv = np.linalg.inv(new_Psi_tilde)
    C1 = np.linalg.inv(Psi_tilde_past - new_past_Psi_tilde.T@new_Psi_tilde_inv@new_past_Psi_tilde)
    C2 = np.linalg.inv(new_Psi_tilde - new_past_Psi_tilde@Psi_tilde_past_inv@new_past_Psi_tilde.T)
    bloc11 = np.multiply(C1, Sigma_tilde_past)
    bloc12 = np.multiply(- Psi_tilde_past_inv@new_past_Psi_tilde.T@C2, new_past_Sigma_tilde.conj().T)
    bloc21 = np.multiply(- C2@new_past_Psi_tilde@Psi_tilde_past_inv, new_past_Sigma_tilde)
    bloc22 = np.multiply(C2, new_Sigma_tilde)
    cost = w_past.conj().T@bloc11@w_past + w_past.conj().T@bloc12@w_new \
        + w_new.conj().T@bloc21@w_past + w_new.conj().T@bloc22@w_new 
    return np.real(cost.squeeze())

def cost_fct_KL_jax(w_new, w_past, Psi_tilde_past, Psi_tilde_past_inv, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde):
    """
     A function computes the cost for the KullBack-Leibler (KL) optimization problem.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Psi_tilde_past : Coherence matrix of past data
            * Sigma_tilde_past : Covariance matrix of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
        
        Output :
            * cost : the cost value
    """
    # Psi_tilde_past_inv = jnp.linalg.inv(Psi_tilde_past)
    new_Psi_tilde_inv = jnp.linalg.inv(new_Psi_tilde)
    C1 = jnp.linalg.inv(Psi_tilde_past - new_past_Psi_tilde.T@new_Psi_tilde_inv@new_past_Psi_tilde)
    C2 = jnp.linalg.inv(new_Psi_tilde - new_past_Psi_tilde@Psi_tilde_past_inv@new_past_Psi_tilde.T)
    bloc11 = jnp.multiply(C1, Sigma_tilde_past)
    bloc12 = jnp.multiply(- Psi_tilde_past_inv@new_past_Psi_tilde.T@C2, new_past_Sigma_tilde.conj().T)
    bloc21 = jnp.multiply(- C2@new_past_Psi_tilde@Psi_tilde_past_inv, new_past_Sigma_tilde)
    bloc22 = jnp.multiply(C2, new_Sigma_tilde)
    cost = w_past.conj().T@bloc11@w_past + w_past.conj().T@bloc12@w_new \
        + w_new.conj().T@bloc21@w_past + w_new.conj().T@bloc22@w_new 
    return jnp.real(cost.squeeze())

def cost_fct_KL_regularized(w_new, w_past, Sigma_tilde, Psi_tilde_past, Psi_tilde_past_inv, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta):
    """
     A function computes the cost for the KullBack-Leibler (KL) optimization problem with a shrinkage to identity covariance matrix.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Sigma_tilde : covariance matrix of all data
            * Psi_tilde_past : Coherence matrix of past data
            * Sigma_tilde_past : Covariance matrix of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * beta : regularization parameter
        
        Output :
            * cost : the cost value
    """
    k = w_new.shape[0]
    p = w_past.shape[0]
    I_k = np.eye(k, dtype=np.complex128)
    I_p = np.eye(p, dtype=np.complex128)
    # Psi_tilde_past_inv = np.linalg.inv(Psi_tilde_past)
    new_Psi_tilde_inv = np.linalg.inv(new_Psi_tilde)
    C1 = np.linalg.inv(Psi_tilde_past - new_past_Psi_tilde.T@new_Psi_tilde_inv@new_past_Psi_tilde)
    C2 = np.linalg.inv(new_Psi_tilde - new_past_Psi_tilde@Psi_tilde_past_inv@new_past_Psi_tilde.T)
    element2 = (1 - beta) * np.trace(Sigma_tilde) / (p+k)
    Sigma_tilde_past_bloc = beta * Sigma_tilde_past + element2 * I_p
    new_Sigma_tilde_bloc = beta * new_Sigma_tilde + element2 * I_k
    bloc11 = np.multiply(C1, Sigma_tilde_past_bloc)
    bloc12 = np.multiply(- Psi_tilde_past_inv@new_past_Psi_tilde.T@C2, new_past_Sigma_tilde.conj().T)
    bloc21 = np.multiply(- C2@new_past_Psi_tilde@Psi_tilde_past_inv, new_past_Sigma_tilde)
    bloc22 = np.multiply(C2, new_Sigma_tilde_bloc)
    cost = w_past.conj().T@bloc11@w_past + beta * w_past.conj().T@bloc12@w_new \
        + beta *  w_new.conj().T@bloc21@w_past + w_new.conj().T@bloc22@w_new 
    return np.real(cost.squeeze())

def cost_fct_KL_regularized_jax(w_new, w_past, Sigma_tilde, Psi_tilde_past, Psi_tilde_past_inv, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta):
    """
     A function computes the cost for the KullBack-Leibler (KL) optimization problem with a shrinkage to identity covariance matrix.
         Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Sigma_tilde : covariance matrix of all data
            * Psi_tilde_past : Coherence matrix of past data
            * Sigma_tilde_past : Covariance matrix of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * beta : regularization parameter
        
        Output :
            * cost : the cost value
    """
    k = w_new.shape[0]
    p = w_past.shape[0]
    I_k = jnp.eye(k, dtype=jnp.complex128)
    I_p = jnp.eye(p, dtype=jnp.complex128)
    # Psi_tilde_past_inv = jnp.linalg.inv(Psi_tilde_past)
    new_Psi_tilde_inv = jnp.linalg.inv(new_Psi_tilde)
    C1 = jnp.linalg.inv(Psi_tilde_past - new_past_Psi_tilde.T@new_Psi_tilde_inv@new_past_Psi_tilde)
    C2 = jnp.linalg.inv(new_Psi_tilde - new_past_Psi_tilde@Psi_tilde_past_inv@new_past_Psi_tilde.T)
    element2 = (1 - beta) * jnp.trace(Sigma_tilde) / (p+k)
    Sigma_tilde_past_bloc = beta * Sigma_tilde_past + element2 * I_p
    new_Sigma_tilde_bloc = beta * new_Sigma_tilde + element2 * I_k
    bloc11 = jnp.multiply(C1, Sigma_tilde_past_bloc)
    bloc12 = jnp.multiply(- Psi_tilde_past_inv@new_past_Psi_tilde.T@C2, new_past_Sigma_tilde.conj().T)
    bloc21 = jnp.multiply(- C2@new_past_Psi_tilde@Psi_tilde_past_inv, new_past_Sigma_tilde)
    bloc22 = jnp.multiply(C2, new_Sigma_tilde_bloc)
    cost = w_past.conj().T@bloc11@w_past + beta * w_past.conj().T@bloc12@w_new \
        + beta *  w_new.conj().T@bloc21@w_past + w_new.conj().T@bloc22@w_new 
    return jnp.real(cost.squeeze())

# offline 
def costfunction_KL(w_theta, Sigma_tilde, Psi_tilde):
    Psi_tilde_inv = np.linalg.inv(Psi_tilde)
    cost = w_theta.conj().T@((Psi_tilde_inv*Sigma_tilde)@w_theta)
    return np.squeeze(np.real(cost))

def costfunction_KL_jax(w_theta, Sigma_tilde, Psi_tilde):
    Psi_tilde_inv = jnp.linalg.inv(Psi_tilde)
    cost = w_theta.conj().T@((Psi_tilde_inv*Sigma_tilde)@w_theta)
    return jnp.squeeze(jnp.real(cost))

