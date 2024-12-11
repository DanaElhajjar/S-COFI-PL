# -----------------------------------------------------------------
# Librairies
# -----------------------------------------------------------------
import numpy as np
from manifold import ComplexCircle
from cost_functions import *
from gradients import *
import jax.numpy as jnp
from jaxopt import BacktrackingLineSearch
from jax import grad, jit
import time as t
from jax.config import config
config.update("jax_enable_x64",True)

# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------

# Riemannian gradient descent
# ---------------------------

## Frobenius norm
#################

def riem_new_w_LS(w_new, w_past, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_gd, alpha, auto):
    """ Riemannian gradient for least square cost function
        Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Psi_tilde_past : Coherence matrix of past data
            * Sigma_tilde_past : Covariance matrix of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * iter_max_gd : number of gradient descent iterations
            * alpha : gradient descent step
            * auto : T/F 

        Outputs:
            * w_new : vector of the phases
            * cost_w : cost value 
    """
    cost_w = []
    k = w_new.shape[0]
    it = 0
    err = 1
    # manifold 
    CC = ComplexCircle(k)
    while err > 1e-4 and it<iter_max_gd:
        if auto:
            gradient_w_euc_jax = jit(grad(cost_fct_LS_jax, argnums=(0)))(w_new, w_past, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde).conj()
            gradient_w_euc = np.array(gradient_w_euc_jax)
        else:
            gradient_w_euc = gradient_new_w_LS(w_new, w_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde)
        
        # grad euclidien to grad riemannien
        grad_riemann = CC.projection(w_new, gradient_w_euc)

        # iterations
        w_new_riem = CC.retraction(w_new, -alpha*grad_riemann)

        # error computation
        err = np.linalg.norm(w_new_riem-w_new)
        
        # update
        w_new = w_new_riem

        # cost function value
        cost_w.append(cost_fct_LS(w_new, w_past, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde))
        
        it = it+1
    return (w_new, cost_w)

def riem_new_w_LS_regularized(w_new, w_past, Sigma_tilde, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_gd, alpha, beta, auto):
    """ Riemannian gradient for least square cost function with a shrinkage to identity covariance matrix
        Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Psi_tilde_past : Coherence matrix of past data
            * Sigma_tilde_past : Covariance matrix of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * iter_max_gd : number of gradient descent iterations
            * alpha : gradient descent step
            * beta : regularization parameter
            * auto : T/F 

        Outputs:
            * w_new : vector of the phases
            * cost_w : cost value 
    """
    cost_w = []
    k = w_new.shape[0]
    it = 0
    err = 1
    # manifold 
    CC = ComplexCircle(k)
    while err > 1e-4 and it<iter_max_gd:
        if auto:
            gradient_w_euc_jax =  jit(grad(cost_fct_LS_regularized_jax, argnums=(0)))(w_new, w_past, Sigma_tilde, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta).conj()
            gradient_w_euc = np.array(gradient_w_euc_jax)
        else:
            gradient_w_euc = gradient_new_w_LS_regularized(w_new, w_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta)
        
        # grad euclidien to grad riemannien
        grad_riemann = CC.projection(w_new, gradient_w_euc)

        # iterations
        w_new_riem = CC.retraction(w_new, -alpha*grad_riemann)

        # error computation
        err = np.linalg.norm(w_new_riem-w_new)
        
        # update
        w_new = w_new_riem

        # cost function value
        cost_w.append(cost_fct_LS_regularized(w_new, w_past, Sigma_tilde, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta))
        
        it = it+1
    return (w_new, cost_w)

grad_euc_LS_func = jit(grad(costfunction_LS_jax, argnums=[1]))
def RG_LS_IPL(Sigma_tilde,maxIter, alpha, auto):

    # Riemannian Gradient for Least Squares cost function
    # Data :
    #       * Sigma_tilde : covariance estimation (possibily after a regularization step)
    #       * maxIter : maximum iteration of the gradient descent
    #       * auto : True : computation of the gradient by jax if true. Else analytical gradient
    # Output : w_theta : complex vector of phasis 

    p = Sigma_tilde.shape[0]
    Psi_tilde = np.abs(Sigma_tilde)
    w_theta = np.ones((p,1),dtype=complex)
    w_theta_jnp = jnp.array(w_theta)
    Sigma_tilde_jnp = jnp.array(Sigma_tilde)
    Psi_tilde_jnp = jnp.abs(Sigma_tilde_jnp)
    
    cost = []
    cost_temp = costfunction_LS(w_theta, Sigma_tilde, Psi_tilde)
    cost.append(cost_temp)

    CC = ComplexCircle(p)

    # gradient Riemmannien
    err = 1
    it = 1
    # alpha_0 = 0.01441152 # 1e-2
    while err > 1e-4 and it<maxIter:
        # calcul gradient euclidien
        if auto:
            grad_euc_jax = grad_euc_LS_func(Sigma_tilde_jnp, w_theta_jnp, Psi_tilde_jnp)[0].conj()
            grad_euc = np.array(grad_euc_jax)
            # print('grad auto = ',grad_euc)
            # grad_euc = grad_costfunction_LS(Sigma_tilde, w_theta)
            # print('grad ana = ',grad_euc)
        else:
            grad_euc = grad_costfunction_LS(Sigma_tilde, w_theta)

        # grad euclidien to grad riemannien
        grad_riemann = CC.projection(w_theta,grad_euc)

        # iterations
        w_theta_new = CC.retraction(w_theta,-alpha*grad_riemann)

        # calcul de l'erreur
        err = np.linalg.norm(w_theta_new-w_theta)

        # maj
        w_theta = w_theta_new
        w_theta_jnp = jnp.array(w_theta)
        it = it +1
        cost_temp = costfunction_LS(w_theta, Sigma_tilde, Psi_tilde)
        cost.append(cost_temp)

    return(w_theta, cost)


# def riem_new_w_LS_shrinkageToTapering(w_new, w_past, tapering, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_gd, alpha, beta, auto):
#     """ Riemannian gradient for least square cost function with a shrinkage to tapering covariance matrix
#         Inputs :
#             * w_new : phases vector of new data
#             * w_past : phases vector of past data
#             * Psi_tilde_past : Coherence matrix of past data
#             * Sigma_tilde_past : Covariance matrix of past data
#             * new_past_Psi_tilde : Coherence vector between past and new data
#             * new_past_Sigma_tilde : Covariance vector between past and new data
#             * new_Psi_tilde : Coherence matrix of the new data
#             * new_Sigma_tilde : Covariance matrix of the new data
#             * iter_max_gd : number of gradient descent iterations
#             * alpha : gradient descent step
#             * beta : regularization parameter
#             * auto : T/F 

#         Outputs:
#             * w_new : vector of the phases
#             * cost_w : cost value 
#     """
#     cost_w = []
#     k = w_new.shape[0]
#     it = 0
#     err = 1
#     # manifold 
#     CC = ComplexCircle(k)
#     while err > 1e-4 and it<iter_max_gd:
#         if auto:
#             gradient_w_euc_jax =  jit(grad(cost_fct_LS_ShrinkageToTapering_jax, argnums=(0)))(w_new, w_past, tapering, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta).conj()    
#             gradient_w_euc = np.array(gradient_w_euc_jax)
#         else:
#             gradient_w_euc = gradient_new_w_LS_ShrinkageToTapering(w_new, w_past, tapering, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta)
        
#         # grad euclidien to grad riemannien
#         grad_riemann = CC.projection(w_new, gradient_w_euc)

#         # iterations
#         w_new_riem = CC.retraction(w_new, -alpha*grad_riemann)

#         # error computation
#         err = np.linalg.norm(w_new_riem-w_new)
        
#         # update
#         w_new = w_new_riem

#         # cost function value
#         cost_w.append(cost_fct_LS_ShrinkageToTapering_jax(w_new, w_past, tapering, Psi_tilde_past, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta))
        
#         it = it+1
#     return (w_new, cost_w)


## Kullback Leibler divergence
##############################

def riem_new_w_KL(w_new, w_past, Psi_tilde_past, Psi_tilde_past_inv, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_gd, alpha, auto):
    """ Riemannian gradient for KullBack-Leibler cost function
        Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Psi_tilde_past : Coherence matrix of past data
            * Sigma_tilde_past : Covariance matrix of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * iter_max_gd : number of gradient descent iterations
            * alpha : gradient descent step
            * auto : T/F 

        Outputs:
            * w_new : vector of the phases
            * cost_w : cost value 
    """
    cost_w = []
    k = w_new.shape[0]
    it = 0
    err = 1
    # manifold
    CC = ComplexCircle(k)
    while err > 1e-4 and it<iter_max_gd:
        if auto:
            pass
        else:
            gradient_w_euc = gradient_new_w_KL(w_new, w_past, Psi_tilde_past_inv, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde)

        # grad euclidien to grad riemannien
        grad_riemann = CC.projection(w_new, gradient_w_euc)

        # iterations
        w_new_riem = CC.retraction(w_new, -alpha*grad_riemann)

        # calcul de l'erreur
        err = np.linalg.norm(w_new_riem-w_new)
        
        w_new = w_new_riem
        # w_new = w_new.reshape((k,))

        cost_w.append(cost_fct_KL(w_new, w_past, Psi_tilde_past, Psi_tilde_past_inv, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde))
        it = it+1    
    return (w_new, cost_w)

def riem_new_w_KL_regularized(w_new, w_past, Sigma_tilde, Psi_tilde_past, Psi_tilde_past_inv, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_gd, alpha, beta, auto):
    """ Riemannian gradient for KullBacl-Leibler cost function with a shrinkage to identity covariance matrix
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
            * iter_max_gd : number of gradient descent iterations
            * alpha : gradient descent step
            * beta : regularization parameter
            * auto : T/F 

        Outputs:
            * w_new : vector of the phases
            * cost_w : cost value 
    """
    cost_w = []
    k = w_new.shape[0]
    it = 0
    err = 1
    # manifold
    CC = ComplexCircle(k)
    while err > 1e-4 and it<iter_max_gd:
        if auto:
            pass
        else:
            gradient_w_euc = gradient_new_w_KL_regularized(w_new, w_past, Sigma_tilde, Psi_tilde_past_inv, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta)

        # grad euclidien to grad riemannien
        grad_riemann = CC.projection(w_new, gradient_w_euc)

        # iterations
        w_new_riem = CC.retraction(w_new, -alpha*grad_riemann)

        # calcul de l'erreur
        err = np.linalg.norm(w_new_riem-w_new)
        
        w_new = w_new_riem
        # w_new = w_new.reshape((k,))

        cost_w.append(cost_fct_KL_regularized(w_new, w_past, Sigma_tilde, Psi_tilde_past, Psi_tilde_past_inv, Sigma_tilde_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, beta))
        it = it+1    
    return (w_new, cost_w)



grad_euc_KL_func = jit(grad(costfunction_KL_jax, argnums=[0]))
def RG_KL_IPL(Sigma_tilde,maxIter,alpha_0, auto):

    # Riemannian Gradient for KL cost function
    # Data :
    #       * Sigma_tilde : covariance estimation (possibily after a regularization step)
    #       * maxIter : maximum iteration of the gradient descent
    #       * auto : True : computation of the gradient by jax if true. Else analytical gradient
    # Output : w_theta : complex vector of phasis 

    p = Sigma_tilde.shape[0]
    Psi_tilde = np.abs(Sigma_tilde)
    w_theta = np.ones((p,1),dtype=complex)
    w_theta_jnp = jnp.array(w_theta)
    Sigma_tilde_jnp = jnp.array(Sigma_tilde)
    Psi_tilde_jnp = jnp.abs(Sigma_tilde_jnp)
    
    cost = []
    cost_temp = costfunction_KL(w_theta, Sigma_tilde, Psi_tilde)
    cost.append(cost_temp)

    # Manifold
    CC = ComplexCircle(p)

    # gradient Riemmannien
    err = 1
    it = 1
    # alpha_0 = 0.107 # 1e-2
    while err > 1e-4 and it<maxIter:
        # calcul gradient euclidien
        if auto:
            grad_euc_jax = grad_euc_KL_func(Sigma_tilde_jnp, w_theta_jnp, Psi_tilde_jnp)[0].conj()
            grad_euc = np.array(grad_euc_jax)
        else:
            grad_euc = grad_costfunction_KL(w_theta, Sigma_tilde)

        # grad euclidien to grad riemannien
        grad_riemann = CC.projection(w_theta,grad_euc)

        # iterations
        w_theta_new = CC.retraction(w_theta,-alpha_0*grad_riemann)

        # calcul de l'erreur
        err = np.linalg.norm(w_theta_new-w_theta)

        # maj
        w_theta = w_theta_new
        w_theta_jnp = jnp.array(w_theta)
        it = it +1
        cost_temp = costfunction_KL(w_theta, Sigma_tilde, Psi_tilde)
        cost.append(cost_temp)

    return(w_theta, cost)

# Majorization - Minimzation 
# --------------------------

## Frobenius norm
#################

def MM_LS_new_w(new_w, w_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_MM):
    """ Majorization Minimization problem for least square (LS) optimization problem
        Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * iter_max_MM : number of MM algorithm iterations

        Outputs : 
            * new_w : vector of the phases
    """
    M = 4 * np.multiply(new_past_Psi_tilde, new_past_Sigma_tilde)@w_past
    N = 4 * np.multiply(new_Psi_tilde, new_Sigma_tilde)
    for _ in range(iter_max_MM):
        tilde_w = M + N@new_w
        new_w = np.exp(1j*np.angle(tilde_w))
    return new_w

def MM_LS_new_w_regularized(new_w, w_past, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, Sigma_tilde, iter_max_MM, beta):
    """ Majorization Minimization problem for least square (LS) optimization problem with a shrinkage to identity matrix
        Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * iter_max_MM : number of MM algorithm iterations

        Outputs : 
            * new_w : vector of the phases
    """
    p = w_past.shape[0]
    k = new_w.shape[0]
    I_k = np.eye(k, dtype=np.complex128)
    element2 = (1 - beta) * np.trace(Sigma_tilde) / (p+k)
    M = 4 * beta * np.multiply(new_past_Psi_tilde, new_past_Sigma_tilde)@w_past
    N = 4 * np.multiply(new_Psi_tilde, beta * new_Sigma_tilde + element2 * I_k)
    for _ in range(iter_max_MM):
        tilde_w = M + N@new_w
        new_w = np.exp(1j*np.angle(tilde_w))
    return new_w

def MM_LS_IPL(Sigma_tilde,maxIter):
    # MM LS cost function
    # Data :
    #       * Sigma_tilde : covariance estimation (possibily after a regularization step)
    #       * maxIter : maximum iteration of the gradient descent
    # Output : w_theta : complex vector of phasis 
    p = Sigma_tilde.shape[0]
    M = np.multiply(abs(Sigma_tilde),Sigma_tilde)
    
    w = np.ones((p,1))
    
    for i in range (maxIter):
        tilde_w = M@w 
        w = np.exp(1j*np.angle(tilde_w))
    return w

# def MM_LS_new_w_ShrinkageToTapering(new_w, w_past, tapering_matrix, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_MM, beta):
#     """ Majorization Minimization problem for least square (LS) optimization problem with a shrinkage to tapering matrix
#         Inputs :
#             * w_new : phases vector of new data
#             * w_past : phases vector of past data
#             * new_past_Psi_tilde : Coherence vector between past and new data
#             * new_past_Sigma_tilde : Covariance vector between past and new data
#             * new_Psi_tilde : Coherence matrix of the new data
#             * new_Sigma_tilde : Covariance matrix of the new data
#             * iter_max_MM : number of MM algorithm iterations

#         Outputs : 
#             * new_w : vector of the phases
#     """
#     p = w_past.shape[0]
#     k = new_w.shape[0]
#     new_past_tapering_lower= tapering_matrix[p:p+k, 0:p]
#     new_tapering = tapering_matrix[p:p+k, p:p+k]
#     new_past_Sigma_tilde_bloc = beta * new_past_Sigma_tilde + (1 - beta) * np.multiply(new_past_tapering_lower, new_past_Sigma_tilde)
#     new_Sigma_tilde_bloc = beta * new_Sigma_tilde + (1 - beta) * np.multiply(new_tapering, new_Sigma_tilde)
#     M = 4  * np.multiply(new_past_Psi_tilde, new_past_Sigma_tilde_bloc)@w_past
#     N = 4 * np.multiply(new_Psi_tilde, new_Sigma_tilde_bloc)
#     for _ in range(iter_max_MM):
#         tilde_w = M + N@new_w
#         new_w = np.exp(1j*np.angle(tilde_w))
#     return new_w

## Kullback Leibler divergence
##############################

def MM_KL_new_w(new_w, w_past, Psi_tilde_past_inv, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, iter_max_MM):
    """ Majorization Minimization problem for KullBack-Leibler divergence (KL) optimization problem
        Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Psi_tilde_past : Coherence matrix of the past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * iter_max_MM : number of MM algorithm iterations

        Outputs : 
            * new_w : vector of the phases
    """
    k = new_w.shape[0]
    I_k = np.eye(k)
    new_w = new_w.reshape((k, 1))
    # Psi_tilde_past_inv = np.linalg.inv(Psi_tilde_past)
    C2 = np.linalg.inv(new_Psi_tilde - new_past_Psi_tilde@Psi_tilde_past_inv@new_past_Psi_tilde.T)
    M = 2 * np.multiply(C2, new_Sigma_tilde)
    _, D, _ = np.linalg.svd(M)
    lambda_max = D[0]
    M_lambdaI_minus = M - lambda_max * I_k
    bloc_past = 2 * np.multiply(C2@new_past_Psi_tilde@Psi_tilde_past_inv, new_past_Sigma_tilde)@w_past 

    for _ in range(iter_max_MM):
        tilde_w = bloc_past - M_lambdaI_minus@new_w
        new_w = np.exp(1j * np.angle(tilde_w))
    return new_w
    
def MM_KL_new_w_regularized(new_w, w_past, Psi_tilde_past_inv, new_past_Psi_tilde, new_past_Sigma_tilde, new_Psi_tilde, new_Sigma_tilde, Sigma_tilde, iter_max_MM, beta):
    """ Majorization Minimization problem for KullBack-Leibler divergence (KL) optimization problem with a shrinkage to identity covariance matrix
        Inputs :
            * w_new : phases vector of new data
            * w_past : phases vector of past data
            * Psi_tilde_past : Coherence matrix of the past data
            * new_past_Psi_tilde : Coherence vector between past and new data
            * new_past_Sigma_tilde : Covariance vector between past and new data
            * new_Psi_tilde : Coherence matrix of the new data
            * new_Sigma_tilde : Covariance matrix of the new data
            * Sigma_tilde : Covariance matrix of all data
            * iter_max_MM : number of MM algorithm iterations

        Outputs : 
            * new_w : vector of the phases
    """
    p = w_past.shape[0]
    k = new_w.shape[0]
    I_k = np.eye(k)
    alpha = (1 - beta) * np.trace(Sigma_tilde) / (p+k)
    new_w = new_w.reshape((new_w.shape[0], 1))
    # Psi_tilde_past_inv = np.linalg.inv(Psi_tilde_past)
    # new_Psi_tilde_inv = np.linalg.inv(new_Psi_tilde)
    C2 = np.linalg.inv(new_Psi_tilde - new_past_Psi_tilde@Psi_tilde_past_inv@new_past_Psi_tilde.T)
    M = np.multiply(C2, beta * new_Sigma_tilde + alpha * I_k)
    _, D, _ = np.linalg.svd(M)
    lambda_max = D[0]
    M_lambdaI_minus = M - lambda_max * I_k 
    bloc_past = 2 * beta *  np.multiply(C2@new_past_Psi_tilde@Psi_tilde_past_inv, new_past_Sigma_tilde)@w_past 

    for _ in range(iter_max_MM):
        tilde_w = bloc_past - 2 * M_lambdaI_minus@new_w
        new_w = np.exp(1j * np.angle(tilde_w))
    return new_w


def MM_KL_IPL(Sigma_tilde,maxIter):
    # MM KL cost function
    # Data :
    #       * Sigma_tilde : covariance estimation (possibily after a regularization step)
    #       * maxIter : maximum iteration of the gradient descent
    # Output : w_theta : complex vector of phasis 
        
        p = Sigma_tilde.shape[0]
        M = np.multiply(np.linalg.inv(abs(Sigma_tilde)),Sigma_tilde)
        _, D, _ = np.linalg.svd(M)
        lambdamax = D[0]
        lambdaI_minus = lambdamax*(np.eye(p)) - M
        
        w = np.ones((p,1))
        
        for i in range (maxIter):
            tilde_w = lambdaI_minus@w 
            w = np.exp(1j*np.angle(tilde_w))
        return w