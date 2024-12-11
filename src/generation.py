# -----------------------------------------------------------------
# Librairies
# -----------------------------------------------------------------
import numpy as np
import numpy.random as random
import scipy as sc

seed = 7777
rng = np.random.default_rng(seed)

# -----------------------------------------------------------------
# Functions
# -----------------------------------------------------------------
def generate_data_for_estimation_real(p, N, 𝛍, 𝚺):
    """ A function that genrate real gaussian samples
        Inputs : 
            * p : size of a vector data
            * N : number of observations
            * 𝛍 : mean
            * 𝚺 : covariance matrix
        Output : 
            * X : real data matrix of shape (p, n) """
    𝐗 = np.empty((p,N), dtype=float)
    𝐗 = np.random.multivariate_normal(𝛍, 𝚺, N).T
    return 𝐗

def multivariate_complex_normal_samples(mean, covariance, N, pseudo_covariance=0):
    """ A function to generate multivariate complex normal vectos as described in:
        Picinbono, B. (1996). Second-order complex random vectors and normal
        distributions. IEEE Transactions on Signal Processing, 44(10), 2637–2640.
        Inputs:
            * mean = vector of size p, mean of the distribution
            * covariance = the covariance matrix of size p*p(Gamma in the paper)
            * pseudo_covariance = the pseudo-covariance of size p*p (C in the paper)
                for a circular distribution omit the parameter
            * N = number of Samples
        Outputs:
            * Z = Samples from the complex Normal multivariate distribution, size p*N"""

    (p, p) = covariance.shape
    Gamma = covariance
    C = pseudo_covariance

    # Computing elements of matrix Gamma_2r
    Gamma_x = 0.5 * np.real(Gamma + C)
    Gamma_xy = 0.5 * np.imag(-Gamma + C)
    Gamma_yx = 0.5 * np.imag(Gamma + C)
    Gamma_y = 0.5 * np.real(Gamma - C)

    # Matrix Gamma_2r as a block matrix
    Gamma_2r = np.block([[Gamma_x, Gamma_xy], [Gamma_yx, Gamma_y]])

    # Generating the real part and imaginary part
    mu = np.hstack((mean.real, mean.imag))
    v = np.random.multivariate_normal(mu, Gamma_2r, N).T
    X = v[0:p, :]
    Y = v[p:, :]
    return X + 1j * Y


def generate_data_for_estimation_complex(p, N, 𝛍, 𝚺, pseudo_𝚺):
    """ A function that generate complex gaussian samples
        Inputs : 
            * p : size of a vector data
            * N : number of observations
            * 𝛍 : mean
            * 𝚺 : covariance matrix
            * pseudo_𝚺 : pseudo covariance matrix
        Output : 
            * X : complex data matrix of shape (p, n) """
    𝐗 = np.empty((p,N), dtype=complex)
    𝐗 = multivariate_complex_normal_samples(𝛍, 𝚺, N, pseudo_𝚺)
    return 𝐗


def generate_scaled_gaussian_data_for_estimation_real(p, n, mean, Sigma, gamma_shape):
    """ A function that generate real scaled gaussian samples
        Inputs : 
            * p : size of a vector data
            * N : number of observations
            * mean : mean
            * Sigma : covariance matrix
            * gamma_shape : gamma shape of tau's distribution
        Output : 
            * X : real data matrix of shape (p, n)
            * tau_true : tau's distribution """
    Y = generate_data_for_estimation_real(p, n, mean, Sigma)
    tau_true = rng.gamma(gamma_shape, 1/gamma_shape, size=(n,))
    X = Y * np.sqrt(tau_true) # Y.T * np.sqrt(tau_true).reshape((n, 1))
    # X = X.T # in order to have (p, n) matrix
    return (X, tau_true)


def generate_scaled_gaussian_data_for_estimation_complex(p, n, mean, Sigma, pseudo_Sigma, gamma_shape):
    """ A function that generate real scaled gaussian samples
        Inputs : 
            * p : size of a vector data
            * N : number of observations
            * mean : mean
            * Sigma : covariance matrix
            * gamma_shape : gamma shape of tau's distribution
        Output : 
            * X : real data matrix of shape (p, n)
            * tau_true : tau's distribution """
    Y = generate_data_for_estimation_complex(p, n, mean, Sigma, pseudo_Sigma)
    tau_true = rng.gamma(gamma_shape, 1/gamma_shape, size=(n,))
    X = Y * np.sqrt(tau_true) # Y.T * np.sqrt(tau_true).reshape((n, 1))
    # X = X.T # in order to have (p, n) matrix
    return (X, tau_true)

def simulateLRSigma(Sigma, rank):
    """ A function that simulate a low-rank Sigma
    Inputs :
        * Sigma : the real core of the covariance matrix
        * rank : integer 
    Outputs : 
        * low-rank structure of the input matrix Sigma
    """
    u,s,vh = np.linalg.svd(Sigma)
    u_signal = u[:,:rank]
    u_noise = u[:,rank:]
    sigma = np.mean(s[rank:])
    return u_signal @ np.diag(s[:rank])@u_signal.conj().T + sigma*u_noise@u_noise.conj().T

def simulateCov(trueSigma, truetheta):
    """ A function that simulate the true covariance matrix
    Inputs : 
        * trueSigma : the true core of the covariance matrix
        * truetheta : the true phase values
    Outputs : 
        * trueCov : the true covariance matrix """
    diag_theta = np.diag(np.exp(np.dot(1j,truetheta)))
    truecov = (diag_theta.dot(trueSigma).dot(diag_theta.conj().T))
    return truecov

def simulateMV(covariance, N,L):
    """ A function that simulate complex gaussian data 
    Inputs : 
        * covariance : the covariance matrix
        * N : integer
        * L : integer
    Outputs : 
        * X : vector of data 
        """
    Csqrt = sc.linalg.sqrtm(covariance)
    X = np.dot(Csqrt*np.sqrt(1/2),(random.randn(N,L) +1j*random.randn(N,L)))
    return X

def simulateMVscaledG(covariance, nu, N,L):
    """ A function that simulate complex non gaussian data 
    Inputs : 
        * covariance : the covariance matrix
        * nu : paramter for the gamma distribution
        * N : integer
        * L : integer
    Outputs : 
        * X : vector of gaussian data 
        * Y : vector of scaled gaussian data
        """
    Csqrt = sc.linalg.sqrtm(covariance)
    X = np.dot(Csqrt*np.sqrt(1/2),(random.randn(N,L) +1j*random.randn(N,L)))
    tau = random.gamma(nu,1/nu, L ) # size=(L,))
    tau_mat = np.tile(tau,(N,1))
    Y = np.multiply(X,np.sqrt(tau_mat))
    return X, Y

def phasegeneration(choice,p):
    """ A function that generate phase differences
    Inputs : 
        * choice : random or linear
        * N : integer
    Outputs : 
        * delta_thetasim : vector of phase differences """
    if choice == 'random':
        theta_sim = np.array([random.uniform(-np.pi,np.pi) for i in range(p)])
        delta_thetasim0 = np.array((theta_sim-theta_sim[0]))
        delta_thetasim = np.angle(np.exp(1j*delta_thetasim0))
    elif choice[0]  == 'linear':
        thetastep = choice[1]
        delta_thetasim = np.linspace(0,thetastep,p)
    return delta_thetasim

def sampledistributionchoice(trueC,p, N,sampledist):
    """ A function that generate data based on the choice of the distribution (Gaussian or Scaled Gaussian)
    Inputs : 
        * trueC : the true covariance martix
        * size : tuple of the size
        * sample_dist : 'Gaussian' or '(ScaledGaussian, nu)'
    Outputs : 
        * X : data vector"""
    if sampledist == 'Gaussian':
        X = simulateMV(trueC,p,N) # simulate Gaussian distribution
        return X
    elif sampledist[0] == 'ScaledGaussian':
        _,X = simulateMVscaledG(trueC,sampledist[1],p, N) #simulate non-Gaussian distribution
        return X