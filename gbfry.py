import numpy as np
import numpy.random as npr
import numba as nb
from etstablernd import etstablernd

@nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8, nb.i4), fastmath=True)
def GGPsumrnd(eta, sigma, c, n):
    if sigma < 1e-8:
        S = np.zeros(n)
        for i in range(n):
            S[i] = npr.gamma(eta, 1./c)
        return S
    else:
        return etstablernd(eta/sigma, sigma, c, n)

@nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8, nb.i4), fastmath=True)
def GBFRYsumrndold(eta, tau, sigma, c, n):
    kappa = tau - sigma
    S = GGPsumrnd(eta*c**kappa/kappa, sigma, c, n)
    for i in range(n):
        K = npr.poisson(eta*c**tau/tau/kappa)
        for _ in range(K):
            S[i] += npr.gamma(1-sigma, 1)/(npr.beta(tau, 1)*c + 1e-10)
    return S

@nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8, nb.i4), fastmath=True)
def GBFRYsumrnd(eta, tau, sigma, c, n):
    S = GGPsumrnd(eta/c**sigma, sigma, c, n)
    for i in range(n):
        K = npr.poisson(eta/tau)
        for _ in range(K):
            S[i] += npr.gamma(1-sigma, 1)/(npr.beta(tau, 1)*c + 1e-20)
    return S

# @nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8, nb.i4), fastmath=True)
# def GBFRYsumrndparts(eta, tau, sigma, c, n):
#     S1 = GGPsumrnd(eta/c**sigma, sigma, c, n)
#     S2 = np.zeros(n)
#     for i in range(n):
#         K = npr.poisson(eta/tau)
#         for _ in range(K):
#             S2[i] += npr.gamma(1-sigma, 1)/(npr.beta(tau, 1)*c + 1e-20)
#     return np.column_stack((S1, S2))

@nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8, nb.i4), fastmath=True)
def GGPsumrndnew(eta, sigma, c, n):
    if sigma < 1e-8:
        S = np.zeros(n)
        for i in range(n):
            K = npr.poisson(eta*c**sigma/(-1*sigma))
            for _ in range(K):
                S[i] += npr.gamma(-sigma, 1./c)
        return S
    else:
        return etstablernd(eta/sigma, sigma, c, n)

@nb.njit(nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8, nb.i4), fastmath=True)
def GBFRYsumrndnew(eta, tau, sigma, c, n):
    S = GGPsumrndnew(eta/c**sigma, sigma, c, n)
    for i in range(n):
        K = npr.poisson(eta/tau)
        for _ in range(K):
            S[i] += npr.gamma(1-sigma, 1)/(npr.beta(tau, 1)*c + 1e-20)
    return S

#@nb.njit(nb.f8[:, :](nb.f8, nb.f8, nb.f8, nb.f8, nb.i4, nb.i4), fastmath=True)
def GBFRYsumrndparts(eta, tau, sigma, c, n, parts):
    S = np.zeros(n)
    eta_new = eta
    for _ in range(parts):
        Si = GGPsumrndnew(eta_new/c**sigma, sigma, c, n)
        eta_old = eta_new
        eta_new = eta_old*(1 - sigma)/(tau + 1 - sigma)
        sigma -= 1
        S = np.column_stack((S, Si))
    S2 = np.zeros(n)
    for i in range(n):
        K = npr.poisson(eta_old/tau)
        for _ in range(K):
            S2[i] += npr.gamma(-sigma, 1)/(npr.beta(tau, 1)*c + 1e-20)
    S_final = np.column_stack((S, S2))
    return S_final[:, 1:]
