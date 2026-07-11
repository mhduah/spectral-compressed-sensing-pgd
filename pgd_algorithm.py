"""
Projected Gradient Descent (PGD) for Spectral Compressed Sensing.

Cai, Wang & Wei (2018), SIAM J. Optimization 28(3):2625-2653.

Variable conventions
--------------------
x        : signal to recover, shape (n,)
y_tilde  : preconditioned signal, y_tilde = D * x
H(x)     : Hankel matrix of x, rank r
G        : normalised Hankel lifting, G = H D^{-1}
           G(y_tilde) = H(D^{-1} D x) = H(x)
           G*(M)      = D^{-1} H*(M)

Factorisation:  H(x) = Z_U Z_V*
Recovery:       x = D^{-2} H*(Z_U Z_V*)   [NOT G*(L)]

Data fidelity:  observations are y_obs = x[Omega]
                paper works with y_tilde[Omega] = D[Omega] * y_obs
                fidelity term:  ||P_Omega(G*(L) - y_tilde)||^2

Key parameters (from paper §2.3, Algorithm 1)
---------------------------------------------
eps0    = 1/11    : inflation factor for sigma estimate
alpha_mu = 2      : safety factor for mu estimate
sigma   = sigma1(L^0) / (1 - eps0)   : inflated singular value bound
tau     = sqrt(mu * cs * r * sigma / n)  : incoherence threshold
eta     = c / sigma  for small constant c  : step size
"""

import numpy as np
from scipy.linalg import svd
import time

from utils import (
    build_hankel, hankel_adj, build_preconditioner,
    sampling_adjoint,
    project_onto_incoherence_set, compute_projection_threshold,
    relative_error
)


# ============================================================
# Internal helpers
# ============================================================

def _signal_from_factors(Z_U, Z_V, n, n1, D):
    """x = D^{-2} H*(Z_U Z_V*)"""
    return hankel_adj(Z_U @ Z_V.T.conj(), n, n1) / D**2


def _GG_star(L, n, n1, D):
    """GG*(L) = H( H*(L) / D^2 )"""
    return build_hankel(hankel_adj(L, n, n1) / D**2, n1)


def _G_star(L, n, n1, D):
    """G*(L) = D^{-1} H*(L)"""
    return hankel_adj(L, n, n1) / D


def _G_fwd(x, n1, D):
    """G(x) = H(x / D)"""
    return build_hankel(x / D, n1)


# ============================================================
# Objective
# ============================================================

def _objective(Z_U, Z_V, y_tilde, Omega, n, n1, D, p, lam):
    """
    F(Z) = ||(I-GG*)L||_F^2 + p^{-1}||P_Omega(G*L - y_tilde)||^2
         + (lam/2)||Z_U*Z_U - Z_V*Z_V||_F^2
    """
    L = Z_U @ Z_V.T.conj()
    f1 = np.linalg.norm(L - _GG_star(L, n, n1, D), 'fro')**2
    res = _G_star(L, n, n1, D)[Omega] - y_tilde
    f2 = np.sum(np.abs(res)**2) / p
    B = Z_U.T.conj() @ Z_U - Z_V.T.conj() @ Z_V
    g = 0.5 * np.linalg.norm(B, 'fro')**2
    return f1 + f2 + lam * g


# ============================================================
# Wirtinger gradient (verified correct by finite differences)
# ============================================================

def _gradient(Z_U, Z_V, y_tilde, Omega, n, n1, D, p, lam):
    """
    grad_{Z_U*} F = R Z_V + lam Z_U B
    grad_{Z_V*} F = R^H Z_U - lam Z_V B

    where R = (I-GG*)L + p^{-1} G P_Omega^* P_Omega(G*L - y_tilde)
          B = Z_U* Z_U - Z_V* Z_V
    """
    L = Z_U @ Z_V.T.conj()

    # Residual matrix R
    term1 = L - _GG_star(L, n, n1, D)
    res_vec = np.zeros(n, dtype=np.complex128)
    res_vec[Omega] = _G_star(L, n, n1, D)[Omega] - y_tilde
    term2 = _G_fwd(res_vec, n1, D) / p
    R = term1 + term2

    B = Z_U.T.conj() @ Z_U - Z_V.T.conj() @ Z_V
    return R @ Z_V + lam * (Z_U @ B), R.T.conj() @ Z_U - lam * (Z_V @ B)


# ============================================================
# Initialisation (Algorithm 1, Cai-Wang-Wei 2018)
# ============================================================

def initialize_pgd(y_obs, Omega, r, n, n1, D,
                   eps0=1/11, alpha_mu=2.0):
    """
    Spectral initialisation:
        L^0 = p^{-1} Tr_r( G P_Omega^*(y_obs) )
            = p^{-1} Tr_r( H(D^{-1} * embed(y_obs)) )

    Parameters corrected per paper §2.3:
        sigma   = sigma1(L^0) / (1 - eps0)
        mu      = alpha_mu * mu_estimated
        tau     = sqrt(mu * cs * r * sigma / n)

    Returns
    -------
    Z_U0, Z_V0 : initial factors   (shapes n1 x r, n2 x r)
    sigma      : inflated singular value bound
    mu         : inflated incoherence parameter
    """
    n2 = n + 1 - n1
    p = len(Omega) / n

    z = sampling_adjoint(y_obs, Omega, n)
    H0 = build_hankel(z / D, n1) / p           # = G P_Omega^*(y) / p

    U, s, Vh = svd(H0, full_matrices=False)
    r_eff = min(r, len(s))
    U_r  = U[:, :r_eff]
    s_r  = s[:r_eff]
    V_r  = Vh[:r_eff, :].T.conj()

    s_sqrt = np.sqrt(np.maximum(s_r, 0.0))
    Z_U0   = U_r * s_sqrt[np.newaxis, :]
    Z_V0   = V_r * s_sqrt[np.newaxis, :]

    sigma1_raw = s_r[0]

    # Incoherence with safety factor
    mu_U = n1 * np.max(np.sum(np.abs(U_r)**2, axis=1)) / r_eff
    mu_V = n2 * np.max(np.sum(np.abs(V_r)**2, axis=1)) / r_eff
    mu   = alpha_mu * max(mu_U, mu_V, 1.0)

    # Inflated sigma (paper eq. after Alg 1)
    sigma = sigma1_raw / (1.0 - eps0)

    return Z_U0, Z_V0, sigma, mu


# ============================================================
# Main PGD recovery
# ============================================================

def pgd_recovery(y_obs, Omega, r, n, n1=None,
                 lam=0.25, eta_scale=0.01,
                 max_iter=2000, tol=1e-6,
                 x_true=None, verbose=False, return_history=False,
                 rng=None):
    """
    Projected Gradient Descent for Spectral Compressed Sensing.

    Parameters
    ----------
    y_obs      : ndarray (m,)   observations  x_true[Omega]
    Omega      : ndarray (m,)   sampling indices (0-based)
    r          : int            target rank
    n          : int            signal length
    n1         : int, optional  Hankel row count (default n//2+1)
    lam        : float          balancing weight (default 0.25)
    eta_scale  : float          step size = eta_scale / sigma  (default 0.01)
    max_iter   : int            iteration cap
    tol        : float          success threshold on relative error
    x_true     : ndarray        ground truth (for monitoring / stopping)
    verbose    : bool
    return_history : bool

    Returns
    -------
    result : dict
        'x'          recovered signal
        'rel_error'  relative error (if x_true provided)
        'converged'  bool
        'iterations' int
        'time'       wall seconds
        'history'    dict (if return_history)
    """
    t0 = time.time()

    if n1 is None:
        n1 = n // 2 + 1
    n2 = n + 1 - n1
    p  = len(Omega) / n

    D = build_preconditioner(n, n1)
    y_tilde = D[Omega] * y_obs          # preconditioned observations

    # ---- Initialisation ----
    Z_U, Z_V, sigma, mu = initialize_pgd(y_obs, Omega, r, n, n1, D)

    tau = compute_projection_threshold(mu, r, sigma, n, n1, n2)
    Z_U, Z_V = project_onto_incoherence_set(Z_U, Z_V, tau)

    eta = eta_scale / sigma             # fixed step size

    history  = {'rel_errors': [], 'obj_values': []}
    best_err = np.inf
    best_x   = _signal_from_factors(Z_U, Z_V, n, n1, D).copy()
    converged = False

    for k in range(max_iter):
        gU, gV = _gradient(Z_U, Z_V, y_tilde, Omega, n, n1, D, p, lam)
        Z_U -= eta * gU
        Z_V -= eta * gV
        Z_U, Z_V = project_onto_incoherence_set(Z_U, Z_V, tau)

        x_curr = _signal_from_factors(Z_U, Z_V, n, n1, D)

        if x_true is not None:
            err = relative_error(x_curr, x_true)
            if err < best_err:
                best_err = err
                best_x   = x_curr.copy()
            if return_history:
                history['rel_errors'].append(err)
            if verbose and k % 100 == 0:
                print(f"  k={k:5d}  err={err:.3e}")
            if err < tol:
                converged = True
                break
        else:
            best_x = x_curr
            if return_history:
                f = _objective(Z_U, Z_V, y_tilde, Omega, n, n1, D, p, lam)
                history['obj_values'].append(float(f.real))

    result = {
        'x': best_x,
        'converged': converged,
        'iterations': k + 1,
        'time': time.time() - t0,
        'sigma': sigma, 'mu': mu, 'tau': tau, 'eta': eta,
    }
    if x_true is not None:
        result['rel_error'] = relative_error(best_x, x_true)
    if return_history:
        result['history'] = history

    return result


# ============================================================
# Multiple restarts
# ============================================================

def pgd_with_restarts(y_obs, Omega, r, n, n1=None, lam=0.25,
                      max_iter=2000, tol=1e-6, num_restarts=3,
                      x_true=None, verbose=False, rng=None):
    """PGD with multiple restarts; return best result."""
    best_result = None
    best_error  = np.inf
    for restart in range(num_restarts):
        result = pgd_recovery(y_obs, Omega, r, n, n1=n1, lam=lam,
                              max_iter=max_iter, tol=tol,
                              x_true=x_true, verbose=verbose)
        err = result.get('rel_error', np.inf)
        if err < best_error:
            best_error  = err
            best_result = result
        if best_error < tol:
            break
    best_result['num_restarts'] = restart + 1
    return best_result
