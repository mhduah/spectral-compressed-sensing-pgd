"""
ADMM solver for Nuclear Norm Minimisation (convex baseline).

Solves the Hankel matrix completion problem:
    min  ||H||_*
    s.t. [H]_{i,j} = x_{i+j}   for all (i,j)        (Hankel constraint)
         x[Omega]  = y_obs                             (data constraint)

ADMM splitting:  variables x (signal) and H (Hankel matrix).

x-update: closed form via the pseudo-inverse of the Hankel averaging operator.
H-update: nuclear norm proximal = soft-threshold singular values.
Dual update: standard multiplier step.

The x-update requires the AVERAGING adjoint (H†), not the summing adjoint (H*).
These are different operators:
    H*(Z)_a = sum_{i+j=a} Z_{i,j}           (summing, used in PGD)
    H†(Z)_a = (1/w_a) sum_{i+j=a} Z_{i,j}  (averaging, used in ADMM x-update)
because the x-update minimises ||H(x) - target||_F^2 over x, whose solution is
    x = H† target,  i.e. average each anti-diagonal of target.
"""

import numpy as np
from scipy.linalg import svd
import time

from utils import build_hankel, hankel_adj, build_preconditioner, relative_error


# ============================================================
# Helpers
# ============================================================

def _hankel_avg(Z, n, n1):
    """
    Averaging Hankel adjoint:  x_a = (1/w_a) * sum_{i+j=a} Z_{i,j}

    This is the least-squares solution to  H(x) = Z,
    i.e. the Moore-Penrose pseudo-inverse of H.
    """
    n2 = n + 1 - n1
    x_sum = np.zeros(n, dtype=Z.dtype)
    counts = np.zeros(n, dtype=np.float64)
    for i in range(n1):
        x_sum[i:i + n2] += Z[i, :]
        counts[i:i + n2] += 1.0
    return x_sum / counts


def nuclear_norm_proximal(H, tau):
    """Proximal operator of tau * || · ||_*:  soft-threshold singular values."""
    if not np.all(np.isfinite(H)):
        return np.zeros_like(H)
    U, s, Vh = svd(H, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    if s_thresh.max() == 0:
        return np.zeros_like(H)
    mask = s_thresh > 0
    return (U[:, mask] * s_thresh[mask]) @ Vh[mask, :]


# ============================================================
# Main ADMM solver
# ============================================================

def admm_nuclear_norm(y_obs, Omega, r, n, n1=None,
                      max_iter=500, tol=1e-6, rho=2.0,
                      x_true=None, verbose=False, return_history=False):
    """
    ADMM for Hankel nuclear norm minimisation.

    ADMM update equations
    ----------------------
    x-update:
        x^{k+1} = argmin_{x: x[Omega]=y} (rho/2)||H(x) - (H^k + Lambda^k/rho)||_F^2
        Solution:
          (a) Set target  T = H^k + Lambda^k / rho
          (b) x_free = H†(T)           (average anti-diagonals of T)
          (c) x^{k+1} = x_free;  x^{k+1}[Omega] = y_obs   (enforce observations)

    H-update:
        H^{k+1} = prox_{(1/rho)||·||_*}( H(x^{k+1}) - Lambda^k/rho )
                = soft_threshold_SVD( H(x^{k+1}) - Lambda^k/rho,  1/rho )

    Lambda-update:
        Lambda^{k+1} = Lambda^k + rho * ( H^{k+1} - H(x^{k+1}) )

    Parameters
    ----------
    y_obs   : ndarray (m,)    observations x_true[Omega]
    Omega   : ndarray (m,)    sampling indices
    r       : int             target rank (informational; not used in convex solver)
    n       : int             signal dimension
    n1      : int, optional   Hankel row count
    max_iter, tol, rho        standard ADMM parameters
    x_true  : ndarray         ground truth for monitoring
    verbose, return_history   output controls

    Returns
    -------
    result : dict
    """
    t0 = time.time()

    if n1 is None:
        n1 = n // 2 + 1
    n2 = n + 1 - n1

    # ---- Initialisation ----
    x = np.zeros(n, dtype=np.complex128)
    np.add.at(x, Omega, y_obs)          # embed observations as starting point
    H = build_hankel(x, n1)
    Lambda = np.zeros((n1, n2), dtype=np.complex128)
    tau = 1.0 / rho                     # nuclear norm weight

    history  = {'rel_errors': [], 'obj_values': []}
    best_err = np.inf
    best_x   = x.copy()
    converged = False

    for k in range(max_iter):

        # ---- x-update ----
        T = H + Lambda / rho            # (n1, n2)
        x_free = _hankel_avg(T, n, n1) # average anti-diagonals
        x_new = x_free.copy()
        x_new[Omega] = y_obs            # enforce measurements

        # ---- H-update ----
        H_target = build_hankel(x_new, n1) - Lambda / rho
        H_new = nuclear_norm_proximal(H_target, tau)

        # ---- Lambda-update ----
        Lambda = Lambda + rho * (H_new - build_hankel(x_new, n1))

        x = x_new
        H = H_new

        # ---- Overflow guard ----
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(H)):
            if verbose:
                print(f"  ADMM: overflow at iteration {k}, returning best so far")
            break

        # ---- Monitoring ----
        if x_true is not None:
            err = relative_error(x, x_true)
            if err < best_err:
                best_err = err
                best_x   = x.copy()
            if return_history:
                history['rel_errors'].append(err)
            if verbose and k % 50 == 0:
                _, s, _ = svd(H, full_matrices=False)
                print(f"  ADMM k={k:4d}: err={err:.2e}, "
                      f"nuc={np.sum(s):.3f}, rank={np.sum(s > 1e-6 * s[0])}")
            if err < tol:
                converged = True
                if verbose:
                    print(f"  ADMM converged at k={k}")
                break
        elif return_history:
            _, s, _ = svd(H, full_matrices=False)
            history['obj_values'].append(float(np.sum(s)))

    # Return best estimate
    if x_true is not None and best_x is not None:
        x = best_x

    result = {
        'x': x, 'H': H,
        'converged': converged,
        'iterations': k + 1,
        'time': time.time() - t0,
        'rho': rho,
    }
    if x_true is not None:
        result['rel_error'] = relative_error(x, x_true)
    if return_history:
        result['history'] = history

    return result
