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
                      max_iter=500, rho=2.0,
                      relative_iterate_tol=1e-6, primal_residual_tol=1e-6,
                      min_iter=10, patience=5,
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
    max_iter, rho             standard ADMM parameters
    relative_iterate_tol      stop when ||x^{k+1}-x^k|| / ||x^k|| <= this
    primal_residual_tol       stop when ||H^{k+1}-H(x^{k+1})||_F / ||H^{k+1}||_F <= this
    min_iter, patience        both criteria must hold for `patience`
                              consecutive iterations, not before
                              iteration `min_iter`, before accepting
                              convergence. This solver's convergence
                              trajectory was verified to be smooth (no
                              transient false-plateau, unlike ANM below),
                              so small defaults suffice here; kept for
                              interface consistency and defensive safety.
    x_true  : ndarray         ground truth for diagnostics ONLY -- it is
                              never used to decide when to stop or which
                              iterate to return
    verbose, return_history   output controls

    Returns
    -------
    result : dict with keys 'x', 'H', 'converged', 'status', 'iterations',
             'time', and 'rel_error' if x_true is given.
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

    history  = {
        'relative_iterate': [],
        'primal_residual': [],
        'relative_error': [relative_error(x, x_true)] if x_true is not None else [],
    }
    converged = False
    status = "max_iter"
    iteration = 0
    consecutive_ok = 0

    for k in range(max_iter):

        # ---- x-update ----
        T = H + Lambda / rho            # (n1, n2)
        x_free = _hankel_avg(T, n, n1) # average anti-diagonals
        x_new = x_free.copy()
        x_new[Omega] = y_obs            # enforce measurements

        # ---- H-update ----
        H_target = build_hankel(x_new, n1) - Lambda / rho
        H_new = nuclear_norm_proximal(H_target, tau)

        # ---- Overflow guard (before using H_new/x_new further) ----
        if not np.all(np.isfinite(x_new)) or not np.all(np.isfinite(H_new)):
            status = "nonfinite_iterate"
            if verbose:
                print(f"  ADMM: overflow at iteration {k}, stopping")
            break

        # ---- Lambda-update ----
        H_of_x_new = build_hankel(x_new, n1)
        Lambda = Lambda + rho * (H_new - H_of_x_new)

        rel_iterate = float(
            np.linalg.norm(x_new - x)
            / max(np.linalg.norm(x), np.finfo(float).eps)
        )
        primal_res = float(
            np.linalg.norm(H_new - H_of_x_new, 'fro')
            / max(np.linalg.norm(H_new, 'fro'), np.finfo(float).eps)
        )

        x = x_new
        H = H_new
        iteration = k + 1

        if return_history:
            history['relative_iterate'].append(rel_iterate)
            history['primal_residual'].append(primal_res)
            if x_true is not None:
                history['relative_error'].append(relative_error(x, x_true))

        if verbose and (k % 50 == 0 or k < 5):
            msg = f"  ADMM k={k:4d}: rel_step={rel_iterate:.2e} primal_res={primal_res:.2e}"
            if x_true is not None:
                msg += f" rel_err={relative_error(x, x_true):.2e}"
            print(msg)

        if rel_iterate <= relative_iterate_tol and primal_res <= primal_residual_tol:
            consecutive_ok += 1
        else:
            consecutive_ok = 0

        if iteration >= min_iter and consecutive_ok >= patience:
            converged = True
            status = "relative_iterate_and_primal_residual"
            if verbose:
                print(f"  ADMM converged at k={k}")
            break

    result = {
        'x': x, 'H': H,
        'converged': converged,
        'status': status,
        'iterations': iteration,
        'time': time.time() - t0,
        'rho': rho,
    }
    if x_true is not None:
        result['rel_error'] = relative_error(x, x_true)
    if return_history:
        result['history'] = history

    return result
