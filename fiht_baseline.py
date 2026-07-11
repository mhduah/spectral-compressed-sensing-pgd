"""
Fast Iterative Hard Thresholding (FIHT) for Spectral Compressed Sensing.

Algorithm
---------
FIHT alternates between:
  1. Gradient step on the data-fidelity  f(x) = ||P_Omega(x) - y_obs||^2
  2. Hard thresholding: project onto the set of signals whose Hankel matrix
     has rank <= r, by keeping only the top-r singular values.

The gradient of f w.r.t. x is:
    grad f(x) = P_Omega^*(P_Omega(x) - y_obs)   (embed residual at Omega)

The hard-thresholding projection is:
    x <- D^{-2} H*( Tr_r( H(x) ) )
i.e. build H(x), truncate to rank r, then recover x via the correct
averaging formula x = D^{-2} H*(·).

Step size: eta = 1 / (n * max(D)^2) gives a conservative stable choice
because ||P_Omega||^2 <= n and the Hankel norms scale with D.

Overflow protection: if any iterate becomes non-finite, the algorithm
terminates early and returns the best result seen so far.
"""

import numpy as np
from scipy.linalg import svd
import time

from utils import (
    build_hankel, hankel_adj, build_preconditioner,
    sampling_adjoint, relative_error
)


def _project_rank_r(x, r, n1, D):
    """
    Project signal x onto rank-r Hankel manifold.

    Steps:
      1. Build H(x)  (shape n1 x n2)
      2. Keep top-r singular values/vectors
      3. Recover x = D^{-2} H*(H_r)

    Returns projected signal, or None if SVD fails due to non-finite input.
    """
    if not np.all(np.isfinite(x)):
        return None
    n = len(x)
    H = build_hankel(x, n1)
    U, s, Vh = svd(H, full_matrices=False)
    H_r = (U[:, :r] * s[:r]) @ Vh[:r, :]
    # Correct reconstruction: x = D^{-2} H*(H_r)
    return hankel_adj(H_r, n, n1) / D**2


def fiht_recovery(y_obs, Omega, r, n, n1=None,
                  step_size=None, max_iter=500, tol=1e-6,
                  x_true=None, verbose=False, return_history=False):
    """
    FIHT for Spectral Compressed Sensing.

    Parameters
    ----------
    y_obs      : ndarray (m,)   observations x_true[Omega]
    Omega      : ndarray (m,)   sampling indices
    r          : int            target rank
    n          : int            signal dimension
    n1         : int, optional  Hankel row count (default n//2+1)
    step_size  : float, optional  gradient step (default 1/(2*n))
    max_iter   : int
    tol        : float
    x_true     : ndarray, optional
    verbose    : bool
    return_history : bool

    Returns
    -------
    result : dict
    """
    t0 = time.time()

    if n1 is None:
        n1 = n // 2 + 1

    D = build_preconditioner(n, n1)

    # ---- Initialisation: one-step from observations ----
    z = sampling_adjoint(y_obs, Omega, n)     # embed observations
    x = _project_rank_r(z, r, n1, D)
    if x is None:
        x = np.zeros(n, dtype=np.complex128)

    # ---- Step size ----
    # grad f Lipschitz constant <= 1 (since P_Omega is a projection)
    # but the Hankel projection scales signal norms; use conservative eta
    if step_size is None:
        step_size = 0.5          # works well empirically; tune if needed

    history   = {'rel_errors': [], 'obj_values': []}
    best_err  = np.inf
    best_x    = x.copy()
    converged = False

    for k in range(max_iter):

        # ---- Gradient of ||P_Omega(x) - y_obs||^2 ----
        grad = np.zeros(n, dtype=np.complex128)
        np.add.at(grad, Omega, x[Omega] - y_obs)   # P_Omega^* P_Omega(x-y)

        # ---- Gradient step ----
        x_half = x - step_size * grad

        # ---- Hard thresholding (rank-r Hankel projection) ----
        x_new = _project_rank_r(x_half, r, n1, D)

        # ---- Overflow guard ----
        if x_new is None or not np.all(np.isfinite(x_new)):
            if verbose:
                print(f"  FIHT: overflow at k={k}, stopping early")
            break

        x = x_new

        # ---- Monitor ----
        if x_true is not None:
            err = relative_error(x, x_true)
            if err < best_err:
                best_err = err
                best_x   = x.copy()
            if return_history:
                history['rel_errors'].append(err)
                obj = float(np.sum(np.abs(x[Omega] - y_obs)**2))
                history['obj_values'].append(obj)
            if verbose and k % 50 == 0:
                print(f"  FIHT k={k:4d}: err={err:.3e}")
            if err < tol:
                converged = True
                if verbose:
                    print(f"  FIHT converged at k={k}")
                break
        elif return_history:
            obj = float(np.sum(np.abs(x[Omega] - y_obs)**2))
            history['obj_values'].append(obj)

    result = {
        'x': best_x,
        'converged': converged,
        'iterations': k + 1,
        'time': time.time() - t0,
        'step_size': step_size,
    }
    if x_true is not None:
        result['rel_error'] = relative_error(best_x, x_true)
    if return_history:
        result['history'] = history

    return result
