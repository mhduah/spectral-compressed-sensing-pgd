"""Dense reference implementation of the actual FIHT iteration.

This follows Algorithm 2 of Cai, Wang & Wei, "Fast and Provable Algorithms
for Spectrally Sparse Signal Reconstruction via Low-Rank Hankel Matrix
Completion".  It includes the tangent-space projection that distinguishes
FIHT from ordinary IHT.
"""
from __future__ import annotations

import time
import numpy as np

from utils import (
    build_hankel,
    build_preconditioner,
    hankel_adj,
    relative_error,
    sampling_adjoint,
    truncated_svd,
)


def _rank_r_matrix(M: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vh = truncated_svd(M, r)
    L = (U * s[None, :]) @ Vh
    return L, U, s, Vh.conj().T


def _hankel_pinv(M: np.ndarray, n: int, n1: int, D: np.ndarray) -> np.ndarray:
    return hankel_adj(M, n, n1) / (D**2)


def _tangent_projection(M: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    UUH_M = U @ (U.conj().T @ M)
    M_VVH = (M @ V) @ V.conj().T
    UUH_M_VVH = U @ ((U.conj().T @ M) @ V) @ V.conj().T
    return UUH_M + M_VVH - UUH_M_VVH


def fiht_recovery(
    y_obs: np.ndarray,
    omega: np.ndarray,
    r: int,
    n: int,
    *,
    n1: int | None = None,
    max_iter: int = 500,
    relative_iterate_tol: float = 1e-6,
    x_true: np.ndarray | None = None,
    return_history: bool = False,
    verbose: bool = False,
) -> dict:
    """Recover a signal with FIHT using the paper's p^{-1} gradient step.

    Ground truth, when supplied, is diagnostic only and is never used to stop
    or select an iterate.
    """
    start = time.perf_counter()
    y_obs = np.asarray(y_obs, dtype=np.complex128)
    omega = np.asarray(omega, dtype=int)
    if n1 is None:
        n1 = n // 2 + 1
    p = len(omega) / n
    if not (0 < p <= 1):
        raise ValueError("This numerical protocol expects 1 <= m <= n")

    D = build_preconditioner(n, n1)
    embedded = sampling_adjoint(y_obs, omega, n)
    L, U, _, V = _rank_r_matrix(build_hankel(embedded, n1) / p, r)
    x = _hankel_pinv(L, n, n1, D)

    # Reference scale used only to detect divergence (blow-up), never to
    # decide convergence or select an iterate based on x_true. In some
    # genuinely infeasible regimes (high rank relative to sample count),
    # FIHT's rank-r truncation + tangent-space update is not guaranteed
    # non-expansive and the iterate can grow without bound; left
    # unchecked this produces overflow warnings and, on borderline runs,
    # a "converged"-looking but astronomically large finite error that
    # would otherwise dominate any average taken over trials.
    initial_scale = max(np.linalg.norm(x), np.linalg.norm(y_obs), 1.0)
    divergence_threshold = max(1e4 * initial_scale, 1e6)

    history = {
        "relative_iterate": [],
        "sample_residual": [],
        "relative_error": [relative_error(x, x_true)] if x_true is not None else [],
    }
    converged = False
    status = "max_iter"
    iteration = 0

    for k in range(max_iter):
        residual = np.zeros(n, dtype=np.complex128)
        np.add.at(residual, omega, y_obs - x[omega])
        lifted_gradient_step = build_hankel(x + residual / p, n1)
        W = _tangent_projection(lifted_gradient_step, U, V)

        # Check before calling LAPACK. The previous guard occurred only after
        # the SVD and therefore could not prevent SVD non-convergence.
        if not np.all(np.isfinite(W)):
            status = "nonfinite_before_svd"
            break

        if np.max(np.abs(W)) > divergence_threshold:
            status = "diverged_before_svd"
            break

        try:
            L_new, U_new, _, V_new = _rank_r_matrix(W, r)
        except (np.linalg.LinAlgError, FloatingPointError):
            status = "svd_nonconvergence"
            break

        x_new = _hankel_pinv(L_new, n, n1, D)
        

        if not np.all(np.isfinite(x_new)):
            status = "nonfinite_iterate"
            break

        # Check magnitude via max(abs(.)) (no squaring) before any norm
        # computation on x_new, since squaring an already-enormous
        # (but technically finite) value is what triggers the overflow
        # warning; this check reliably fires well before that regime.
        if np.max(np.abs(x_new)) > divergence_threshold:
            status = "diverged"
            break

        rel_step = float(
            np.linalg.norm(x_new - x)
            / max(np.linalg.norm(x), np.finfo(float).eps)
        )
        sample_res = float(
            np.linalg.norm(x_new[omega] - y_obs)
            / max(np.linalg.norm(y_obs), np.finfo(float).eps)
        )

        x, L, U, V = x_new, L_new, U_new, V_new
        iteration = k + 1

        if return_history:
            history["relative_iterate"].append(rel_step)
            history["sample_residual"].append(sample_res)
            if x_true is not None:
                history["relative_error"].append(relative_error(x, x_true))

        if verbose and (k % 50 == 0 or k < 5):
            msg = f"FIHT k={k + 1:4d} rel_step={rel_step:.3e} sample_res={sample_res:.3e}"
            if x_true is not None:
                msg += f" rel_err={relative_error(x, x_true):.3e}"
            print(msg)

        if rel_step <= relative_iterate_tol:
            converged = True
            status = "relative_iterate"
            break

    result = {
        "x": x,
        "converged": converged,
        "status": status,
        "iterations": iteration,
        "time": time.perf_counter() - start,
        "step_size": 1.0 / p,
    }
    if x_true is not None:
        result["rel_error"] = relative_error(x, x_true)
    if return_history:
        result["history"] = history
    return result
