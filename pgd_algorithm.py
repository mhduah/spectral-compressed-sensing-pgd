"""Projected Gradient Descent aligned with Cai, Wang & Wei (2018).

Key alignment points
--------------------
* raw observations are converted to the paper's y = D x coordinates;
* L0 = p^{-1} T_r(H(P_Omega x)), equivalently p^{-1} T_r(G P_Omega y);
* lambda = 1/4;
* row-wise projection onto C;
* backtracking line search for the numerical implementation;
* stopping uses iterate/objective changes, never the unknown ground truth.

The implementation below is dense and intended for reproducible 1-D experiments.
It does not claim the FFT-accelerated O(r^2 n + r n log n) implementation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from utils import (
    build_hankel,
    build_preconditioner,
    compute_projection_threshold,
    G_adjoint,
    G_forward,
    hankel_adj,
    project_onto_incoherence_set,
    relative_error,
    sampling_adjoint,
    truncated_svd,
)


@dataclass(frozen=True)
class PGDStopping:
    relative_iterate: float = 1e-7
    relative_objective: float = 1e-5


def _signal_from_factors(
    Z_U: np.ndarray, Z_V: np.ndarray, n: int, n1: int, D: np.ndarray
) -> np.ndarray:
    return hankel_adj(Z_U @ Z_V.conj().T, n, n1) / (D**2)


def _GG_star(L: np.ndarray, n: int, n1: int, D: np.ndarray) -> np.ndarray:
    return build_hankel(hankel_adj(L, n, n1) / (D**2), n1)


def _objective(
    Z_U: np.ndarray,
    Z_V: np.ndarray,
    y_tilde_obs: np.ndarray,
    omega: np.ndarray,
    n: int,
    n1: int,
    D: np.ndarray,
    p: float,
    lam: float,
) -> float:
    L = Z_U @ Z_V.conj().T
    hankel_residual = L - _GG_star(L, n, n1, D)
    data_residual = G_adjoint(L, n, n1, D)[omega] - y_tilde_obs
    balance = Z_U.conj().T @ Z_U - Z_V.conj().T @ Z_V
    value = (
        np.linalg.norm(hankel_residual, "fro") ** 2
        + np.vdot(data_residual, data_residual).real / p
        + lam * 0.5 * np.linalg.norm(balance, "fro") ** 2
    )
    return float(value)


def _gradient(
    Z_U: np.ndarray,
    Z_V: np.ndarray,
    y_tilde_obs: np.ndarray,
    omega: np.ndarray,
    n: int,
    n1: int,
    D: np.ndarray,
    p: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    L = Z_U @ Z_V.conj().T
    structural = L - _GG_star(L, n, n1, D)

    residual_samples = G_adjoint(L, n, n1, D)[omega] - y_tilde_obs
    residual_vector = np.zeros(n, dtype=np.complex128)
    # This also handles a sampling-with-replacement experiment correctly.
    np.add.at(residual_vector, omega, residual_samples)
    data_term = G_forward(residual_vector, n1, D) / p

    R = structural + data_term
    balance = Z_U.conj().T @ Z_U - Z_V.conj().T @ Z_V
    grad_U = R @ Z_V + lam * Z_U @ balance
    grad_V = R.conj().T @ Z_U - lam * Z_V @ balance
    return grad_U, grad_V


def initialize_pgd(
    y_obs: np.ndarray,
    omega: np.ndarray,
    r: int,
    n: int,
    n1: int,
    D: np.ndarray,
    *,
    eps0: float = 1.0 / 11.0,
    alpha_mu: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """One-step hard-thresholding initialization from raw observations.

    The paper initializes with p^{-1} T_r(G P_Omega(y)), y = D x.  Since D is
    diagonal, G P_Omega(Dx) = H(P_Omega x).  Therefore raw observations must
    be embedded and Hankel-lifted *without* an extra division by D.
    """
    if not (0 < eps0 < 1):
        raise ValueError("eps0 must lie in (0,1)")
    p = len(omega) / n
    if p <= 0:
        raise ValueError("At least one observation is required")

    embedded_raw = sampling_adjoint(y_obs, omega, n)
    lifted = build_hankel(embedded_raw, n1) / p
    U, s, Vh = truncated_svd(lifted, r)
    if len(s) == 0 or s[0] <= np.finfo(float).eps:
        raise RuntimeError("Degenerate spectral initialization")

    s_sqrt = np.sqrt(np.maximum(s, 0.0))
    Z_U = U * s_sqrt[None, :]
    V = Vh.conj().T
    Z_V = V * s_sqrt[None, :]

    n2 = n + 1 - n1
    r_eff = len(s)
    c_s = max(n / n1, n / n2)
    # Estimate the smallest mu consistent with the paper's row-norm bounds.
    mu_U = n * np.max(np.sum(np.abs(U) ** 2, axis=1)) / (c_s * r_eff)
    mu_V = n * np.max(np.sum(np.abs(V) ** 2, axis=1)) / (c_s * r_eff)
    mu = float(alpha_mu * max(mu_U, mu_V, 1.0))
    sigma = float(s[0] / (1.0 - eps0))
    return Z_U, Z_V, sigma, mu


def pgd_recovery(
    y_obs: np.ndarray,
    omega: np.ndarray,
    r: int,
    n: int,
    *,
    n1: int | None = None,
    lam: float = 0.25,
    max_iter: int = 5000,
    stopping: PGDStopping = PGDStopping(),
    eta_initial_scale: float = 1.0,
    backtracking_factor: float = 0.5,
    max_backtracking: int = 30,
    x_true: np.ndarray | None = None,
    return_history: bool = False,
    verbose: bool = False,
) -> dict:
    """Run PGD with a monotone backtracking line search.

    ``x_true`` is used only to report diagnostic error.  It never determines
    stopping, step acceptance, or which iterate is returned.
    """
    start = time.perf_counter()
    y_obs = np.asarray(y_obs, dtype=np.complex128)
    omega = np.asarray(omega, dtype=int)
    if n1 is None:
        n1 = n // 2 + 1
    n2 = n + 1 - n1
    p = len(omega) / n
    if not (0 < p <= 1):
        raise ValueError("This numerical protocol expects 1 <= m <= n")

    D = build_preconditioner(n, n1)
    y_tilde_obs = D[omega] * y_obs

    Z_U, Z_V, sigma, mu = initialize_pgd(y_obs, omega, r, n, n1, D)
    tau = compute_projection_threshold(mu, r, sigma, n, n1, n2)
    Z_U, Z_V = project_onto_incoherence_set(Z_U, Z_V, tau)

    eta = eta_initial_scale / sigma
    objective = _objective(Z_U, Z_V, y_tilde_obs, omega, n, n1, D, p, lam)
    x_current = _signal_from_factors(Z_U, Z_V, n, n1, D)

    history = {
        "objective": [objective],
        "relative_iterate": [],
        "step_size": [],
        "backtracking_steps": [],
        "relative_error": [relative_error(x_current, x_true)] if x_true is not None else [],
    }

    status = "max_iter"
    converged = False
    iteration = 0

    for k in range(max_iter):
        grad_U, grad_V = _gradient(
            Z_U, Z_V, y_tilde_obs, omega, n, n1, D, p, lam
        )

        accepted = False
        trial_eta = eta
        for bt in range(max_backtracking + 1):
            U_trial = Z_U - trial_eta * grad_U
            V_trial = Z_V - trial_eta * grad_V
            U_trial, V_trial = project_onto_incoherence_set(U_trial, V_trial, tau)
            objective_trial = _objective(
                U_trial, V_trial, y_tilde_obs, omega, n, n1, D, p, lam
            )
            if np.isfinite(objective_trial) and objective_trial <= objective:
                accepted = True
                break
            trial_eta *= backtracking_factor

        if not accepted:
            status = "line_search_failed"
            break

        x_new = _signal_from_factors(U_trial, V_trial, n, n1, D)
        rel_iterate = float(
            np.linalg.norm(x_new - x_current)
            / max(np.linalg.norm(x_current), np.finfo(float).eps)
        )
        rel_objective = float(
            abs(objective_trial - objective)
            / max(abs(objective), np.finfo(float).eps)
        )

        Z_U, Z_V = U_trial, V_trial
        x_current = x_new
        objective = objective_trial
        eta = trial_eta
        iteration = k + 1

        if return_history:
            history["objective"].append(objective)
            history["relative_iterate"].append(rel_iterate)
            history["step_size"].append(eta)
            history["backtracking_steps"].append(bt)
            if x_true is not None:
                history["relative_error"].append(relative_error(x_current, x_true))

        if verbose and (k % 100 == 0 or k < 5):
            message = (
                f"PGD k={k + 1:5d} F={objective:.3e} "
                f"rel_step={rel_iterate:.3e} eta={eta:.3e} bt={bt}"
            )
            if x_true is not None:
                message += f" rel_err={relative_error(x_current, x_true):.3e}"
            print(message)

        if rel_iterate <= stopping.relative_iterate:
            status = "relative_iterate"
            converged = True
            break
        if rel_objective <= stopping.relative_objective:
            status = "relative_objective"
            converged = True
            break

        # Permit a modest increase after an accepted step; the next iteration
        # still has to pass the monotonicity test.
        eta /= backtracking_factor ** 0.25

    result = {
        "x": x_current,
        "converged": converged,
        "status": status,
        "iterations": iteration,
        "time": time.perf_counter() - start,
        "objective": objective,
        "sigma": sigma,
        "mu": mu,
        "tau": tau,
        "eta_last": eta,
        "lambda": lam,
    }
    if x_true is not None:
        result["rel_error"] = relative_error(x_current, x_true)
    if return_history:
        result["history"] = history
    return result
