"""
Symmetric Hankel Projected Gradient Descent (SHGD).

Li, J., Cui, W., and Zhang, X. (2024). Projected Gradient Descent for
Spectral Compressed Sensing via Symmetric Hankel Factorization.
IEEE Transactions on Signal Processing, 72, 1590-1606.
doi:10.1109/TSP.2024.3378004

This module implements SHGD exactly following the paper's equations (9),
(10), (8), and Algorithm 1. Unlike PGD, which factorises the (generally
rectangular) Hankel matrix asymmetrically as H(x) = Z_U Z_V^H, SHGD
exploits the fact that when n is odd, H(x) is a SQUARE, COMPLEX SYMMETRIC
matrix (H(x) = H(x)^T, a direct consequence of the Hankel structure), and
factorises it as H(x) = Z Z^T using a single factor matrix Z and no
balancing regularisation term.

Coordinate conventions (see utils.py for the underlying operators)
--------------------------------------------------------------------
n        : signal length, must be odd (n = 2*ns - 1)
ns       : Hankel matrix dimension (square: ns x ns)
G        : normalised lifting operator, G = H D^{-1}, G*G = I
y_tilde  : preconditioned signal, y_tilde = D * x  (only observed at Omega
           as D[Omega] * y_obs, since only x[Omega] is observed directly)

Factorisation:  G(y_tilde) = H(x) = Z Z^T   (note: transpose, NOT
                                              conjugate transpose)

Objective (paper eq. 9):
    f(Z) = (1/(4p)) * <P_Omega(G*(ZZ^T) - y_tilde), G*(ZZ^T) - y_tilde>
         + (1/4) * ||(I - GG*)(ZZ^T)||_F^2

Wirtinger gradient (paper eq. 10):
    grad_{Z-bar} f(Z) = p^{-1} * (G P_Omega(G*(ZZ^T) - y_tilde)) * Z-bar
                        + (I - GG*)(ZZ^T) * Z-bar

Projection set (paper eq. 8):
    C = { Z : ||Z||_{2,inf} <= 2*sqrt(mu*r*sigma/n) }

Initialisation (paper Algorithm 1): one-step hard thresholding via TAKAGI
factorisation (not standard SVD) of the initial estimate, followed by
projection onto C.
"""

import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
import time

from utils import (
    build_hankel, hankel_adj, build_preconditioner,
    sampling_adjoint, relative_error
)


# ============================================================
# Takagi factorisation (verified against machine precision on
# random complex-symmetric test matrices of both full and low rank;
# see accompanying test_takagi.py)
# ============================================================

def takagi(A):
    """
    Takagi (Autonne-Takagi) factorisation of a complex symmetric matrix:
        A = U @ diag(sigma) @ U.T
    with U unitary (columns orthonormal in the complex sense) and sigma
    real, non-negative, sorted in descending order.

    Computed via the standard 2n x 2n real-symmetric realification: for
    A = Ar + i*Ai (Ar, Ai real, both symmetric since A is complex
    symmetric), form
        S = [[Ar,  Ai],
             [Ai, -Ar]]
    which is real symmetric with eigenvalues in +/- pairs; the Takagi
    values are the non-negative eigenvalues of S, and the Takagi vectors
    u = p + i*q are built from the corresponding eigenvector (p, q) of S,
    with ||u||_2 = ||(p,q)||_2 (no extra normalisation factor needed).

    Note: when A is rank-deficient, the eigenvectors of S spanning the
    zero-eigenvalue (degenerate) subspace need not individually combine
    into unitary complex vectors, since scipy/numpy's eigensolver is not
    aware of the complex pairing structure within a degenerate eigenspace.
    This does not affect correctness here: SHGD only ever uses the
    leading r columns (r = model order), corresponding to the r largest,
    well-separated, generically non-degenerate Takagi values, which are
    verified to be unitary and to reconstruct the truncated matrix exactly
    (see test suite).

    Parameters
    ----------
    A : ndarray, shape (n, n), complex symmetric (A == A.T)

    Returns
    -------
    U : ndarray, shape (n, n), unitary
    sigma : ndarray, shape (n,), real, non-negative, descending
    """
    n = A.shape[0]
    Ar = A.real
    Ai = A.imag
    S = np.block([[Ar, Ai], [Ai, -Ar]])

    eigvals, eigvecs = np.linalg.eigh(S)
    idx = np.argsort(eigvals)[::-1]
    top_idx = idx[:n]

    sigma = eigvals[top_idx]
    vecs = eigvecs[:, top_idx]

    p = vecs[:n, :]
    q = vecs[n:, :]
    U = p + 1j * q

    sigma = np.maximum(sigma, 0.0)
    return U, sigma


def takagi_truncated(A, r, degeneracy_tol=1e-6, rng_seed=0):
    """
    Rank-r truncated Takagi factorization via truncated SVD + phase
    correction, following the authors' MATLAB implementation (which uses
    a truncated SVD followed by a phase correction rather than a full
    2n x 2n real-symmetric eigendecomposition; see SHGD.m). This avoids
    forming and diagonalising the full 2ns x 2ns matrix when only the
    leading r << ns Takagi vectors are needed.

    Reproducibility note: the `svds` (ARPACK) shortcut path below starts
    from a random initial vector. Without an explicit `rng`, scipy draws
    that vector from its own ambient random state, which is NOT fixed by
    seeding numpy's global RNG -- confirmed empirically to produce
    non-bit-for-bit-identical output across separate Python process
    invocations on the identical input matrix, even though the recovered
    error/iteration counts were numerically indistinguishable. Passing an
    explicit `rng_seed` (plumbed through from the caller's trial seed)
    makes the ARPACK start vector, and therefore the output, deterministic
    and reproducible.

    Derivation
    ----------
    Let A (ns x ns) be complex symmetric (A = A^T) and let
        A = W diag(s) V^H
    be its rank-r truncated SVD. Because A is symmetric, for
    non-degenerate singular values the right singular vectors satisfy
    V = conj(W) @ Phi for some diagonal unitary "phase" matrix Phi, i.e.
        Phi = W^T @ V
    is (numerically) diagonal with unit-modulus entries. The Takagi
    vectors are then U = W @ sqrt(Phi), since
        U diag(s) U^T = W sqrt(Phi) diag(s) sqrt(Phi) W^T
                       = W diag(s) Phi W^T = A.

    If singular values are (near-)degenerate, Phi need not be diagonal
    and this shortcut is unsafe; in that case we fall back to the exact
    (more expensive) eigendecomposition-based `takagi()` above, so
    correctness never depends on the generic-case assumption holding.

    Parameters
    ----------
    A : ndarray, shape (ns, ns), complex symmetric
    r : int, number of leading Takagi vectors/values requested
    degeneracy_tol : float, relative tolerance on the off-diagonal mass
        of Phi used to decide whether the shortcut is safe
    rng_seed : int, seed for the ARPACK start vector (see reproducibility
        note above); default 0 preserves the previous call signature but
        callers that care about tying determinism to a specific trial
        should pass the trial's own seed here.

    Returns
    -------
    U_r : ndarray, shape (ns, r)
    sigma_r : ndarray, shape (r,), descending, non-negative
    """
    ns = A.shape[0]
    r_eff = min(r, ns)

    if r_eff < ns - 1:
        # Only the top r_eff singular triplets are computed -- this is the
        # cheap step that replaces the full O(ns^3)-scale 2ns x 2ns eigh.
        W, s, Vh = svds(A, k=r_eff, rng=np.random.default_rng(rng_seed))
        order = np.argsort(s)[::-1]
        W, s, Vh = W[:, order], s[order], Vh[order, :]
    else:
        W, s, Vh = svd(A, full_matrices=False)
        W, s, Vh = W[:, :r_eff], s[:r_eff], Vh[:r_eff, :]

    V = Vh.conj().T
    Phi = W.T @ V
    phase = np.diag(Phi).copy()
    phase_mod = np.abs(phase)
    phase = phase / np.maximum(phase_mod, 1e-14)

    off_diag_mass = np.linalg.norm(Phi - np.diag(np.diag(Phi)), 'fro')
    ref_scale = max(np.linalg.norm(Phi, 'fro'), 1.0)

    if off_diag_mass > degeneracy_tol * ref_scale:
        # Near-degenerate singular values: the diagonal-phase shortcut is
        # not numerically safe here, so fall back to the exact routine.
        U_full, sigma_full = takagi(A)
        return U_full[:, :r_eff], sigma_full[:r_eff]

    # Note the conjugate: substituting V = conj(W) @ Phi into A = W diag(s)
    # V^H and requiring A = U diag(s) U^T for unitary U = W @ Psi gives
    # Psi^2 = conj(Phi), i.e. Psi = sqrt(conj(phase)), not sqrt(phase).
    U_r = W * np.sqrt(np.conj(phase))[np.newaxis, :]
    return U_r, s


# ============================================================
# Core operators (square Hankel case, ns = (n+1)/2)
# ============================================================

def _G_star(L, n, ns, D):
    """G*(L) = D^{-1} H*(L)"""
    return hankel_adj(L, n, ns) / D


def _G_fwd(x, ns, D):
    """G(x) = H(x / D)"""
    return build_hankel(x / D, ns)


def _GG_star(L, n, ns, D):
    """GG*(L) = H(H*(L) / D^2)"""
    return build_hankel(hankel_adj(L, n, ns) / D**2, ns)


def _signal_from_Z(Z, n, ns, D):
    """
    x = D^{-2} H*(Z Z^T), following exactly the same reconstruction logic
    used for PGD (Z Z^T here plays the role of the raw Hankel matrix H(x),
    identically to Z_U Z_V^* in the asymmetric case).
    """
    L = Z @ Z.T
    return hankel_adj(L, n, ns) / D**2


# ============================================================
# Objective and gradient (paper eq. 9 and eq. 10, used verbatim)
# ============================================================

def _objective(Z, y_tilde, Omega, n, ns, D, p):
    L = Z @ Z.T
    GG_star_L = _GG_star(L, n, ns, D)
    f1 = np.linalg.norm(L - GG_star_L, 'fro')**2 / 4.0

    G_star_L = _G_star(L, n, ns, D)
    res = G_star_L[Omega] - y_tilde
    f2 = np.sum(np.abs(res)**2) / (4.0 * p)

    return f1 + f2


def _gradient(Z, y_tilde, Omega, n, ns, D, p):
    """
    Implements paper eq. (10) exactly:
        grad = p^{-1} * G(P_Omega(G*(ZZ^T) - y_tilde)) @ conj(Z)
             + (I - GG*)(ZZ^T) @ conj(Z)
    """
    L = Z @ Z.T

    # Term 1: (I - GG*)(L)
    term1_matrix = L - _GG_star(L, n, ns, D)

    # Term 2: G(P_Omega(G*(L) - y_tilde))
    G_star_L = _G_star(L, n, ns, D)
    res_vec = np.zeros(n, dtype=np.complex128)
    res_vec[Omega] = G_star_L[Omega] - y_tilde
    term2_matrix = _G_fwd(res_vec, ns, D) / p

    R = term1_matrix + term2_matrix
    return R @ Z.conj()


# ============================================================
# Projection onto the incoherence set C (paper eq. 8)
# ============================================================

def _project_C(Z, tau):
    """
    Row-wise clipping onto C = {Z : ||Z||_{2,inf} <= tau}, tau =
    2*sqrt(mu*r*sigma/n) as defined in the paper.
    """
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    scale = np.minimum(1.0, tau / np.maximum(norms, 1e-14))
    return Z * scale


def _compute_tau(mu, r, sigma, n):
    """
    tau = 2*sqrt(mu*r*sigma/n), paper eq. (8), given a plug-in estimate
    of the incoherence parameter mu.

    Kept for reference/testing (e.g. comparing against the theoretical
    eq. 8 expression), but the recovery routine below uses
    `_compute_tau_from_U0`, which matches the threshold actually used in
    the authors' MATLAB implementation and does not import PGD's
    mu/eps0 inflation heuristics.
    """
    return 2.0 * np.sqrt(mu * r * sigma / n)


def _compute_tau_from_U0(U0_r, sigma1):
    """
    Projection threshold matching the authors' MATLAB implementation
    (SHGD.m):
        tau = sqrt( 2 * max_i ||U0_{i,:}||_2^2 * sigma_1(M0) ).

    This is the threshold actually enforced by the reference code and is
    computed directly from the initial Takagi factor U0 and its largest
    Takagi value, sigma1(M0) -- it does not go through an mu/eps0
    inflation step transplanted from the (unrelated) PGD incoherence
    bound, which previously made the projection set unnecessarily large
    or small relative to the authors' implementation.
    """
    row_norm_sq = np.sum(np.abs(U0_r) ** 2, axis=1)
    return np.sqrt(2.0 * np.max(row_norm_sq) * sigma1)


# ============================================================
# Initialisation (paper Algorithm 1: one-step hard thresholding via
# Takagi factorisation, then projection onto C)
# ============================================================

def initialize_shgd(y_obs, Omega, r, n, ns, D, rng_seed=0):
    """
    M^0 = Tr_r( p^{-1} G(P_Omega^*(y_obs)) ) via truncated Takagi
        = U^0 Sigma^0 (U^0)^T
    Z^0_tilde = U^0 (Sigma^0)^{1/2}

    The raw-observation Hankel lifting (no division by D at the vector
    level, only via the G = H D^{-1} operator) mirrors the corrected PGD
    initialisation.

    Unlike the previous version of this function, the incoherence
    parameter mu and the eps0/alpha_mu inflation of sigma (both
    transplanted from the PGD implementation) are no longer computed or
    used here. The projection threshold is instead computed directly
    from U0 and sigma1 by `_compute_tau_from_U0`, matching the authors'
    MATLAB implementation (see `_compute_tau_from_U0` docstring).

    `rng_seed` is forwarded to `takagi_truncated` for deterministic,
    reproducible ARPACK behaviour (see that function's docstring).
    """
    p = len(Omega) / n

    z = sampling_adjoint(y_obs, Omega, n)
    M0 = build_hankel(z, ns) / p   # H(z)/p, consistent with the corrected
                                   # PGD initialisation convention

    U0_r, sigma0_r = takagi_truncated(M0, r, rng_seed=rng_seed)

    sigma1 = sigma0_r[0]
    Z0_tilde = U0_r * np.sqrt(np.maximum(sigma0_r, 0.0))[np.newaxis, :]

    tau = _compute_tau_from_U0(U0_r, sigma1)

    return Z0_tilde, sigma1, tau


# ============================================================
# Main SHGD recovery routine
# ============================================================

def shgd_recovery(y_obs, Omega, r, n, eta_scale=1.5, max_iter=2000,
                  relative_iterate_tol=1e-7, relative_objective_tol=1e-5,
                  backtracking_factor=0.75, armijo_c=0.1,
                  max_backtracking=40, rng_seed=0, x_true=None, verbose=False,
                  return_history=False):
    """
    Symmetric Hankel Projected Gradient Descent.

    Step-size convention (aligned with the authors' MATLAB SHGD.m):
    backtracking starts from eta_0 = eta_scale / sigma1(M0) with
    eta_scale = 1.5 (the authors' alternative fixed-step choice is
    eta_scale = 0.5 with no backtracking, i.e. max_backtracking=0), the
    step is shrunk by backtracking_factor = 3/4 on rejection, and the
    Armijo sufficient-decrease coefficient is armijo_c = 1/10.

    Stopping convention (aligned with pgd_recovery in this package):
    iteration stops when the relative iterate change or the relative
    objective change falls below the given tolerances, or at max_iter.
    ``x_true``, when supplied, is used only to report a diagnostic
    relative error at each iteration (and in the returned result); it is
    never used to decide when to stop or which iterate to return. This
    replaces the previous oracle-based stopping/best-iterate selection,
    which is not a valid recovery procedure since x_true is unknown in
    practice.

    Parameters
    ----------
    y_obs      : ndarray (m,)   observations x_true[Omega]
    Omega      : ndarray (m,)   sampling indices
    r          : int            target rank / model order
    n          : int            signal length; MUST BE ODD (the paper's
                                 square-Hankel construction requires this;
                                 an even-length signal would need
                                 zero-padding by one sample before calling
                                 this function -- not performed
                                 automatically here, to keep this function
                                 a pure drop-in replacement for pgd_recovery
                                 at the odd n=127 used throughout this
                                 thesis).
    eta_scale  : float          initial step size = eta_scale / sigma1(M0)
    max_iter   : int
    relative_iterate_tol   : float  stop when ||x_{k+1}-x_k||/||x_k|| <= this
    relative_objective_tol : float  stop when |F_{k+1}-F_k|/|F_k| <= this
    backtracking_factor : float  step shrink factor on rejection (3/4)
    armijo_c   : float          Armijo sufficient-decrease coefficient (1/10)
    max_backtracking : int      backtracking attempts per iteration
    rng_seed   : int            seed for the ARPACK start vector used in the
                                 Takagi initialisation's svds shortcut path;
                                 pass the caller's trial seed here for
                                 bit-for-bit reproducibility across process
                                 invocations (see takagi_truncated docstring)
    x_true     : ndarray, optional
    verbose    : bool
    return_history : bool

    Returns
    -------
    result : dict, same interface as pgd_recovery for drop-in comparison.
    """
    t0 = time.time()

    assert n % 2 == 1, (
        f"SHGD requires an odd signal length (got n={n}); zero-pad by one "
        "sample before calling shgd_recovery for even-length signals."
    )
    ns = (n + 1) // 2
    p = len(Omega) / n

    D = build_preconditioner(n, ns)
    y_tilde = D[Omega] * y_obs

    Z, sigma1, tau = initialize_shgd(y_obs, Omega, r, n, ns, D, rng_seed=rng_seed)
    Z = _project_C(Z, tau)

    eta = eta_scale / sigma1

    x_current = _signal_from_Z(Z, n, ns, D)
    objective = _objective(Z, y_tilde, Omega, n, ns, D, p)

    history = {
        'objective': [objective],
        'relative_iterate': [],
        'step_size': [],
        'relative_error': [relative_error(x_current, x_true)] if x_true is not None else [],
    }

    status = 'max_iter'
    converged = False
    iteration = 0

    for k in range(max_iter):
        grad = _gradient(Z, y_tilde, Omega, n, ns, D, p)
        grad_norm_sq = np.linalg.norm(grad, 'fro')**2

        f_curr = objective
        eta_k = eta
        Z_try = Z
        accepted = False
        for _ in range(max_backtracking + 1):
            Z_try = _project_C(Z - eta_k * grad, tau)
            f_try = _objective(Z_try, y_tilde, Omega, n, ns, D, p)
            if np.isfinite(f_try) and f_try <= f_curr - armijo_c * eta_k * grad_norm_sq:
                accepted = True
                break
            eta_k *= backtracking_factor

        if not accepted:
            status = 'line_search_failed'
            break

        x_new = _signal_from_Z(Z_try, n, ns, D)
        rel_iterate = float(
            np.linalg.norm(x_new - x_current)
            / max(np.linalg.norm(x_current), np.finfo(float).eps)
        )
        rel_objective = float(
            abs(f_try - f_curr) / max(abs(f_curr), np.finfo(float).eps)
        )

        Z = Z_try
        x_current = x_new
        objective = f_try
        # Allow the step to grow back after an accepted iteration (the
        # next iteration's backtracking loop still enforces the Armijo
        # condition), otherwise a single conservative iteration early on
        # permanently caps eta for the remainder of the run.
        eta = eta_k / backtracking_factor ** 0.25
        iteration = k + 1

        if return_history:
            history['objective'].append(objective)
            history['relative_iterate'].append(rel_iterate)
            history['step_size'].append(eta_k)
            if x_true is not None:
                history['relative_error'].append(relative_error(x_current, x_true))

        if verbose and (k % 100 == 0 or k < 5):
            message = (
                f"SHGD k={k + 1:5d} F={objective:.3e} "
                f"rel_step={rel_iterate:.3e} eta={eta_k:.3e}"
            )
            if x_true is not None:
                message += f" rel_err={relative_error(x_current, x_true):.3e}"
            print(message)

        if rel_iterate <= relative_iterate_tol:
            status = 'relative_iterate'
            converged = True
            break
        if rel_objective <= relative_objective_tol:
            status = 'relative_objective'
            converged = True
            break

    result = {
        'x': x_current,
        'converged': converged,
        'status': status,
        'iterations': iteration,
        'time': time.time() - t0,
        'objective': objective,
        'sigma': sigma1, 'tau': tau, 'eta_last': eta,
    }
    if x_true is not None:
        result['rel_error'] = relative_error(x_current, x_true)
    if return_history:
        result['history'] = history

    return result
