"""
Core utilities for Spectral Compressed Sensing numerical simulations.
Implements signal generation, Hankel matrix operators, and sampling utilities.

Based on:
Cai, J.-F., Wang, T., & Wei, K. (2018). Spectral Compressed Sensing via
Projected Gradient Descent. SIAM Journal on Optimization, 28(3), 2625-2653.

FIXES applied vs original:
  - hankel_adj now computes the TRUE adjoint H* (sum over anti-diagonals),
    not the Moore-Penrose pseudoinverse (average). This restores H*H = D^2.
  - project_onto_incoherence_set uses the correct threshold from the paper:
    tau = sqrt(mu * cs * r * sigma / n), where cs = max(n/n1, n/n2).
"""

import numpy as np
from scipy.linalg import hankel, svd
from numpy.random import default_rng


# ============================================================
# 1. Signal Generation
# ============================================================

def generate_signal(n, r, damped=False, damping_range=(0.01, 0.1),
                    freq_sep=1.5, rng=None):
    """
    Generate a spectrally sparse signal with r complex sinusoids.

    x_t = sum_{k=1}^{r} d_k * exp((2*pi*i*f_k - tau_k) * t)

    Parameters
    ----------
    n : int
        Signal length
    r : int
        Number of sinusoids (model order)
    damped : bool
        If True, generate damped sinusoids
    damping_range : tuple
        Range of damping factors (tau_min, tau_max)
    freq_sep : float
        Minimum frequency separation in units of 1/n (default 1.5/n)
    rng : numpy random generator

    Returns
    -------
    x : ndarray, shape (n,)
        Generated signal
    params : dict
        Ground truth parameters
    """
    if rng is None:
        rng = default_rng()

    # Generate frequencies with minimum separation (rejection sampling)
    min_sep = freq_sep / n
    max_attempts = 10000
    for _ in range(max_attempts):
        freqs = np.sort(rng.random(r))
        diffs = np.diff(freqs)
        wraparound = 1.0 - freqs[-1] + freqs[0]
        all_diffs = np.append(diffs, wraparound)
        if np.all(all_diffs >= min_sep):
            break
    else:
        # Fallback: evenly spaced frequencies with small random perturbation
        freqs = (np.arange(r) / r + rng.random(r) * min_sep * 0.5) % 1.0
        freqs = np.sort(freqs)

    if damped:
        damping = rng.uniform(damping_range[0], damping_range[1], r)
    else:
        damping = np.zeros(r)

    # Amplitudes: |d_k| = 1 + 10^{0.5 * c_k}, phase uniform (matches paper)
    c = rng.random(r)
    magnitudes = 1.0 + 10**(0.5 * c)
    phases = rng.random(r) * 2 * np.pi
    amplitudes = magnitudes * np.exp(1j * phases)

    t = np.arange(n)
    x = np.zeros(n, dtype=np.complex128)
    for k in range(r):
        omega = 2j * np.pi * freqs[k] - damping[k]
        x += amplitudes[k] * np.exp(omega * t)

    params = {
        'frequencies': freqs,
        'damping': damping,
        'amplitudes': amplitudes
    }
    return x, params


def generate_batch_signals(n, r, num_signals=10, damped=False, rng=None):
    """Generate a batch of spectrally sparse signals."""
    if rng is None:
        rng = default_rng()
    signals, params_list = [], []
    for _ in range(num_signals):
        x, p = generate_signal(n, r, damped=damped, rng=rng)
        signals.append(x)
        params_list.append(p)
    return signals, params_list


# ============================================================
# 2. Hankel Matrix Operators
# ============================================================

def build_hankel(x, n1=None):
    """
    Build Hankel matrix from vector x.

    [H(x)]_{i,j} = x_{i+j},  i in [n1], j in [n2],  n1 + n2 = n + 1.

    Parameters
    ----------
    x : ndarray, shape (n,)
    n1 : int, optional  (default n//2 + 1)

    Returns
    -------
    H : ndarray, shape (n1, n2)
    """
    n = len(x)
    if n1 is None:
        n1 = n // 2 + 1
    n2 = n + 1 - n1
    H = hankel(x[:n1], x[n1-1:])
    assert H.shape == (n1, n2), f"Expected ({n1},{n2}), got {H.shape}"
    return H


def hankel_adj(H, n, n1=None):
    """
    TRUE adjoint of the Hankel operator H* : C^{n1 x n2} -> C^n.

    [H*(Z)]_a = sum_{i+j=a} Z_{i,j}   (SUM, not average)

    This satisfies <H(x), Z>_F = <x, H*(Z)>  and  H* H = D^2
    where D is the diagonal preconditioner.

    Parameters
    ----------
    H : ndarray, shape (n1, n2)
    n : int
        Output length (= n1 + n2 - 1)
    n1 : int, optional

    Returns
    -------
    x : ndarray, shape (n,)
    """
    if n1 is None:
        n1 = n // 2 + 1
    n2 = n + 1 - n1

    x = np.zeros(n, dtype=H.dtype)
    for i in range(n1):
        x[i:i+n2] += H[i, :]   # SUM — do NOT divide by counts
    return x


def build_preconditioner(n, n1=None):
    """
    Build diagonal preconditioner D where [D^2]_a = w_a = number of
    anti-diagonal entries in position a of the Hankel matrix.

    This ensures G = H D^{-1} satisfies G* G = I.

    Parameters
    ----------
    n : int
    n1 : int, optional

    Returns
    -------
    D : ndarray, shape (n,)   (the square roots of the anti-diagonal counts)
    """
    if n1 is None:
        n1 = n // 2 + 1
    n2 = n + 1 - n1

    w = np.zeros(n)
    for a in range(n):
        # Anti-diagonal a: pairs (i,j) with i+j=a, 0<=i<n1, 0<=j<n2
        i_min = max(0, a - (n2 - 1))
        i_max = min(n1 - 1, a)
        w[a] = i_max - i_min + 1

    D = np.sqrt(np.maximum(w, 1e-14))
    return D


def precondition(x, D):
    """Apply D:  y = D * x  (element-wise)."""
    return D * x


def unprecondition(y, D):
    """Apply D^{-1}:  x = y / D  (element-wise)."""
    return y / D


# ============================================================
# 3. Normalized Hankel operator G = H D^{-1}  (satisfies G*G = I)
# ============================================================

def G_forward(x, n1, D):
    """
    G(x) = H(D^{-1} x) : C^n -> C^{n1 x n2}

    Parameters
    ----------
    x : ndarray, shape (n,)
    n1 : int
    D : ndarray, shape (n,)   preconditioner

    Returns
    -------
    M : ndarray, shape (n1, n2)
    """
    return build_hankel(x / D, n1)


def G_adjoint(M, n, n1, D):
    """
    G*(M) = D^{-1} H*(M) : C^{n1 x n2} -> C^n

    Parameters
    ----------
    M : ndarray, shape (n1, n2)
    n : int
    n1 : int
    D : ndarray, shape (n,)

    Returns
    -------
    x : ndarray, shape (n,)
    """
    return hankel_adj(M, n, n1) / D


# ============================================================
# 4. Sampling Operators
# ============================================================

def sample_with_replacement(n, m, rng=None):
    """Draw m indices i.i.d. uniform from {0,...,n-1} (duplicates allowed)."""
    if rng is None:
        rng = default_rng()
    return rng.integers(0, n, size=m)


def sample_without_replacement(n, m, rng=None):
    """Draw m distinct indices uniform from {0,...,n-1}."""
    if rng is None:
        rng = default_rng()
    return rng.choice(n, size=m, replace=False)


def apply_sampling_operator(x, Omega):
    """P_Omega(x): return x[Omega]."""
    return x[Omega]


def sampling_adjoint(x_omega, Omega, n):
    """
    Adjoint of sampling operator: place samples into a length-n vector.
    For sampling with replacement, repeated indices are summed.
    """
    x = np.zeros(n, dtype=x_omega.dtype)
    np.add.at(x, Omega, x_omega)
    return x


def create_sampling_mask(n, Omega):
    """Binary mask: True at unique sampled indices."""
    mask = np.zeros(n, dtype=bool)
    mask[np.unique(Omega)] = True
    return mask


# ============================================================
# 5. Projection onto Incoherence Set C
# ============================================================

def project_onto_incoherence_set(Z_U, Z_V, tau):
    """
    Project rows of [Z_U; Z_V] onto the incoherence set

        C = { Z : ||Z||_{2,inf} <= tau }

    where ||Z||_{2,inf} = max_i ||Z_{i,:}||_2.

    Projection is row-wise clipping:
        [P_C(Z)]_{i,:} = Z_{i,:} * min(1, tau / ||Z_{i,:}||_2)

    Parameters
    ----------
    Z_U : ndarray, shape (n1, r)
    Z_V : ndarray, shape (n2, r)
    tau : float
        Row-norm threshold = sqrt(mu * cs * r * sigma / n)

    Returns
    -------
    Z_U_proj, Z_V_proj : projected factors
    """
    def _clip_rows(Z):
        norms = np.linalg.norm(Z, axis=1, keepdims=True)         # (ni, 1)
        scale = np.minimum(1.0, tau / np.maximum(norms, 1e-14))  # clip
        return Z * scale

    return _clip_rows(Z_U), _clip_rows(Z_V)


def compute_projection_threshold(mu, r, sigma1, n, n1, n2):
    """
    Compute the row-norm threshold for the incoherence projection.

        tau = sqrt(mu * cs * r * sigma1 / n)

    where cs = max(n/n1, n/n2)  and  sigma1 = largest singular value of G(y).

    Parameters
    ----------
    mu : float      incoherence parameter
    r : int         rank
    sigma1 : float  largest singular value of G(y) = H(y/D)
    n : int
    n1, n2 : int    Hankel dimensions

    Returns
    -------
    tau : float
    """
    cs = max(n / n1, n / n2)
    tau = np.sqrt(mu * cs * r * sigma1 / n)
    return tau


# ============================================================
# 6. Performance Metrics
# ============================================================

def relative_error(x_est, x_true):
    """||x_est - x_true||_2 / ||x_true||_2"""
    return np.linalg.norm(x_est - x_true) / np.linalg.norm(x_true)


def recovery_success(x_est, x_true, threshold=1e-3):
    """True if relative error < threshold."""
    return relative_error(x_est, x_true) < threshold


def hard_threshold_svd(H, r):
    """Rank-r truncated SVD of H."""
    U, s, Vh = svd(H, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vh[:r, :]


def hankel_relative_error(x_est, x_true, n1=None):
    """Relative error in Hankel Frobenius norm."""
    H_est  = build_hankel(x_est, n1)
    H_true = build_hankel(x_true, n1)
    return np.linalg.norm(H_est - H_true, 'fro') / np.linalg.norm(H_true, 'fro')


def compute_incoherence(x, n1=None):
    """
    Estimate incoherence parameter mu of H(x).

    mu = max( n1 * max_i ||U_{i,:}||^2 / r ,
              n2 * max_j ||V_{j,:}||^2 / r )
    """
    H = build_hankel(x, n1)
    n1_h, n2_h = H.shape
    n = len(x)
    U, s, Vh = svd(H, full_matrices=False)
    r = int(np.sum(s > 1e-10 * s[0]))
    if r == 0:
        return np.inf, 0
    U_r = U[:, :r]
    V_r = Vh[:r, :].T
    mu_U = n1_h * np.max(np.sum(np.abs(U_r)**2, axis=1)) / r
    mu_V = n2_h * np.max(np.sum(np.abs(V_r)**2, axis=1)) / r
    return max(mu_U, mu_V), r


def soft_threshold(x, tau):
    """Soft-thresholding operator (for real arrays)."""
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0)
