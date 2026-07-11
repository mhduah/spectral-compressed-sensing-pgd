"""
Numerical experiments for Spectral Compressed Sensing thesis.

Implements the extended-study experiments comparing:
- PGD (Projected Gradient Descent) - non-convex factorized method
- ADMM Nuclear Norm Minimization - convex relaxation baseline
- FIHT (Fast Iterative Hard Thresholding)

All experiments in this module default to signal length n=127, matching
the configuration used by Cai, Wang, and Wei (2018) throughout their
Section 3 numerical experiments, so that Hankel matrices are exactly
square (n1=n2=64). This is also the signal length used by the primary
phase-transition reproduction in run_reproduction_experiment.py, so that
every experiment in this thesis is run at a single, consistent signal
length. The sole exception is run_scaling_experiment below, which
necessarily sweeps a range of n values to estimate asymptotic complexity.

Based on:
Cai, J.-F., Wang, T., & Wei, K. (2018). Spectral Compressed Sensing via
Projected Gradient Descent. SIAM Journal on Optimization, 28(3), 2625-2653.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
import time
import json

from utils import (
    generate_signal, sample_with_replacement, sample_without_replacement,
    apply_sampling_operator, relative_error, recovery_success
)
from pgd_algorithm import pgd_recovery
from fiht_baseline import fiht_recovery
from admm_baseline import admm_nuclear_norm


# ============================================================
# Experiment 1: Phase Transition Analysis
# ============================================================

def run_phase_transition(n=127, r_values=None, m_ratios=None,
                         num_trials=10, damped=False,
                         algorithm='admm', max_iter=500, tol=1e-3,
                         rng=None, verbose=True):
    """
    Phase transition experiment: success probability vs. (r, m/n).

    Parameters
    ----------
    n : int
        Signal dimension
    r_values : array-like
        Rank values to test
    m_ratios : array-like
        Sampling ratios m/n to test (in [0, 1])
    num_trials : int
        Number of independent trials per configuration
    damped : bool
        Use damped sinusoids
    algorithm : str
        'admm', 'pgd', or 'fiht'
    max_iter : int
        Maximum iterations
    tol : float
        Success threshold for relative error
    rng : numpy random generator
    verbose : bool

    Returns
    -------
    results : dict
        Contains 'r_values', 'm_ratios', 'success_prob' (2D array),
        'avg_iterations', 'avg_time'
    """
    if rng is None:
        rng = np.random.default_rng()

    if r_values is None:
        r_values = np.arange(1, 11, 1)

    if m_ratios is None:
        m_ratios = np.linspace(0.1, 1.0, 19)

    n_r = len(r_values)
    n_m = len(m_ratios)

    success_prob = np.zeros((n_r, n_m))
    avg_iterations = np.zeros((n_r, n_m))
    avg_time = np.zeros((n_r, n_m))
    avg_errors = np.zeros((n_r, n_m))

    total_runs = n_r * n_m * num_trials

    if verbose:
        print(f"Phase transition: {algorithm}, {total_runs} total runs")
        run_count = 0

    for i, r in enumerate(r_values):
        for j, m_ratio in enumerate(m_ratios):
            m = max(int(m_ratio * n), r + 1)

            successes = 0
            iterations_list = []
            time_list = []
            errors_list = []

            for trial in range(num_trials):
                x_true, _ = generate_signal(n, r, damped=damped, rng=rng)
                Omega = sample_without_replacement(n, m, rng=rng)
                y = x_true[Omega]

                if algorithm == 'admm':
                    result = admm_nuclear_norm(
                        y, Omega, r, n, max_iter=max_iter, tol=tol,
                        rho=2.0, x_true=x_true, verbose=False
                    )
                elif algorithm == 'pgd':
                    result = pgd_recovery(
                        y, Omega, r, n, max_iter=max_iter, tol=tol,
                        x_true=x_true, verbose=False
                    )
                elif algorithm == 'fiht':
                    result = fiht_recovery(
                        y, Omega, r, n, max_iter=max_iter, tol=tol,
                        x_true=x_true, verbose=False
                    )
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")

                rel_err = result['rel_error']
                if rel_err < tol:
                    successes += 1

                iterations_list.append(result['iterations'])
                time_list.append(result['time'])
                errors_list.append(rel_err)

                if verbose:
                    run_count += 1
                    if run_count % 100 == 0:
                        print(f"  Progress: {run_count}/{total_runs}")

            success_prob[i, j] = successes / num_trials
            avg_iterations[i, j] = np.mean(iterations_list)
            avg_time[i, j] = np.mean(time_list)
            avg_errors[i, j] = np.mean(errors_list)

    results = {
        'n': n, 'r_values': r_values.tolist(), 'm_ratios': m_ratios.tolist(),
        'num_trials': num_trials, 'damped': damped, 'algorithm': algorithm,
        'success_prob': success_prob.tolist(),
        'avg_iterations': avg_iterations.tolist(),
        'avg_time': avg_time.tolist(),
        'avg_errors': avg_errors.tolist()
    }

    return results


def plot_phase_transition(results, save_path=None):
    """Plot phase transition diagram."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    r_values = np.array(results['r_values'])
    m_ratios = np.array(results['m_ratios'])
    success_prob = np.array(results['success_prob'])
    avg_time = np.array(results['avg_time'])
    avg_errors = np.array(results['avg_errors'])

    R, M = np.meshgrid(r_values, m_ratios, indexing='ij')

    # Plot 1: Success probability heatmap
    ax = axes[0]
    im = ax.pcolormesh(R, M, success_prob, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
    ax.set_xlabel('Rank r')
    ax.set_ylabel('Sampling ratio m/n')
    ax.set_title(f'Success Probability ({results["algorithm"].upper()})')
    plt.colorbar(im, ax=ax)

    # Plot 2: Average time
    ax = axes[1]
    im2 = ax.pcolormesh(R, M, avg_time, cmap='viridis', shading='auto')
    ax.set_xlabel('Rank r')
    ax.set_ylabel('Sampling ratio m/n')
    ax.set_title('Average Time (s)')
    plt.colorbar(im2, ax=ax)

    # Plot 3: Average error
    ax = axes[2]
    im3 = ax.pcolormesh(R, M, np.log10(avg_errors + 1e-10), cmap='coolwarm', shading='auto')
    ax.set_xlabel('Rank r')
    ax.set_ylabel('Sampling ratio m/n')
    ax.set_title('Log Average Error')
    plt.colorbar(im3, ax=ax)

    damped_str = "damped" if results['damped'] else "undamped"
    fig.suptitle(f'Phase Transition: n={results["n"]}, {damped_str}, {results["num_trials"]} trials')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved phase transition plot to {save_path}")

    return fig


# ============================================================
# Experiment 2: Convergence Analysis
# ============================================================

def run_convergence_analysis(n=127, r=5, m_ratio=0.4,
                              num_trials=5, damped=False,
                              max_iter=300, rng=None):
    """
    Analyze convergence behavior for ADMM and PGD.
    """
    if rng is None:
        rng = np.random.default_rng()

    m = int(m_ratio * n)

    convergence_data = {
        'n': n, 'r': r, 'm_ratio': m_ratio, 'damped': damped,
        'admm': [], 'pgd': [], 'fiht': []
    }

    print(f"  Convergence analysis: n={n}, r={r}, m/n={m_ratio}")
    for trial in range(num_trials):
        x_true, _ = generate_signal(n, r, damped=damped, rng=rng)
        Omega = sample_without_replacement(n, m, rng=rng)
        y = x_true[Omega]

        # ADMM
        result_admm = admm_nuclear_norm(
            y, Omega, r, n, max_iter=max_iter, tol=1e-10,
            rho=2.0, x_true=x_true, verbose=False, return_history=True
        )
        convergence_data['admm'].append(result_admm['history']['rel_errors'])

        # PGD
        result_pgd = pgd_recovery(
            y, Omega, r, n, max_iter=max_iter, tol=1e-10,
            x_true=x_true, verbose=False, return_history=True
        )
        convergence_data['pgd'].append(result_pgd['history']['rel_errors'])

        # FIHT
        result_fiht = fiht_recovery(
            y, Omega, r, n, max_iter=max_iter, tol=1e-10,
            x_true=x_true, verbose=False, return_history=True
        )
        convergence_data['fiht'].append(result_fiht['history']['rel_errors'])

    return convergence_data


def plot_convergence(convergence_data, save_path=None):
    """Plot convergence curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    methods = ['admm', 'pgd', 'fiht']
    colors = {'admm': 'blue', 'pgd': 'green', 'fiht': 'red'}
    labels = {'admm': 'ADMM (Convex)', 'pgd': 'PGD (Non-convex)', 'fiht': 'FIHT'}

    for method in methods:
        histories = convergence_data[method]
        # Plot individual trials
        for trial_hist in histories:
            iters = np.arange(len(trial_hist))
            ax.semilogy(iters, trial_hist, color=colors[method], alpha=0.2, linewidth=0.8)

        # Plot mean
        max_len = max(len(h) for h in histories)
        mean_hist = np.zeros(max_len)
        counts = np.zeros(max_len)
        for h in histories:
            for i, v in enumerate(h):
                mean_hist[i] += v
                counts[i] += 1
        mean_hist /= np.maximum(counts, 1)

        iters = np.arange(max_len)
        ax.semilogy(iters, mean_hist, color=colors[method], linewidth=2.5, label=labels[method])

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative Error')
    ax.set_title(f'Convergence: n={convergence_data["n"]}, r={convergence_data["r"]}, '
                 f'm/n={convergence_data["m_ratio"]}')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")

    return fig


# ============================================================
# Experiment 3: Algorithm Comparison (ADMM vs PGD vs FIHT)
# ============================================================

def run_algorithm_comparison(n=127, r_values=None, m_ratios=None,
                              num_trials=10, damped=False,
                              max_iter=500, tol=1e-3, rng=None):
    """
    Direct comparison between ADMM, PGD, and FIHT on identical problem
    instances (same signal, same sampling pattern, for each trial).

    Default signal length n=127 matches the configuration used by
    Cai, Wang, and Wei (2018), Section 3, so that Hankel matrices are
    exactly square (n1=n2=64). See utils.build_preconditioner.
    """
    if rng is None:
        rng = np.random.default_rng()

    if r_values is None:
        r_values = np.arange(1, 7, 1)

    if m_ratios is None:
        m_ratios = np.linspace(0.1, 1.0, 19)

    methods = ['admm', 'pgd', 'fiht']
    results = {
        'n': n, 'r_values': r_values.tolist(), 'm_ratios': m_ratios.tolist(),
        'num_trials': num_trials, 'damped': damped
    }

    for method in methods:
        results[method] = {
            'success_prob': np.zeros((len(r_values), len(m_ratios))),
            'avg_iterations': np.zeros((len(r_values), len(m_ratios))),
            'avg_time': np.zeros((len(r_values), len(m_ratios))),
            'avg_errors': np.zeros((len(r_values), len(m_ratios)))
        }

    for i, r in enumerate(r_values):
        print(f"  r={r}")
        for j, m_ratio in enumerate(m_ratios):
            m = max(int(m_ratio * n), r + 1)

            for method in methods:
                successes = 0
                iters = []
                times = []
                errors = []

                for trial in range(num_trials):
                    x_true, _ = generate_signal(n, r, damped=damped, rng=rng)
                    Omega = sample_without_replacement(n, m, rng=rng)
                    y = x_true[Omega]

                    if method == 'admm':
                        res = admm_nuclear_norm(y, Omega, r, n,
                                                 max_iter=max_iter, tol=tol,
                                                 rho=2.0, x_true=x_true, verbose=False)
                    elif method == 'pgd':
                        res = pgd_recovery(y, Omega, r, n,
                                            max_iter=max_iter, tol=tol,
                                            x_true=x_true, verbose=False)
                    else:
                        res = fiht_recovery(y, Omega, r, n,
                                             max_iter=max_iter, tol=tol,
                                             x_true=x_true, verbose=False)

                    rel_err = res['rel_error']
                    successes += (rel_err < tol)
                    iters.append(res['iterations'])
                    times.append(res['time'])
                    errors.append(rel_err)

                results[method]['success_prob'][i, j] = successes / num_trials
                results[method]['avg_iterations'][i, j] = np.mean(iters)
                results[method]['avg_time'][i, j] = np.mean(times)
                results[method]['avg_errors'][i, j] = np.mean(errors)

    # Convert numpy arrays to lists for JSON
    for method in methods:
        for key in results[method]:
            results[method][key] = results[method][key].tolist()

    return results


def plot_comparison(comp_results, save_path=None):
    """Plot side-by-side comparison of ADMM vs PGD vs FIHT."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    r_values = np.array(comp_results['r_values'])
    m_ratios = np.array(comp_results['m_ratios'])
    R, M = np.meshgrid(r_values, m_ratios, indexing='ij')

    methods = ['admm', 'pgd', 'fiht']
    titles = ['ADMM (Convex)', 'PGD (Non-convex)', 'FIHT']

    # Row 0: Success probability
    for idx, method in enumerate(methods):
        ax = axes[0, idx]
        sp = np.array(comp_results[method]['success_prob'])
        im = ax.pcolormesh(R, M, sp, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
        ax.set_title(f'{titles[idx]}\nSuccess Probability')
        ax.set_xlabel('Rank r')
        ax.set_ylabel('Sampling ratio m/n')
        plt.colorbar(im, ax=ax)

    # Row 1: Average error, Time comparison, Overall summary
    for idx, method in enumerate(methods):
        ax = axes[1, idx]
        err = np.array(comp_results[method]['avg_errors'])
        im = ax.pcolormesh(R, M, np.log10(err + 1e-10), cmap='coolwarm', shading='auto')
        ax.set_title(f'{titles[idx]}\nLog Average Error')
        ax.set_xlabel('Rank r')
        ax.set_ylabel('Sampling ratio m/n')
        plt.colorbar(im, ax=ax)

    damped_str = "damped" if comp_results['damped'] else "undamped"
    fig.suptitle(f'Algorithm Comparison: n={comp_results["n"]}, {damped_str}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    return fig


# ============================================================
# Experiment 4: Noise Robustness
# ============================================================

def run_noise_robustness(n=127, r=5, snr_values=None,
                          num_trials=20, max_iter=500, rng=None):
    """Test robustness to additive Gaussian noise."""
    if rng is None:
        rng = np.random.default_rng()

    if snr_values is None:
        snr_values = np.array([5, 10, 15, 20, 25, 30, 40, 50])

    m = int(0.5 * n)

    results = {
        'n': n, 'r': r, 'm': m,
        'snr_values': snr_values.tolist(),
        'admm_errors': [],
        'pgd_errors': [],
        'fiht_errors': []
    }

    for snr_db in snr_values:
        print(f"  SNR={snr_db}dB")
        admm_errs = []
        pgd_errs = []
        fiht_errs = []

        for _ in range(num_trials):
            x_true, _ = generate_signal(n, r, damped=False, rng=rng)
            Omega = sample_without_replacement(n, m, rng=rng)

            # Add noise
            signal_power = np.mean(np.abs(x_true)**2)
            noise_power = signal_power / (10**(snr_db/10))
            noise = (np.sqrt(noise_power/2) * rng.standard_normal(n) +
                     1j * np.sqrt(noise_power/2) * rng.standard_normal(n))
            x_noisy = x_true + noise
            y = x_noisy[Omega]

            # ADMM
            res_admm = admm_nuclear_norm(y, Omega, r, n,
                                          max_iter=max_iter, tol=1e-10,
                                          rho=2.0, x_true=x_true, verbose=False)
            admm_errs.append(res_admm['rel_error'])

            # PGD
            res_pgd = pgd_recovery(y, Omega, r, n,
                                    max_iter=max_iter, tol=1e-10,
                                    x_true=x_true, verbose=False)
            pgd_errs.append(res_pgd['rel_error'])

            # FIHT
            res_fiht = fiht_recovery(y, Omega, r, n,
                                      max_iter=max_iter, tol=1e-10,
                                      x_true=x_true, verbose=False)
            fiht_errs.append(res_fiht['rel_error'])

        results['admm_errors'].append(admm_errs)
        results['pgd_errors'].append(pgd_errs)
        results['fiht_errors'].append(fiht_errs)

    return results


def plot_noise_robustness(noise_results, save_path=None):
    """Plot noise robustness curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    snr_values = np.array(noise_results['snr_values'])

    for method, color, label in [('admm', 'blue', 'ADMM (Convex)'),
                                  ('pgd', 'green', 'PGD (Non-convex)'),
                                  ('fiht', 'red', 'FIHT')]:
        errors = noise_results[f'{method}_errors']
        mean_err = [np.mean(e) for e in errors]
        std_err = [np.std(e) for e in errors]
        ax.errorbar(snr_values, mean_err, yerr=std_err, marker='o',
                    label=label, capsize=5, linewidth=2, color=color)

    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Relative Error', fontsize=12)
    ax.set_title(f'Noise Robustness: n={noise_results["n"]}, r={noise_results["r"]}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved noise robustness plot to {save_path}")

    return fig


# ============================================================
# Experiment 5: Computational Scaling
# ============================================================

def run_scaling_experiment(n_values=None, r_ratio=0.05, m_ratio=0.4,
                           num_trials=5, max_iter=200, rng=None):
    """Measure computational scaling with signal dimension n."""
    if rng is None:
        rng = np.random.default_rng()

    if n_values is None:
        n_values = [32, 64, 128, 256]

    results = {
        'n_values': n_values,
        'r_ratio': r_ratio,
        'm_ratio': m_ratio,
    }

    for method in ['admm', 'pgd', 'fiht']:
        results[method] = {
            'times': [],
            'time_stds': [],
            'iters': []
        }

    for n in n_values:
        r = max(1, int(r_ratio * n))
        m = int(m_ratio * n)
        print(f"  n={n}, r={r}, m={m}")

        for method in ['admm', 'pgd', 'fiht']:
            times = []
            iters_list = []

            for _ in range(num_trials):
                x_true, _ = generate_signal(n, r, damped=False, rng=rng)
                Omega = sample_without_replacement(n, m, rng=rng)
                y = x_true[Omega]

                if method == 'admm':
                    res = admm_nuclear_norm(y, Omega, r, n,
                                             max_iter=max_iter, tol=1e-3,
                                             rho=2.0, x_true=x_true, verbose=False)
                elif method == 'pgd':
                    res = pgd_recovery(y, Omega, r, n,
                                        max_iter=max_iter, tol=1e-3,
                                        x_true=x_true, verbose=False)
                else:
                    res = fiht_recovery(y, Omega, r, n,
                                         max_iter=max_iter, tol=1e-3,
                                         x_true=x_true, verbose=False)

                times.append(res['time'])
                iters_list.append(res['iterations'])

            results[method]['times'].append(np.mean(times))
            results[method]['time_stds'].append(np.std(times))
            results[method]['iters'].append(np.mean(iters_list))

    return results


def plot_scaling(scaling_results, save_path=None):
    """Plot computational scaling results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_values = np.array(scaling_results['n_values'])

    methods = ['admm', 'pgd', 'fiht']
    colors = {'admm': 'blue', 'pgd': 'green', 'fiht': 'red'}
    labels = {'admm': 'ADMM', 'pgd': 'PGD', 'fiht': 'FIHT'}

    # Time scaling
    ax = axes[0]
    for method in methods:
        times = np.array(scaling_results[method]['times'])
        time_stds = np.array(scaling_results[method]['time_stds'])
        ax.errorbar(n_values, times, yerr=time_stds,
                    marker='o', label=labels[method], capsize=5,
                    linewidth=2, color=colors[method])

    ax.set_xlabel('Signal dimension n', fontsize=12)
    ax.set_ylabel('Runtime (s)', fontsize=12)
    ax.set_title('Computational Scaling', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Iterations
    ax = axes[1]
    for method in methods:
        iters = np.array(scaling_results[method]['iters'])
        ax.plot(n_values, iters, 'o-', label=labels[method],
                color=colors[method], linewidth=2)

    ax.set_xlabel('Signal dimension n', fontsize=12)
    ax.set_ylabel('Average Iterations', fontsize=12)
    ax.set_title('Iteration Count', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved scaling plot to {save_path}")

    return fig


# ============================================================
# Save/Load Results
# ============================================================

def save_results(results, filepath):
    """Save experiment results to JSON."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    results_json = convert(results)

    with open(filepath, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results(filepath):
    """Load experiment results from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)
