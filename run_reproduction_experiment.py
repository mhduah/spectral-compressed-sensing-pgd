"""
Reproduction of the phase transition experiment in
Cai, Wang, and Wei (2018), "Spectral Compressed Sensing via Projected
Gradient Descent", SIAM J. Optimization 28(3), Section 3.1.

This script reproduces the ORIGINAL PAPER'S PROTOCOL as closely as possible:

  - Signal length            n = 127  (so that n1 = n2 = 64)
  - Sampling ratio grid       p in {0.10, 0.15, ..., 0.95}   (18 equispaced values)
  - Trials per configuration  50
  - Rank sweep                r = 1, 2, 3, ... incremented until the algorithm
                              fails ALL 50 trials at every p in the grid
  - Frequency regimes         (a) no separation enforced
                              (b) minimum wraparound separation >= 1.5/n
  - Algorithms compared       ANM, EMaC, FIHT, PGD   (all four, matching the
                              original paper; ADMM is NOT one of the four
                              baselines here -- it is reserved for the
                              extended comparison in experiments.py)
  - Success criterion         80% success threshold curve: for each r, the
                              smallest p at which the empirical success rate
                              first reaches or exceeds 0.8, reported as the
                              phase transition boundary (matching Figure 1
                              of the original paper)

Output
------
Results are saved to results/phase_transition_reproduction_{regime}.json,
containing the full success-probability grid (not just the 80% threshold
curve) so that heatmaps equivalent to this thesis's other phase-transition
figures can also be produced from the same data if desired.

Usage
-----
    python run_reproduction_experiment.py                 # full protocol
    python run_reproduction_experiment.py --quick          # reduced for testing
"""

import argparse
import json
import os
import time

import numpy as np

from utils import generate_signal
from anm_emac_baselines import anm_recovery, emac_recovery
from pgd_algorithm import pgd_recovery
from fiht_baseline import fiht_recovery


# ============================================================
# Configuration matching the original paper's protocol
# ============================================================

N_PAPER = 127                      # signal length used in Cai-Wang-Wei Sec. 3.1
P_GRID = [round(0.10 + 0.05 * i, 2) for i in range(18)]   # 0.10, 0.15, ..., 0.95
N_TRIALS_FULL = 50                 # trials per (r, p) configuration
R_MAX_FULL = 12                    # safety cap on rank sweep (paper sweeps until
                                    # total failure; we cap to bound runtime)
SUCCESS_THRESHOLD_RMSE = 1e-3       # per the convention used throughout this thesis
SUCCESS_PROB_THRESHOLD = 0.8         # the "80% success" criterion of the paper

ALGORITHMS = {
    'ANM':  lambda y, Omega, r, n: anm_recovery(y, Omega, r, n,
                                                 max_iter=2000, rho=1.0,
                                                 x_true=None),
    'EMaC': lambda y, Omega, r, n: emac_recovery(y, Omega, r, n,
                                                  max_iter=500,
                                                  x_true=None),
    'FIHT': lambda y, Omega, r, n: fiht_recovery(y, Omega, r, n,
                                                  max_iter=500,
                                                  x_true=None),
    'PGD':  lambda y, Omega, r, n: pgd_recovery(y, Omega, r, n,
                                                 max_iter=2000,
                                                 x_true=None),
}


# ============================================================
# Single-trial runner
# ============================================================

def run_single_trial(algo_name, x_true, Omega, r, n, tol=SUCCESS_THRESHOLD_RMSE):
    """
    Run one algorithm on one problem instance and return whether it
    succeeded (relative error below tol) together with the relative error
    and wall time.
    """
    y_obs = x_true[Omega]

    if algo_name == 'ANM':
        res = anm_recovery(y_obs, Omega, r, n, max_iter=2000, rho=1.0, x_true=x_true)
    elif algo_name == 'EMaC':
        res = emac_recovery(y_obs, Omega, r, n, max_iter=500, x_true=x_true)
    elif algo_name == 'FIHT':
        res = fiht_recovery(y_obs, Omega, r, n, max_iter=500, x_true=x_true)
    elif algo_name == 'PGD':
        res = pgd_recovery(y_obs, Omega, r, n, max_iter=2000, x_true=x_true)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    err = res['rel_error']
    success = err < tol
    return success, err, res['time']


# ============================================================
# Phase transition sweep, matching the paper's rank-increment rule
# ============================================================

def run_phase_transition_reproduction(n, p_grid, n_trials, r_max,
                                      separated, algorithms,
                                      seed=0, verbose=True):
    """
    Reproduce the phase transition experiment of Cai, Wang, Wei (2018),
    Section 3.1, for the given frequency regime.

    For each rank r = 1, 2, ..., the success probability is measured at
    every p in p_grid, using n_trials independent random trials per
    (r, p) configuration and a freshly generated signal for every trial.
    The rank sweep stops once an algorithm records zero successes across
    the entire p_grid for two consecutive ranks (matching the paper's
    "increment r until total failure" rule, with one rank of slack to
    avoid stopping on an unlucky zero).

    Parameters
    ----------
    n          : int    signal length (127 for the paper reproduction)
    p_grid     : list   sampling ratios to test
    n_trials   : int    trials per (r, p) configuration
    r_max      : int    upper bound on rank sweep (safety cap)
    separated  : bool   whether to enforce the 1.5/n frequency separation
    algorithms : list   subset of {'ANM','EMaC','FIHT','PGD'} to run
    seed       : int    base random seed (offset per (algo, r, p, trial)
                        for full reproducibility)
    verbose    : bool

    Returns
    -------
    results : dict
        {
          'n': n, 'p_grid': p_grid, 'n_trials': n_trials, 'separated': separated,
          'r_values': [...],
          '<algo>': {
              'success_prob': [[...]],   # r_values x p_grid
              'avg_rel_error': [[...]],  # r_values x p_grid
              'avg_time': [[...]],       # r_values x p_grid
          },
          ...
        }
    """
    results = {
        'n': n, 'p_grid': p_grid, 'n_trials': n_trials,
        'separated': separated, 'r_values': []
    }
    for algo in algorithms:
        results[algo] = {'success_prob': [], 'avg_rel_error': [], 'avg_time': []}

    consecutive_total_failure = {algo: 0 for algo in algorithms}
    stopped = {algo: False for algo in algorithms}

    r = 1
    while r <= r_max and not all(stopped.values()):
        results['r_values'].append(r)

        for algo in algorithms:
            succ_row, err_row, time_row = [], [], []

            if stopped[algo]:
                # Pad with zeros once an algorithm has been stopped, so
                # the grids stay rectangular across all algorithms.
                results[algo]['success_prob'].append([0.0] * len(p_grid))
                results[algo]['avg_rel_error'].append([1.0] * len(p_grid))
                results[algo]['avg_time'].append([0.0] * len(p_grid))
                continue

            for pi, p in enumerate(p_grid):
                m = max(int(round(p * n)), 1)
                successes, errs, times = 0, [], []

                for trial in range(n_trials):
                    trial_seed = seed + hash((algo, r, pi, trial)) % (2**31)
                    rng = np.random.default_rng(trial_seed)

                    freq_sep = 1.5 if separated else 0.0
                    x_true, _ = generate_signal(n, r, damped=False,
                                                freq_sep=freq_sep, rng=rng)
                    Omega = rng.choice(n, size=m, replace=False)

                    success, err, t = run_single_trial(algo, x_true, Omega, r, n)
                    successes += int(success)
                    errs.append(err)
                    times.append(t)

                succ_row.append(successes / n_trials)
                err_row.append(float(np.mean(errs)))
                time_row.append(float(np.mean(times)))

            results[algo]['success_prob'].append(succ_row)
            results[algo]['avg_rel_error'].append(err_row)
            results[algo]['avg_time'].append(time_row)

            if max(succ_row) == 0.0:
                consecutive_total_failure[algo] += 1
            else:
                consecutive_total_failure[algo] = 0

            if consecutive_total_failure[algo] >= 2:
                stopped[algo] = True

            if verbose:
                best = max(succ_row)
                print(f"  [{'sep' if separated else 'unsep'}] "
                      f"algo={algo:5s} r={r:2d}  best success={best:.2f}  "
                      f"{'(stopping)' if stopped[algo] else ''}")

        r += 1

    return results


# ============================================================
# Extract the 80% phase transition curve from a full results dict
# ============================================================

def extract_80pct_curve(results, algo):
    """
    For each rank r, find the smallest p in p_grid at which the success
    probability first reaches or exceeds 0.8 (the paper's reported
    threshold). Returns a list of (r, p_threshold) pairs; p_threshold is
    None if the 80% threshold was never reached for that r.
    """
    p_grid = results['p_grid']
    curve = []
    for ri, r in enumerate(results['r_values']):
        succ_row = results[algo]['success_prob'][ri]
        p_thresh = None
        for p, s in zip(p_grid, succ_row):
            if s >= SUCCESS_PROB_THRESHOLD:
                p_thresh = p
                break
        curve.append((r, p_thresh))
    return curve


# ============================================================
# Main entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the phase transition experiment of "
                    "Cai, Wang, Wei (2018), Section 3.1.")
    parser.add_argument('--quick', action='store_true',
                       help="Reduced protocol for fast testing "
                            "(fewer p values, fewer trials, smaller r_max).")
    parser.add_argument('--algorithms', nargs='+',
                       default=['ANM', 'EMaC', 'FIHT', 'PGD'],
                       help="Subset of algorithms to run.")
    parser.add_argument('--regimes', nargs='+', default=['unseparated', 'separated'],
                       choices=['unseparated', 'separated'])
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.quick:
        print("=" * 60)
        print("QUICK MODE: reduced protocol for fast testing")
        print("=" * 60)
        n = 31
        p_grid = [0.2, 0.4, 0.6, 0.8]
        n_trials = 5
        r_max = 4
    else:
        print("=" * 60)
        print("FULL PROTOCOL: n=127, 18-point p-grid, 50 trials per cell")
        print("=" * 60)
        n = N_PAPER
        p_grid = P_GRID
        n_trials = N_TRIALS_FULL
        r_max = R_MAX_FULL

    for regime in args.regimes:
        separated = (regime == 'separated')
        print(f"\n--- Running regime: {regime} ---")
        t0 = time.time()

        results = run_phase_transition_reproduction(
            n=n, p_grid=p_grid, n_trials=n_trials, r_max=r_max,
            separated=separated, algorithms=args.algorithms,
            seed=2026, verbose=True)

        elapsed = time.time() - t0
        results['elapsed_seconds'] = elapsed

        out_path = os.path.join(
            args.output_dir, f"phase_transition_reproduction_{regime}.json")
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nSaved results to {out_path}  (elapsed {elapsed:.1f}s)")

        for algo in args.algorithms:
            curve = extract_80pct_curve(results, algo)
            print(f"\n80% success threshold curve for {algo} ({regime}):")
            for r, p in curve:
                p_str = f"{p:.2f}" if p is not None else "not reached"
                print(f"    r={r:2d}:  p* = {p_str}")


if __name__ == '__main__':
    main()
