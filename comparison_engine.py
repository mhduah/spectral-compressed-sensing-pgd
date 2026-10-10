"""
Shared engine for the two comparison experiments run over this thesis's
n=127 signal length:

    experiment_a_reproduction.py    -- Cai-Wang-Wei (2018) Figure 1 style
                                        reproduction: PGD, FIHT, ANM, ADMM
                                        (as the EMaC-equivalent solver),
                                        escalating rank on a shared
                                        comparison schedule (all algorithms
                                        advance together; escalation stops
                                        only once every algorithm has
                                        failed, keeping every row paired).
    experiment_b_shgd_extension.py  -- PGD vs SHGD vs FIHT extension on a
                                        fixed (r, p) grid.

Everything that both scripts need -- seeding, per-trial execution,
robust aggregation, checkpoint save/load, and the algorithm registry --
lives here exactly once, so the two experiments cannot silently drift
out of sync (same seeding convention, same success criterion, same
non-oracle stopping philosophy, same JSON schema).

Reproducibility guarantees enforced here (each one independently
verified before being relied on):

  1. `base_seed` is stored in the checkpoint and verified on resume.
  2. `r_values` is stored in the checkpoint and verified on resume
     (fixed-grid mode).
  3. `solver_params` is a COMPLETE record -- every parameter that can
     affect a solver's numerical trajectory (including max_backtracking,
     min_iter, patience/consecutive-ok-count, not just max_iter and the
     headline tolerances) is stored verbatim and passed explicitly to
     the solver, so a later edit to a function's own default cannot
     silently change the meaning of an existing results file.
  4. `protocol_params` records the shared experimental protocol itself
     (success tolerance, sampling mode, rounding convention for m=p*n,
     damping, numpy/scipy versions) so the file is self-describing.
  5. SHGD's Takagi/ARPACK step is seeded per-trial (via `rng_seed`,
     derived from the same integer trial seed used for signal/sampling
     generation), because `scipy.sparse.linalg.svds` was confirmed to
     return non-bit-for-bit-identical output across separate process
     invocations on an identical input matrix when no `rng` is supplied.
  6. Trial-level records (not just cell aggregates) are written to a
     companion JSONL file next to the checkpoint, preserving the paired
     structure needed for paired statistics (McNemar tests, paired
     runtime comparisons, per-instance success/failure identification).
  7. Per-cell statistics distinguish successful-trial timing/iteration
     counts from all-trial statistics, since mixing converged and
     max-iter/diverged runs into one average can make a robust algorithm
     look faster or slower than it is for the wrong reason.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import scipy

from utils import generate_signal
from pgd_algorithm import pgd_recovery, PGDStopping
from fiht_baseline import fiht_recovery
from shgd_algorithm import shgd_recovery
from admm_baseline import admm_nuclear_norm
from anm_emac_baselines import anm_recovery

N = 127
SUCCESS_TOL = 1e-3  # paper's literal criterion: relative error <= 1e-3


# ============================================================
# Algorithm registry with explicit, recorded hyperparameters
# ============================================================
#
# Every entry pins down EVERY parameter that affects numerical results,
# including ones previously left to rely on a function's own default
# (PGD/SHGD max_backtracking; ADMM/ANM min_iter and patience). These are
# the values written verbatim into `solver_params` in the checkpoint
# JSON.

DEFAULT_SOLVER_PARAMS = {
    'PGD': {
        'max_iter': 2000,
        'relative_iterate_tol': PGDStopping().relative_iterate,
        'relative_objective_tol': PGDStopping().relative_objective,
        'lambda': 0.25,
        'eta_initial_scale': 1.0,
        'backtracking_factor': 0.5,
        'max_backtracking': 30,
    },
    'FIHT': {
        'max_iter': 500,
        'relative_iterate_tol': 1e-6,
    },
    'SHGD': {
        'max_iter': 2000,
        'relative_iterate_tol': 1e-7,
        'relative_objective_tol': 1e-5,
        'eta_scale': 1.5,
        'backtracking_factor': 0.75,
        'armijo_c': 0.1,
        'max_backtracking': 40,
    },
    'ADMM': {
        'max_iter': 500,
        'rho': 2.0,
        'relative_iterate_tol': 1e-6,
        'primal_residual_tol': 1e-6,
        'min_iter': 10,
        'patience': 5,
    },
    'ANM': {
        'max_iter': 2000,
        'rho': 1.0,
        'relative_iterate_tol': 1e-6,
        'primal_residual_tol': 1e-6,
        'min_iter': 50,
        'patience': 10,
    },
}


def build_registry(solver_params: dict) -> dict:
    """
    Build the {name -> callable(y_obs, Omega, r, n, x_true, trial_seed)
    -> dict} registry from an explicit solver_params dict (normally
    DEFAULT_SOLVER_PARAMS, or DEFAULT_SOLVER_PARAMS with deliberate
    overrides). Every wrapper uses ONLY the values in solver_params --
    nothing is read from a function's own defaults -- so solver_params
    is a complete, faithful record of what was actually run.

    `trial_seed` is accepted by every wrapper for a uniform call
    signature but is only actually used by SHGD, where it seeds the
    Takagi/ARPACK initialisation step for cross-process determinism.
    """
    registry = {}

    if 'PGD' in solver_params:
        sp = solver_params['PGD']
        registry['PGD'] = lambda y_obs, Omega, r, n, x_true, trial_seed, sp=sp: pgd_recovery(
            y_obs, Omega, r, n,
            max_iter=sp['max_iter'],
            lam=sp['lambda'],
            eta_initial_scale=sp['eta_initial_scale'],
            backtracking_factor=sp['backtracking_factor'],
            max_backtracking=sp['max_backtracking'],
            stopping=PGDStopping(
                relative_iterate=sp['relative_iterate_tol'],
                relative_objective=sp['relative_objective_tol'],
            ),
            x_true=x_true,
        )

    if 'FIHT' in solver_params:
        sp = solver_params['FIHT']
        registry['FIHT'] = lambda y_obs, Omega, r, n, x_true, trial_seed, sp=sp: fiht_recovery(
            y_obs, Omega, r, n,
            max_iter=sp['max_iter'],
            relative_iterate_tol=sp['relative_iterate_tol'],
            x_true=x_true,
        )

    if 'SHGD' in solver_params:
        sp = solver_params['SHGD']
        registry['SHGD'] = lambda y_obs, Omega, r, n, x_true, trial_seed, sp=sp: shgd_recovery(
            y_obs, Omega, r, n,
            max_iter=sp['max_iter'],
            relative_iterate_tol=sp['relative_iterate_tol'],
            relative_objective_tol=sp['relative_objective_tol'],
            eta_scale=sp['eta_scale'],
            backtracking_factor=sp['backtracking_factor'],
            armijo_c=sp['armijo_c'],
            max_backtracking=sp['max_backtracking'],
            rng_seed=trial_seed,
            x_true=x_true,
        )

    if 'ADMM' in solver_params:
        sp = solver_params['ADMM']
        registry['ADMM'] = lambda y_obs, Omega, r, n, x_true, trial_seed, sp=sp: admm_nuclear_norm(
            y_obs, Omega, r, n,
            max_iter=sp['max_iter'],
            rho=sp['rho'],
            relative_iterate_tol=sp['relative_iterate_tol'],
            primal_residual_tol=sp['primal_residual_tol'],
            min_iter=sp['min_iter'],
            patience=sp['patience'],
            x_true=x_true,
        )

    if 'ANM' in solver_params:
        sp = solver_params['ANM']
        registry['ANM'] = lambda y_obs, Omega, r, n, x_true, trial_seed, sp=sp: anm_recovery(
            y_obs, Omega, r, n,
            max_iter=sp['max_iter'],
            rho=sp['rho'],
            relative_iterate_tol=sp['relative_iterate_tol'],
            primal_residual_tol=sp['primal_residual_tol'],
            min_iter=sp['min_iter'],
            patience=sp['patience'],
            x_true=x_true,
        )

    return registry


# ============================================================
# Seeding (paired across algorithms; pure integer arithmetic)
# ============================================================

def make_seed(base_seed, regime_code, r, pi, trial):
    """
    Deterministic integer seed. No hash() on strings/tuples (PYTHONHASHSEED
    randomizes that across process invocations), and no dependence on
    which algorithm is being evaluated -- every algorithm therefore sees
    the identical signal and sampling pattern (and, for SHGD, the
    identical Takagi/ARPACK start vector) for a given
    (regime, r, p-index, trial) cell, which is what makes the comparison
    paired rather than an average over algorithm-specific difficulties.
    """
    return (
        base_seed
        + regime_code * 100_000_000
        + r * 1_000_000
        + pi * 10_000
        + trial
    )


# ============================================================
# Protocol metadata (the shared experimental protocol itself)
# ============================================================

def compute_protocol_params(n, p_grid):
    m_values = [int(round(p * n)) for p in p_grid]
    actual_p_grid = [m / n for m in m_values]
    return {
        'success_tol': SUCCESS_TOL,
        'success_condition': 'relative_error <= success_tol',
        'sampling_mode': 'without_replacement',
        'm_rounding': 'round',
        'damped': False,
        'm_values': m_values,
        'actual_p_grid': actual_p_grid,
        'numpy_version': np.__version__,
        'scipy_version': scipy.__version__,
    }


# ============================================================
# Per-trial execution and robust aggregation
# ============================================================

def run_single_trial(
    registry,
    algo,
    x_true,
    Omega,
    r,
    n,
    trial_seed,
    tol=SUCCESS_TOL,
):
    """
    Run one registered algorithm on an already-generated instance.

    Numerical linear-algebra failures are recorded as failed trials
    instead of terminating the entire Monte Carlo experiment.
    """
    if algo not in registry:
        raise ValueError(
            f"Unknown algorithm {algo!r}; registered algorithms are "
            f"{sorted(registry)}"
        )

    y_obs = x_true[Omega]
    trial_start = time.perf_counter()

    try:
        res = registry[algo](
            y_obs,
            Omega,
            r,
            n,
            x_true,
            trial_seed,
        )

    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
    ) as exc:
        elapsed = time.perf_counter() - trial_start

        return (
            False,
            float("inf"),
            elapsed,
            f"numerical_failure:{type(exc).__name__}",
            0,
        )

    err = res.get("rel_error", float("inf"))

    # Literal paper criterion:
    # relative reconstruction error <= 1e-3.
    success = np.isfinite(err) and err <= tol

    return (
        success,
        err,
        res.get("time", time.perf_counter() - trial_start),
        res.get("status", ""),
        res.get("iterations", -1),
    )

def safe_stats(errs, times, iters, successes):
    """
    Aggregate per-trial results robustly, distinguishing successful runs
    from failed/diverged ones so a fast-but-frequently-diverging solver
    cannot look faster than a slower-but-reliable one by dragging down
    the average with quick failures (or vice versa for max-iter runs).
    """
    errs = np.asarray(errs, dtype=float)
    times = np.asarray(times, dtype=float)
    iters = np.asarray(iters, dtype=float)
    successes = np.asarray(successes, dtype=bool)

    finite_mask = np.isfinite(errs)
    non_finite_count = int((~finite_mask).sum())
    finite_errs = errs[finite_mask]
    mean_err = float(np.mean(finite_errs)) if finite_errs.size else float('nan')
    median_err = float(np.median(finite_errs)) if finite_errs.size else float('nan')

    n_trials = len(errs)
    n_success = int(successes.sum())
    failure_rate = 1.0 - (n_success / n_trials) if n_trials else float('nan')

    avg_time_all = float(np.mean(times)) if len(times) else float('nan')
    median_time_all = float(np.median(times)) if len(times) else float('nan')
    avg_time_success = float(np.mean(times[successes])) if n_success else float('nan')
    avg_iter_success = float(np.mean(iters[successes])) if n_success else float('nan')

    return {
        'avg_rel_error': mean_err,
        'median_rel_error': median_err,
        'non_finite_count': non_finite_count,
        'avg_time_all': avg_time_all,
        'median_time_all': median_time_all,
        'avg_time_success': avg_time_success,
        'avg_iterations_success': avg_iter_success,
        'failure_rate': failure_rate,
    }


def run_cell(registry, r, p, n, n_trials, separated, base_seed, algorithms,
             regime_code, pi, trial_log=None):
    """
    Run every algorithm in `algorithms` over the SAME n_trials random
    instances for this (r, p) cell -- signal and sampling pattern (and,
    for SHGD, the ARPACK start-vector seed) are derived once per trial
    and reused across all algorithms, so differences between algorithms
    reflect the algorithms and not differences in problem difficulty.

    If `trial_log` is a list, one dict per (algorithm, trial) is
    appended to it with the full paired, trial-level record (not just
    the cell aggregate), so paired statistics can be computed later.
    """
    stats = {algo: {'successes': [], 'errs': [], 'times': [], 'iters': []}
              for algo in algorithms}
    m = max(int(round(p * n)), 1)
    freq_sep = 1.5 if separated else 0.0

    for trial in range(n_trials):
        seed = make_seed(base_seed, regime_code, r, pi, trial)
        rng = np.random.default_rng(seed)
        x_true, _ = generate_signal(n, r, damped=False,
                                     freq_sep=freq_sep, rng=rng)
        Omega = rng.choice(n, size=m, replace=False)

        for algo in algorithms:
            success, err, t, status, iters = run_single_trial(
                registry, algo, x_true, Omega, r, n, trial_seed=seed)
            stats[algo]['successes'].append(success)
            stats[algo]['errs'].append(err)
            stats[algo]['times'].append(t)
            stats[algo]['iters'].append(iters)
            if trial_log is not None:
                trial_log.append({
                    'regime': 'separated' if separated else 'unseparated',
                    'r': r, 'p': p, 'm': m, 'trial': trial, 'seed': seed,
                    'algorithm': algo, 'success': bool(success),
                    'relative_error': err, 'time': t,
                    'iterations': iters, 'status': status,
                })

    out = {}
    for algo, s in stats.items():
        agg = safe_stats(s['errs'], s['times'], s['iters'], s['successes'])
        agg['success_prob'] = float(np.mean(s['successes']))
        out[algo] = agg
    return out


def write_trial_cell(log_dir, regime, r, pi, records):
    """
    Write one (r, p-index) cell's trial-level records to its own file,
    via a temporary file + atomic rename.

    This replaces an earlier append-only single JSONL file, which was
    NOT crash-idempotent: the previous order was (1) run the cell,
    (2) append its records to the shared log, (3) save the checkpoint.
    A process termination between (2) and (3) left the checkpoint
    unaware the cell had been processed, so resuming re-ran the same
    cell and appended a second, duplicate set of records for it -- this
    was reproduced directly (the log ended up with two rows sharing one
    trial key). Because this function's filename is a deterministic
    function of (regime, r, pi) and the write is atomic, re-running an
    unfinished cell simply overwrites its own file with an identical
    result; no deduplication step is ever needed downstream.
    """
    if not records:
        return
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{regime}_r{r:03d}_p{pi:02d}.jsonl")
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')
    os.replace(tmp, path)


def merge_trial_logs(log_dir, combined_path):
    """
    Convenience helper: concatenate every per-cell trial file in
    `log_dir` into a single combined JSONL file at `combined_path`, in
    filename-sorted (i.e. r-then-p) order. Purely a convenience for
    downstream analysis; the per-cell files in `log_dir` remain the
    authoritative, crash-safe record and are not deleted.
    """
    if not os.path.isdir(log_dir):
        return
    cell_files = sorted(
        f for f in os.listdir(log_dir) if f.endswith('.jsonl')
    )
    os.makedirs(os.path.dirname(combined_path) or '.', exist_ok=True)
    with open(combined_path, 'w') as out:
        for fname in cell_files:
            with open(os.path.join(log_dir, fname)) as f:
                out.write(f.read())


# ============================================================
# Checkpoint save/load with base_seed / r_values / solver_params /
# protocol_params verification
# ============================================================

def save_state(out_path, state):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, out_path)


def load_or_init(out_path, *, n, p_grid, n_trials, separated, algorithms,
                  base_seed, solver_params, mode, r_values=None, note=''):
    """
    Load an existing checkpoint (verifying it matches the requested
    configuration) or initialize a fresh one.

    `mode` is 'fixed' (experiment B: r_values is a pre-specified, static
    grid) or 'escalating' (experiment A: r_values grows over time as
    rank is escalated on a shared schedule across all algorithms until
    every algorithm fails). Both modes store and
    verify base_seed, solver_params, and protocol_params; fixed mode
    additionally stores and verifies r_values (escalating mode's
    r_start/r_step/r_max/patience are verified by
    experiment_a_reproduction.py itself, since they govern how r_values
    grows rather than being a static value to compare).
    """
    unknown = set(algorithms) - set(solver_params)
    if unknown:
        raise ValueError(f"No solver_params entry for: {sorted(unknown)}")

    protocol_params = compute_protocol_params(n, p_grid)

    if os.path.exists(out_path):
        with open(out_path) as f:
            state = json.load(f)
        assert state['mode'] == mode, (
            f"Existing file was built in mode {state['mode']!r}, "
            f"requested mode is {mode!r}."
        )
        assert state['n'] == n and state['p_grid'] == p_grid
        assert state['n_trials'] == n_trials and state['separated'] == separated
        assert set(state['algorithms']) == set(algorithms), (
            "Existing results file was built with a different algorithm "
            "set; use a different --out-dir or delete the file."
        )
        assert state['base_seed'] == base_seed, (
            f"Existing file used base_seed={state['base_seed']}, "
            f"requested base_seed={base_seed}. Resuming with a different "
            "seed would silently mix differently-seeded trials into the "
            "same results file."
        )
        if mode == 'fixed':
            assert state['r_values'] == r_values, (
                "Existing file's r_values does not match the requested "
                "r_values; use a different --out-dir or delete the file."
            )
        assert state['solver_params'] == solver_params, (
            "Existing file's solver_params does not match the requested "
            "solver_params. If you intended to change a solver "
            "hyperparameter, use a different --out-dir: resuming under "
            "the same filename with different hyperparameters would mix "
            "results computed under two different numerical protocols "
            "into what looks like one consistent file."
        )
        # numpy/scipy version differences are recorded but not fatal --
        # they don't necessarily change results, so we warn rather than
        # refuse to resume. Every OTHER protocol field, however, defines
        # the experimental protocol itself (success criterion, sampling
        # convention, sample counts, ...) and a silent mismatch there
        # would invalidate the comparison, so those are asserted exactly
        # like solver_params.
        old_pp = state.get('protocol_params', {})
        version_keys = {'numpy_version', 'scipy_version'}
        for key, current_value in protocol_params.items():
            if key in version_keys:
                continue
            assert old_pp.get(key) == current_value, (
                f"Protocol mismatch for {key!r}: checkpoint used "
                f"{old_pp.get(key)!r}, current run requests "
                f"{current_value!r}. Resuming under a changed protocol "
                "field would silently mix results computed under two "
                "different experimental definitions; use a different "
                "--out-dir or delete the file if the change is intended."
            )
        if old_pp.get('numpy_version') != protocol_params['numpy_version'] \
           or old_pp.get('scipy_version') != protocol_params['scipy_version']:
            print(
                "WARNING: resuming under a different numpy/scipy version "
                f"than the checkpoint was started with "
                f"(checkpoint: {old_pp.get('numpy_version')}/"
                f"{old_pp.get('scipy_version')}, current: "
                f"{protocol_params['numpy_version']}/"
                f"{protocol_params['scipy_version']}). Library-version "
                "changes can shift ARPACK/SVD internals; consider a fresh "
                "run if bit-for-bit reproducibility matters here."
            )
        return state

    state = {
        'mode': mode,
        'n': n, 'p_grid': p_grid,
        'r_values': list(r_values) if r_values is not None else [],
        'n_trials': n_trials, 'separated': separated,
        'algorithms': list(algorithms),
        'base_seed': base_seed,
        'solver_params': solver_params,
        'protocol_params': protocol_params,
        'note': note,
        'progress': 0,
    }
    for algo in algorithms:
        state[algo] = {
            'success_prob': [], 'avg_rel_error': [], 'median_rel_error': [],
            'non_finite_count': [], 'avg_time_all': [], 'median_time_all': [],
            'avg_time_success': [], 'avg_iterations_success': [],
            'failure_rate': [],
        }
    return state


# ============================================================
# 80%-success rank-boundary extraction (the actual Figure 1 statistic)
# ============================================================

def extract_rank_boundary(state, algo, threshold=0.8):
    """
    For each sampling ratio p in state['p_grid'], return the largest rank
    r for which state[algo]['success_prob'][r-row][p-col] >= threshold
    (0 if no rank at that p meets the threshold). This is the statistic
    Cai-Wang-Wei actually plot in Figure 1 (an 80%-success
    phase-transition boundary r_max(p)), rather than the full
    success/error grid.
    """
    r_values = state['r_values']
    success = np.asarray(state[algo]['success_prob'], dtype=float)  # (n_r, n_p)
    n_p = len(state['p_grid'])

    boundary = []
    for pi in range(n_p):
        successful_ranks = [
            r_values[ri] for ri in range(len(r_values))
            if ri < success.shape[0] and success[ri, pi] >= threshold
        ]
        boundary.append(max(successful_ranks) if successful_ranks else 0)
    return boundary
