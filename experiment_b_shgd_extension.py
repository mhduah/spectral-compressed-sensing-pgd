"""
Experiment B: symmetric-factorisation extension (PGD vs SHGD vs FIHT).

This experiment does NOT claim to reproduce Cai-Wang-Wei (2018) Figure 1.
It asks a different, more specific question:

    Does exploiting the complex symmetry of the square Hankel lifting
    (SHGD, Li et al. 2024) improve recovery efficiency or stability
    relative to asymmetric PGD, on the same paired problem instances?

Design choices deliberately relaxed relative to Experiment A (the
Cai-Wang-Wei reproduction), because they are not required to answer the
question above:

  - a fixed, finite rank grid (not an escalate-until-failure sweep);
  - 20-50 trials per cell rather than exactly 50;
  - only PGD, SHGD, and (optionally) FIHT as a contextual, non-symmetric
    baseline -- ANM/ADMM are omitted here since they are not part of the
    symmetric-vs-asymmetric factorisation question this experiment asks.

What IS still required and enforced (shared with Experiment A via
comparison_engine.py):

  - paired instances (identical signal + sampling pattern, and for
    SHGD an identical ARPACK start-vector seed, across algorithms for a
    given (regime, r, p, trial) cell);
  - non-oracle stopping (x_true is diagnostic-only in every solver);
  - base_seed and r_values recorded and verified on resume;
  - solver_params recorded verbatim and COMPLETE (max_backtracking,
    min_iter, patience, etc. -- not just max_iter and tolerances);
  - trial-level records (not just cell aggregates) written to a
    companion JSONL file, preserving pairing for paired statistics
    (McNemar tests, paired runtime comparisons, per-instance
    success/failure identification);
  - success/failure-separated timing and iteration statistics, so a
    solver that fails quickly doesn't look artificially fast.

Checkpointing is per (r, p) cell, same as before.

Usage
-----
    python experiment_b_shgd_extension.py --regime unseparated --time-budget 240
    (re-run the same command repeatedly until it prints "ALL DONE")
"""

import argparse
import os
import time

from comparison_engine import (
    N, DEFAULT_SOLVER_PARAMS, build_registry, run_cell, load_or_init,
    save_state, write_trial_cell, merge_trial_logs,
)

STAT_KEYS = ('success_prob', 'avg_rel_error', 'median_rel_error',
             'non_finite_count', 'avg_time_all', 'median_time_all',
             'avg_time_success', 'avg_iterations_success', 'failure_rate')


def run_comparison(n, p_grid, r_values, n_trials, separated, algorithms,
                    solver_params, base_seed, out_path, trial_log_dir,
                    time_budget=None, verbose=True):
    registry = build_registry(solver_params)
    state = load_or_init(
        out_path, n=n, p_grid=p_grid, n_trials=n_trials, separated=separated,
        algorithms=algorithms, base_seed=base_seed, solver_params=solver_params,
        mode='fixed', r_values=r_values,
        note='Experiment B (SHGD extension): PGD vs SHGD vs FIHT on a fixed '
             'rank/ratio grid with paired instances. NOT a claim of '
             'reproducing Cai-Wang-Wei Figure 1 -- see experiment_a_'
             'reproduction.py for that.',
    )
    n_r, n_p = len(r_values), len(p_grid)
    total_cells = n_r * n_p
    regime_code = 1 if separated else 0
    t0 = time.time()

    start_cell = state['progress']
    for cell in range(start_cell, total_cells):
        if time_budget is not None and time.time() - t0 > time_budget:
            if verbose:
                print("Time budget reached; stopping mid-run. "
                      "Re-run the same command to resume.")
            save_state(out_path, state)
            return state
        ri, pi = divmod(cell, n_p)
        r, p = r_values[ri], p_grid[pi]

        trial_log = []
        cell_results = run_cell(registry, r, p, n, n_trials, separated,
                                 base_seed, algorithms, regime_code, pi,
                                 trial_log=trial_log)
        regime_name = 'separated' if separated else 'unseparated'
        write_trial_cell(trial_log_dir, regime_name, r, pi, trial_log)

        if pi == 0:
            for algo in algorithms:
                for key in STAT_KEYS:
                    state[algo][key].append([])
        for algo in algorithms:
            c = cell_results[algo]
            for key in STAT_KEYS:
                state[algo][key][ri].append(c[key])
            if verbose:
                flag = f" (non_finite={c['non_finite_count']})" if c['non_finite_count'] else ""
                print(f"  [{'sep' if separated else 'unsep'}] "
                      f"algo={algo:5s} r={r:2d} p={p:.2f}  "
                      f"success={c['success_prob']:.2f}  "
                      f"median_err={c['median_rel_error']:.2e}  "
                      f"avg_iter(success)={c['avg_iterations_success']:.1f}{flag}")
        state['progress'] = cell + 1
        save_state(out_path, state)

    if verbose and state['progress'] >= total_cells:
        print("ALL DONE for this regime.")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regime', choices=['unseparated', 'separated'],
                         required=True)
    parser.add_argument('--r-values', nargs='+', type=int,
                         default=[1, 3, 5, 7, 9, 11])
    parser.add_argument('--p-grid', nargs='+', type=float,
                         default=[round(0.10 + 0.05 * i, 2) for i in range(18)])
    parser.add_argument('--n-trials', type=int, default=30,
                         help="30-50 recommended for this extension; unlike "
                              "Experiment A, exactly 50 is not required.")
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--algorithms', nargs='+',
                         default=['PGD', 'SHGD', 'FIHT'],
                         choices=list(DEFAULT_SOLVER_PARAMS))
    parser.add_argument('--time-budget', type=float, default=None)
    parser.add_argument('--out-dir', default='results')
    args = parser.parse_args()

    separated = args.regime == 'separated'
    out_path = os.path.join(args.out_dir, f'experiment_b_{args.regime}.json')
    trial_log_dir = os.path.join(args.out_dir, f'experiment_b_{args.regime}_trials')

    solver_params = {a: DEFAULT_SOLVER_PARAMS[a] for a in args.algorithms}

    t0 = time.time()
    state = run_comparison(N, args.p_grid, args.r_values, args.n_trials, separated,
                            args.algorithms, solver_params, args.seed, out_path,
                            trial_log_dir, time_budget=args.time_budget)
    print(f"Call finished in {time.time() - t0:.1f}s. Wrote {out_path}")
    print(f"Per-cell trial records written to {trial_log_dir}/")
    n_r, n_p = len(args.r_values), len(args.p_grid)
    if state['progress'] >= n_r * n_p:
        combined_path = os.path.join(args.out_dir, f'experiment_b_{args.regime}_trials.jsonl')
        merge_trial_logs(trial_log_dir, combined_path)
        print(f"Run complete: merged trial records into {combined_path}")


if __name__ == '__main__':
    main()
