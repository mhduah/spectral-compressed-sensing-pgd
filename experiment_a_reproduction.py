"""
Experiment A: reproduction of the Cai-Wang-Wei (2018) Section 3.1
phase-transition experiment.

Protocol followed
------------------
  - n = 127 (odd, matching the square-Hankel configuration used
    throughout this thesis);
  - p in {0.10, 0.15, ..., 0.95} (18 sampling ratios);
  - 50 trials per (algorithm, r, p) cell;
  - undamped signals, uniform random phase, magnitude 1 + 10^(0.5c);
  - both frequency regimes (unseparated, and separated by >= 1.5/n);
  - success = relative error <= 1e-3 (paper's literal criterion);
  - rank r escalates from r_start upward in steps of r_step (shared
    across all active algorithms, to keep the comparison paired) and
    stops when EITHER (a) every algorithm has recorded zero success
    across the ENTIRE p_grid for `patience` consecutive rank values in a
    row, i.e. further escalation would be uninformative for every
    algorithm still being compared, OR (b) r exceeds `r_max`, the
    largest rank the n1 x n2 Hankel matrix can structurally support
    (min(n1, n2); 64 for n=127). This reproduces the "increase r until
    total failure" stopping rule described in the paper, interpreted at
    the level of the whole comparison (not per-algorithm dropout), which
    keeps every row of the output grid paired across algorithms, plus a
    hard structural ceiling the paper's rule doesn't need to state
    because it never runs past it in practice.

Algorithm set
-------------
PGD, FIHT, ANM, and ADMM. ADMM here solves the Hankel nuclear-norm
minimisation problem, which is exactly the EMaC (Chen & Chi 2014)
convex program; it is reported under the name ADMM (not EMaC) to be
transparent that it is our own ADMM solver, not the CVX/CVXPY
implementation the original paper used. See anm_emac_baselines.py for
the reasoning behind not double-reporting EMaC and ADMM as if they were
two independent methods.

SHGD is deliberately NOT included in this experiment: it postdates the
paper by six years and belongs to Experiment B (the symmetric-
factorisation extension), not to a reproduction of Figure 1.

Checkpointing
-------------
Checkpointing is at (r, p) CELL granularity, not full-row granularity:
a rank row (all p in p_grid, for the current r) can be interrupted
mid-row by --time-budget and resumed exactly where it left off, via
state['partial_row']. This matters because a single rank row at
n_trials=50 across 4 algorithms and 18 sampling ratios can take
considerably longer than a typical --time-budget value, so row-level
checkpointing alone could not guarantee approximately-on-time
termination.

Known deviations from the original paper (documented, not hidden)
-------------------------------------------------------------------
  - ANM and ADMM are custom ADMM-based solvers, not the CVX defaults
    used in the original paper; phase-transition SHAPE comparisons may
    still be informative, but absolute boundaries and runtimes are not
    directly comparable to the published figures.
  - PGD/FIHT stopping uses the *projected* candidate's objective/iterate
    change, not the paper's unprojected-step objective change; this is a
    minor, documented deviation from the literal stopping rule in the
    paper (see pgd_algorithm.py / fiht_baseline.py docstrings).
  - Rank escalates on a single shared schedule across all algorithms
    (stopping only once ALL are exhausted) rather than the paper's
    per-(algorithm, m) independent escalation; this is more conservative
    (keeps the comparison paired) but can run further than strictly
    necessary for an individual algorithm.
  - Dense NumPy/SciPy operators throughout; no FFT-accelerated
    O(r^2 n + r n log n) implementation, so runtimes are not comparable
    to the paper's MATLAB/PROPACK implementation.

Usage
-----
    python experiment_a_reproduction.py --regime unseparated --time-budget 240
    (re-run the same command repeatedly until it prints "ALL DONE")
"""

import argparse
import os
import time

from comparison_engine import (
    N, DEFAULT_SOLVER_PARAMS, build_registry, run_cell, load_or_init,
    save_state, extract_rank_boundary, write_trial_cell, merge_trial_logs,
)

REPRODUCTION_ALGORITHMS = ['PGD', 'FIHT', 'ANM', 'ADMM']
STAT_KEYS = ('success_prob', 'avg_rel_error', 'median_rel_error',
             'non_finite_count', 'avg_time_all', 'median_time_all',
             'avg_time_success', 'avg_iterations_success', 'failure_rate')


def default_r_max(n):
    """Largest rank the n1 x n2 Hankel matrix can structurally support."""
    n1 = n // 2 + 1
    n2 = n + 1 - n1
    return min(n1, n2)


def run_reproduction(n, p_grid, n_trials, separated, algorithms,
                      solver_params, base_seed, out_path, trial_log_dir,
                      r_start=1, r_step=1, r_max=None, patience=2,
                      time_budget=None, verbose=True):
    if r_max is None:
        r_max = default_r_max(n)

    registry = build_registry(solver_params)
    state = load_or_init(
        out_path, n=n, p_grid=p_grid, n_trials=n_trials, separated=separated,
        algorithms=algorithms, base_seed=base_seed, solver_params=solver_params,
        mode='escalating',
        note='Experiment A (Cai-Wang-Wei reproduction attempt): rank '
             'escalates until every algorithm records zero success across '
             'the full p_grid for `patience` consecutive rank rows, or '
             'until r exceeds r_max. See module docstring for documented '
             'deviations from the original paper (custom ADMM-based '
             'ANM/EMaC solvers instead of CVX, dense linear algebra, '
             'projected-objective stopping test, shared rank schedule).',
    )

    # r_start / r_step / r_max / patience govern how r_values GROWS; they
    # are not themselves part of the static config checked by
    # load_or_init, so they are stored and verified here instead.
    for key, val in (('r_start', r_start), ('r_step', r_step),
                      ('r_max', r_max), ('patience', patience)):
        if key in state:
            assert state[key] == val, (
                f"Existing file used {key}={state[key]}, requested "
                f"{key}={val}. Changing the escalation schedule mid-run "
                "would silently corrupt the rank sequence (e.g. resuming "
                "with a different r_start/r_step after ranks were already "
                "appended is exactly the bug this check prevents); use a "
                "different --out-dir or delete the file if you intend to "
                "change it."
            )
        else:
            state[key] = val

    state.setdefault('consecutive_zero', {algo: 0 for algo in algorithms})
    state.setdefault('done', False)
    state.setdefault('done_reason', None)
    state.setdefault('partial_row', None)

    regime_code = 1 if separated else 0
    t0 = time.time()

    while not state['done']:
        r = state['r_values'][-1] + r_step if state['r_values'] else r_start

        if r > r_max:
            state['done'] = True
            state['done_reason'] = 'r_max'
            save_state(out_path, state)
            if verbose:
                print(f"r={r} exceeds r_max={r_max}; escalation stopped "
                      "(structural ceiling, not algorithmic failure).")
            break

        if state['partial_row'] is not None and state['partial_row']['r'] == r:
            row_cells = state['partial_row']['cells']
        else:
            row_cells = {}
            state['partial_row'] = {'r': r, 'cells': row_cells}

        for pi, p in enumerate(p_grid):
            if str(pi) in row_cells:
                continue  # already computed before an earlier interruption
            if time_budget is not None and time.time() - t0 > time_budget:
                if verbose:
                    print("Time budget reached mid-row; stopping. "
                          "Re-run the same command to resume this exact "
                          f"row (r={r}, next p-index={pi}).")
                save_state(out_path, state)
                return state

            trial_log = []
            cell_results = run_cell(registry, r, p, n, n_trials, separated,
                                     base_seed, algorithms, regime_code, pi,
                                     trial_log=trial_log)
            regime_name = 'separated' if separated else 'unseparated'
            write_trial_cell(trial_log_dir, regime_name, r, pi, trial_log)
            row_cells[str(pi)] = cell_results
            state['partial_row']['cells'] = row_cells
            save_state(out_path, state)  # checkpoint at CELL granularity

            if verbose:
                summary = ", ".join(
                    f"{algo}={cell_results[algo]['success_prob']:.2f}"
                    for algo in algorithms
                )
                print(f"  [{'sep' if separated else 'unsep'}] "
                      f"r={r:3d} p={p:.2f}  success: {summary}")

        # Row complete: finalize into the per-algorithm arrays.
        for algo in algorithms:
            for key in STAT_KEYS:
                state[algo][key].append(
                    [row_cells[str(pi)][algo][key] for pi in range(len(p_grid))])
            row_max_success = max(
                row_cells[str(pi)][algo]['success_prob'] for pi in range(len(p_grid)))
            if row_max_success <= 0.0:
                state['consecutive_zero'][algo] += 1
            else:
                state['consecutive_zero'][algo] = 0

        state['r_values'].append(r)
        state['progress'] = len(state['r_values'])
        state['partial_row'] = None

        if verbose:
            summary = ", ".join(
                f"{algo}(cz={state['consecutive_zero'][algo]})"
                for algo in algorithms
            )
            print(f"  [{'sep' if separated else 'unsep'}] row r={r} complete. {summary}")

        if all(cz >= patience for cz in state['consecutive_zero'].values()):
            state['done'] = True
            state['done_reason'] = 'all_algorithms_failed'
            if verbose:
                print(f"All algorithms failed {patience} consecutive rank "
                      f"rows at r={r}; escalation stopped.")

        save_state(out_path, state)

    if verbose and state['done']:
        print(f"ALL DONE for this regime (reason: {state['done_reason']}).")
    return state


def print_boundaries(state, algorithms, threshold=0.8):
    print(f"\n80%-success rank boundary r_max(p), threshold={threshold}:")
    header = "p_grid".ljust(10) + "".join(a.ljust(8) for a in algorithms)
    print(header)
    boundaries = {a: extract_rank_boundary(state, a, threshold) for a in algorithms}
    for pi, p in enumerate(state['p_grid']):
        row = f"{p:<10.2f}" + "".join(f"{boundaries[a][pi]:<8d}" for a in algorithms)
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regime', choices=['unseparated', 'separated'],
                         required=True)
    parser.add_argument('--p-grid', nargs='+', type=float,
                         default=[round(0.10 + 0.05 * i, 2) for i in range(18)])
    parser.add_argument('--n-trials', type=int, default=50,
                         help="Paper uses 50; kept as the default here.")
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--algorithms', nargs='+',
                         default=REPRODUCTION_ALGORITHMS,
                         choices=list(DEFAULT_SOLVER_PARAMS))
    parser.add_argument('--r-start', type=int, default=1)
    parser.add_argument('--r-step', type=int, default=1)
    parser.add_argument('--r-max', type=int, default=None,
                         help="Default: min(n1, n2) for n=127, i.e. 64 -- "
                              "the largest rank the Hankel matrix can "
                              "structurally support.")
    parser.add_argument('--patience', type=int, default=2,
                         help="Number of consecutive all-zero rank rows "
                              "(across the whole p_grid) required, for "
                              "EVERY algorithm, before escalation stops.")
    parser.add_argument('--time-budget', type=float, default=None)
    parser.add_argument('--out-dir', default='results')
    args = parser.parse_args()

    separated = args.regime == 'separated'
    out_path = os.path.join(args.out_dir, f'experiment_a_{args.regime}.json')
    trial_log_dir = os.path.join(args.out_dir, f'experiment_a_{args.regime}_trials')
    solver_params = {a: DEFAULT_SOLVER_PARAMS[a] for a in args.algorithms}

    t0 = time.time()
    state = run_reproduction(
        N, args.p_grid, args.n_trials, separated, args.algorithms,
        solver_params, args.seed, out_path, trial_log_dir,
        r_start=args.r_start, r_step=args.r_step, r_max=args.r_max,
        patience=args.patience, time_budget=args.time_budget,
    )
    print(f"Call finished in {time.time() - t0:.1f}s. Wrote {out_path}")
    print(f"Per-cell trial records written to {trial_log_dir}/")
    if state.get('done'):
        combined_path = os.path.join(args.out_dir, f'experiment_a_{args.regime}_trials.jsonl')
        merge_trial_logs(trial_log_dir, combined_path)
        print(f"Run complete: merged trial records into {combined_path}")
        print_boundaries(state, args.algorithms)


if __name__ == '__main__':
    main()
