#!/usr/bin/env python3
"""
Main script to run the extended-study experiments for the thesis
(Chapter 5, Sections 5.3 onward): the phase transition comparisons,
convergence analysis, algorithm comparison, noise robustness, and
computational scaling.

This is separate from run_reproduction_experiment.py, which reproduces
the primary phase-transition experiment of Cai, Wang, Wei (2018),
Section 3.1, under the paper's exact protocol (n=127, 18-point p-grid,
50 trials, four algorithms ANM/EMaC/FIHT/PGD, both frequency regimes).

All experiments below use n=127 by default, matching both the original
paper's configuration and the primary reproduction, so that every
numerical result in this thesis is computed at a single, consistent
signal length. The sole exception is the computational scaling
experiment, which necessarily sweeps n in {32, 64, 128, 256} to estimate
asymptotic complexity; n=128 appears there only as one of four points on
this sweep.

Run this to generate all extended-study results and figures:
    python run_all_experiments.py [--output-dir results] [--quick]

Each experiment saves results (as JSON) and plots (as PNG) to the output
directory.
"""

import os
import sys
import argparse
import time
import numpy as np

from experiments import (
    run_phase_transition, plot_phase_transition,
    run_convergence_analysis, plot_convergence,
    run_algorithm_comparison, plot_comparison,
    run_noise_robustness, plot_noise_robustness,
    run_scaling_experiment, plot_scaling,
    save_results
)


def main():
    parser = argparse.ArgumentParser(
        description='Run all numerical experiments for Spectral CS thesis'
    )
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results and plots')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick version with fewer trials')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.quick:
        print("=" * 60)
        print("QUICK MODE: Reduced parameters for fast testing")
        print("=" * 60)
        num_trials_pt = 3
        num_trials_conv = 2
        num_trials_comp = 3
        num_trials_noise = 5
        num_trials_scale = 2
        r_values_pt = np.arange(1, 6, 1)
        m_ratios_pt = np.linspace(0.2, 1.0, 9)
        n_values_scale = [32, 64, 128]
    else:
        print("=" * 60)
        print("FULL MODE: Complete experiments")
        print("=" * 60)
        num_trials_pt = 10
        num_trials_conv = 5
        num_trials_comp = 10
        num_trials_noise = 20
        num_trials_scale = 5
        r_values_pt = np.arange(1, 11, 1)
        m_ratios_pt = np.linspace(0.1, 1.0, 19)
        n_values_scale = [32, 64, 128, 256]

    total_start = time.time()

    # ============================================================
    # Experiment 1: Phase Transition (ADMM - convex baseline)
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Phase Transition - ADMM (Undamped)")
    print("=" * 60)
    t0 = time.time()
    results = run_phase_transition(
        n=127, r_values=r_values_pt, m_ratios=m_ratios_pt,
        num_trials=num_trials_pt, damped=False,
        algorithm='admm', max_iter=500, tol=1e-3,
        rng=rng, verbose=True
    )
    save_results(results, os.path.join(args.output_dir, 'phase_transition_admm.json'))
    plot_phase_transition(results, os.path.join(args.output_dir, 'phase_transition_admm.png'))
    print(f"Completed in {time.time() - t0:.1f}s")

    # ============================================================
    # Experiment 2: Phase Transition (PGD - non-convex)
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Phase Transition - PGD (Undamped)")
    print("=" * 60)
    t0 = time.time()
    results = run_phase_transition(
        n=127, r_values=r_values_pt[:5], m_ratios=m_ratios_pt,
        num_trials=num_trials_pt, damped=False,
        algorithm='pgd', max_iter=500, tol=1e-3,
        rng=rng, verbose=True
    )
    save_results(results, os.path.join(args.output_dir, 'phase_transition_pgd.json'))
    plot_phase_transition(results, os.path.join(args.output_dir, 'phase_transition_pgd.png'))
    print(f"Completed in {time.time() - t0:.1f}s")

    # ============================================================
    # Experiment 3: Phase Transition (ADMM, Damped)
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Phase Transition - ADMM (Damped)")
    print("=" * 60)
    t0 = time.time()
    results = run_phase_transition(
        n=127, r_values=r_values_pt[:6], m_ratios=m_ratios_pt,
        num_trials=num_trials_pt, damped=True,
        algorithm='admm', max_iter=500, tol=1e-3,
        rng=rng, verbose=True
    )
    save_results(results, os.path.join(args.output_dir, 'phase_transition_admm_damped.json'))
    plot_phase_transition(results, os.path.join(args.output_dir, 'phase_transition_admm_damped.png'))
    print(f"Completed in {time.time() - t0:.1f}s")

    # ============================================================
    # Experiment 4: Convergence Analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Convergence Analysis")
    print("=" * 60)
    t0 = time.time()
    conv_results = run_convergence_analysis(
        n=127, r=5, m_ratio=0.4,
        num_trials=num_trials_conv, damped=False,
        max_iter=300, rng=rng
    )
    save_results(conv_results, os.path.join(args.output_dir, 'convergence.json'))
    plot_convergence(conv_results, os.path.join(args.output_dir, 'convergence.png'))
    print(f"Completed in {time.time() - t0:.1f}s")

    # ============================================================
    # Experiment 5: Algorithm Comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: ADMM vs PGD vs FIHT Comparison")
    print("=" * 60)
    t0 = time.time()
    comp_results = run_algorithm_comparison(
        n=127, r_values=r_values_pt[:6], m_ratios=m_ratios_pt,
        num_trials=num_trials_comp, damped=False,
        max_iter=500, tol=1e-3, rng=rng
    )
    save_results(comp_results, os.path.join(args.output_dir, 'comparison.json'))
    plot_comparison(comp_results, os.path.join(args.output_dir, 'comparison.png'))
    print(f"Completed in {time.time() - t0:.1f}s")

    # ============================================================
    # Experiment 6: Noise Robustness
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Noise Robustness")
    print("=" * 60)
    t0 = time.time()
    noise_results = run_noise_robustness(
        n=127, r=5,
        snr_values=np.array([5, 10, 15, 20, 25, 30, 40, 50]),
        num_trials=num_trials_noise, max_iter=500, rng=rng
    )
    save_results(noise_results, os.path.join(args.output_dir, 'noise_robustness.json'))
    plot_noise_robustness(noise_results, os.path.join(args.output_dir, 'noise_robustness.png'))
    print(f"Completed in {time.time() - t0:.1f}s")

    # ============================================================
    # Experiment 7: Computational Scaling
    # ============================================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Computational Scaling")
    print("=" * 60)
    t0 = time.time()
    scale_results = run_scaling_experiment(
        n_values=n_values_scale,
        r_ratio=0.05, m_ratio=0.4,
        num_trials=num_trials_scale, max_iter=200, rng=rng
    )
    save_results(scale_results, os.path.join(args.output_dir, 'scaling.json'))
    plot_scaling(scale_results, os.path.join(args.output_dir, 'scaling.png'))
    print(f"Completed in {time.time() - t0:.1f}s")

    # ============================================================
    # Summary
    # ============================================================
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  {f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
