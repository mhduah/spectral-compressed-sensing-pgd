# Spectral Compressed Sensing: Numerical Reproduction

Numerical experiments for an MSc Mathematics thesis reproducing and extending the
Projected Gradient Descent (PGD) algorithm for spectral compressed sensing.

**Reference Paper:**
> Cai, J.-F., Wang, T., & Wei, K. (2018). Spectral Compressed Sensing via Projected
> Gradient Descent. *SIAM Journal on Optimization*, 28(3), 2625-2653.

**Thesis aim:** reproduce, as faithfully as computational resources permit, the phase
transition experiment of Section 3.1 of the paper above, and use this reproduction as the
basis for an extended empirical comparison of PGD against three further recovery methods.

---

## Status

All implemented recovery methods (PGD, ADMM/EMaC, FIHT, and ANM) are
**corrected and verified**. An earlier version of this codebase contained five
implementation errors that caused PGD, ADMM, and FIHT to fail on every test signal; these
are documented in the thesis (Chapter on Numerical Implementation) and summarised in
[Known Issues, Fixed](#known-issues-fixed) below. All experiments in this repository, and
all numbers reported in the thesis, are from the corrected implementation.

**Every experiment in this thesis is run at signal length `n=127`**, matching the exact
configuration used throughout Section 3 of Cai, Wang, and Wei (2018), so that the Hankel
matrix is exactly square (`n1=n2=64`). The sole exception is the computational scaling
experiment, which necessarily sweeps a range of signal dimensions to estimate asymptotic
complexity; `n=128` appears there only as one of four points on that sweep, not as a
competing convention.

---

## Quick Start

1. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv .venv
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```
   No external SDP solver is required; ANM and EMaC are implemented using NumPy/SciPy.

2. **Reproduce the primary phase transition experiment** (matches Cai-Wang-Wei Section 3.1
   exactly: `n=127`, 18-point sampling ratio grid, 50 trials, four algorithms, both
   frequency regimes):
   ```bash
   python run_reproduction_experiment.py --quick   # fast sanity check, ~1-2 min
   python run_reproduction_experiment.py           # full protocol, several hours
   ```
   Results are saved to `results/phase_transition_reproduction_unseparated.json` and
   `results/phase_transition_reproduction_separated.json`.

3. **Run the extended study** (algorithm comparison, convergence, noise robustness,
   computational scaling — all at `n=127`):
   ```bash
   python run_all_experiments.py --quick   # fast sanity check, ~2 min
   python run_all_experiments.py           # full protocol, ~30-60 min
   ```
   Results and plots are saved to the `results/` directory.

4. Generate the phase-transition heatmap figures used in the thesis:
   ```bash
   python make_phase_transition_figures.py \
       --unsep results/phase_transition_reproduction_unseparated.json \
       --sep   results/phase_transition_reproduction_separated.json \
       --outdir images
   ```

---

## File Structure

| File | Description |
|------|-------------|
| `utils.py` | Core utilities: signal generation, Hankel operators, sampling, incoherence projection |
| `pgd_algorithm.py` | Projected Gradient Descent (non-convex factorised method) — corrected |
| `admm_baseline.py` | ADMM solver for Hankel nuclear norm minimisation (convex baseline) — corrected; this is also the EMaC solver |
| `fiht_baseline.py` | Fast Iterative Hard Thresholding — corrected |
| `anm_emac_baselines.py` | Atomic Norm Minimisation (`anm_recovery`) and a re-export of EMaC (`emac_recovery`), used for the primary reproduction |
| `run_reproduction_experiment.py` | Reproduces the primary phase transition experiment under the paper's exact protocol |
| `experiments.py` | Extended-study experiment implementations and plotting functions (all default to `n=127`) |
| `run_all_experiments.py` | Main script to run the full extended-study suite |
| `make_phase_transition_figures.py` | Generates the heatmap figures for the primary reproduction from its JSON output |
| `Spectral_CS_Demo.ipynb` | Interactive Jupyter notebook |

---

## Implemented Algorithms

### 1. Projected Gradient Descent (PGD) — `pgd_algorithm.py`
The primary subject of this thesis. Factorises the Hankel matrix as `H(x) = Z_U Z_V*` and
performs projected gradient descent on the factor matrices via a Wirtinger-calculus
gradient (derived in full in the thesis). Uses a fixed step size calibrated at
initialisation from an inflated singular-value estimate, following the paper's Algorithm 1.

### 2. ADMM Nuclear Norm Minimisation (Convex Baseline) — `admm_baseline.py`
Solves `min ||H||_* s.t. P_Omega(x) = y, H = Hankel(x)` by ADMM. This is the same convex
program as EMaC (Chen and Chi, 2014); `anm_emac_baselines.py` re-exports this solver as
`emac_recovery` for use in the primary reproduction, where it is compared against ANM,
FIHT, and PGD as one of the paper's four original baselines.

### 3. Fast Iterative Hard Thresholding (FIHT) — `fiht_baseline.py`
Alternates a gradient step on the data-fidelity term with hard thresholding onto the
rank-`r` Hankel manifold.

### 4. Atomic Norm Minimisation (ANM) — `anm_emac_baselines.py`
Solves the SDP-equivalent dual/primal Toeplitz moment problem via ADMM (no external SDP
solver required). Used only in the primary phase transition reproduction, matching the
four-algorithm comparison of the original paper.

---

## Experiments

### Primary reproduction (`run_reproduction_experiment.py`)

Reproduces Cai-Wang-Wei (2018), Section 3.1, under the paper's exact protocol:

| Parameter | Value |
|---|---|
| Signal length | `n=127` |
| Sampling ratio grid | 18 equispaced points, `p in {0.10, 0.15, ..., 0.95}` |
| Trials per configuration | 50 |
| Rank sweep | `r = 1, 2, ...` incremented until two consecutive ranks fail entirely |
| Frequency regimes | Unseparated, and separated (minimum wraparound separation `1.5/n`) |
| Algorithms | ANM, EMaC, FIHT, PGD |

### Extended study (`run_all_experiments.py`)

An independent empirical study, run at the same `n=127`, with a coarser grid and fewer
trials, comparing a three-algorithm subset (ADMM, PGD, FIHT):

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | Phase Transition (ADMM, undamped) | Success rate vs. rank `r` and sampling ratio `m/n`, `r=1..10` |
| 2 | Phase Transition (PGD, undamped) | Same for PGD, `r=1..5` -- shows the non-monotone high-rank behaviour discussed below |
| 3 | Phase Transition (ADMM, damped) | Tests robustness to damping factors, `r=1..6` |
| 4 | Convergence Analysis | Relative error vs. iteration count, `n=127, r=5, m/n=0.4`, 5 trials |
| 5 | Algorithm Comparison | ADMM vs. PGD vs. FIHT on identical problem instances, `r=1..6` |
| 6 | Noise Robustness | Recovery error vs. SNR, `n=127, r=5, m/n=0.5`, 8 SNR levels, 20 trials each |
| 7 | Computational Scaling | Runtime and iteration count vs. `n in {32,64,128,256}` |

---

## Usage

### Primary reproduction
```bash
python run_reproduction_experiment.py --quick                            # fast check
python run_reproduction_experiment.py                                     # full protocol
python run_reproduction_experiment.py --regimes unseparated                # one regime only
python run_reproduction_experiment.py --algorithms ANM PGD                  # subset of algorithms
```

### Extended study
```bash
python run_all_experiments.py --quick                    # ~2 minutes
python run_all_experiments.py                             # ~30-60 minutes, full protocol
python run_all_experiments.py --output-dir my_results      # custom output directory
python run_all_experiments.py --seed 123                    # set random seed
```

---

## Key Findings

These are the headline empirical findings from the corrected implementation, reported in
full in the thesis.

1. **The primary reproduction confirms the paper's qualitative ordering.** At
   low-to-moderate rank, ANM is the most sample-efficient of the four algorithms, PGD is
   second, and EMaC and FIHT trail with comparable thresholds -- matching Figure 1 of
   Cai-Wang-Wei (2018). ANM's advantage grows substantially under frequency separation,
   while EMaC's threshold is essentially unaffected by separation, also matching the paper.

2. **PGD shows a reproducible, non-monotone success probability at high rank under
   unseparated frequencies.** Rather than approaching 1.0 as the sampling ratio increases,
   PGD's success probability plateaus around 0.5-0.8 for `r >= 8` and never reaches the
   paper's 80% threshold for `r >= 10` anywhere on the tested grid. This plateau disappears
   almost entirely under frequency separation, pointing to incoherence (rather than sample
   count) as the binding constraint, and is attributed to the fixed step size used
   throughout being miscalibrated for a subset of trials. This finding is corroborated
   independently by the extended study's own phase transition experiment (a completely
   separate simulation, same `n`, different trial count and seed sequence), and reappears
   in the convergence and noise robustness experiments as bimodal, outlier-driven behaviour.

3. **PGD has the most favourable computational scaling.** Empirically `O(n^1.0)` versus
   `O(n^2.4)` for ADMM, consistent with the paper's `O(rn log n)` per-iteration complexity
   claim via FFT-based Hankel operations.

4. **The outlier-driven instability is not confined to non-convex methods.** At high SNR
   (>=30 dB), even the convex ADMM baseline shows a mean RMSE substantially inflated above
   its median, driven by a small subset of trials that fail to converge within the fixed
   iteration budget despite an otherwise easy problem instance. PGD and FIHT show the same
   pattern more often and at somewhat lower SNR. This suggests the instability is better
   understood as a matter of degree, tied partly to a fixed iteration budget, than as a
   strict convex/non-convex distinction.

---

## Known Issues, Fixed

An earlier version of this codebase produced uniformly zero recovery success across every
tested configuration for PGD, ADMM, and FIHT. Debugging traced this to the following root
causes, all now corrected:

1. **Wrong Hankel adjoint.** The adjoint operator `H*` was implemented as an average over
   each anti-diagonal (the Moore-Penrose pseudo-inverse `H_dagger`) rather than the true
   adjoint (a sum), breaking the identity `H*H = D^2` that the gradient and reconstruction
   formulas depend on. `H*` (summing) is used in `pgd_algorithm.py` and
   `fiht_baseline.py`; the averaging operator `H_dagger` is a separate, correctly-named
   helper used only in the ADMM `x`-update, where it is the correct least-squares operator
   for that sub-problem.

2. **Wrong signal reconstruction formula.** The signal was recovered as
   `D^-1 H*(Z_U Z_V*)`; the correct formula is `x = D^-2 H*(Z_U Z_V*)`.

3. **Coordinate-system mismatch in the data fidelity term.** The objective compared
   `G*(Z_U Z_V*)` against the raw observations `y_obs` instead of the correctly
   preconditioned target `D[Omega] * y_obs`.

4. **Undersized incoherence projection threshold.** The threshold `tau` was computed from
   the raw (uninflated) initial singular value estimate, which underestimates the true
   `sigma_1` by a factor of roughly 3, so the true solution lay outside the feasible set.
   Fixed by applying the paper's inflation formula `sigma = sigma_1(L^0) / (1 - eps_0)`,
   `eps_0 = 1/11`.

5. **Missing balancing regulariser gradient.** The term `lambda * g(Z)` was included in the
   objective but never differentiated; its gradient contribution `lambda * Z_U * B` /
   `-lambda * Z_V * B` (with `B = Z_U*Z_U - Z_V*Z_V`) was added.

All five corrections were verified by finite-difference checks on the Wirtinger gradient
(confirming the expected factor-of-2 relationship between the analytic gradient and the
real directional derivative) and by confirming machine-precision recovery
(`rel_error < 1e-6`) on small test problems before scaling up to the full experiment grid.

---

## Mathematical Formulation

### Signal Model
A spectrally sparse signal with `r` components:
```
x(t) = sum_{k=1}^r d_k * exp((2*pi*i*f_k - tau_k)*t),   t = 0, ..., n-1
```

### Hankel Structure
The Hankel matrix `H(x)` has entries:
```
[H(x)]_{i,j} = x_{i+j},   i in [n1], j in [n2],   n1 + n2 = n + 1
```
For `n=127`: `n1 = n2 = 64` (exactly square, matching the paper).

### Key Property
For a spectrally sparse signal with `r` distinct, non-zero-amplitude components:
`rank(H(x)) = r`.

### Normalised Lifting Operator
`G = H D^-1`, where `D` is the diagonal preconditioner with `[D^2]_a` equal to the number of
entries on the `a`-th anti-diagonal of `H`. This satisfies `G*G = I`.

### Problems Solved
- **Non-convex, factorised:** `min F(Z) = f(Z) + lambda * g(Z)` over `Z = [Z_U; Z_V]`, where
  `Z_U Z_V* ~ H(x)` (PGD).
- **Non-convex, thresholding:** gradient step + hard threshold onto rank-`r` Hankel
  manifold (FIHT).
- **Convex relaxation:** `min ||H(x)||_* s.t. x[Omega] = y` (ADMM / EMaC).
- **Convex, atomic norm:** SDP-equivalent dual Toeplitz moment problem (ANM).

---

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- Matplotlib

No external SDP solver (e.g. `cvxpy`) is required; ANM is solved via ADMM on the dual
Toeplitz moment problem using only NumPy/SciPy.

---

## Notes for Thesis Writing

- The primary reproduction (`run_reproduction_experiment.py`) is the only experiment in
  this thesis whose numbers are directly comparable to a specific published figure
  (Figure 1 of Cai-Wang-Wei 2018). The extended study is an independent empirical
  investigation motivated by, but not a literal reproduction of, any specific table in the
  paper -- see the Comparability Summary table in the thesis (Chapter 4) for the precise
  scope of each experiment.
- The model order sensitivity experiment of the paper's Section 3.4 is not reproduced in
  this codebase; see the thesis conclusion for the reasons and for this as a suggested
  direction for future work.
- All JSON result files record their own `n`, trial count, and grid parameters, so any
  regenerated figure or table can be traced back to the exact configuration that produced
  it.
