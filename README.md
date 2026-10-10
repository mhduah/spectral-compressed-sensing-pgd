# Spectral Compressed Sensing: Numerical Reproduction and SHGD Extension

This repository contains the numerical code, experiment records, and
figure-generation workflow for an MSc Mathematics thesis on spectral compressed
sensing via projected gradient descent.

**Permanent archived release:** https://doi.org/10.25625/NFBEGZ

The project has two main components:

1. **Experiment A:** a Figure 1-style reproduction of the phase-transition study
   in Cai, Wang, and Wei (2018), comparing PGD, FIHT, ANM, and a Hankel
   nuclear-norm ADMM solver;
2. **Experiment B:** an extension comparing PGD with Symmetric Hankel Gradient
   Descent (SHGD) and FIHT on the same paired problem instances.

The repository also contains a single figure-generation script that rebuilds
Figures 5.1--5.5 of the thesis from the completed trial-level records.

## Main references

> Cai, J.-F., Wang, T., and Wei, K. (2018). *Spectral Compressed Sensing via
> Projected Gradient Descent*. SIAM Journal on Optimization, 28(3), 2625--2653.

> Li, J., Cui, W., and Zhang, X. (2024). *Projected Gradient Descent for Spectral
> Compressed Sensing via Symmetric Hankel Factorization*. IEEE Transactions on
> Signal Processing, 72, 1590--1606.

---

## Archived thesis release

The source code, numerical experiment records, environment specifications, and
figure-generation scripts corresponding to the thesis-submission release are
permanently archived in Göttingen Research Online (GRO.data):

**DOI:** https://doi.org/10.25625/NFBEGZ

The archived dataset corresponds to:

- **Release:** `v1.0-thesis`
- **Git commit:** `4089eb6b40b8c3cad68115926f0c33478c98317d`
- **GRO.data version:** `1.0`

This GitLab repository remains the development repository. The GRO.data DOI
identifies the fixed archival version associated with the thesis-submission
release. The tag `v1.0-thesis` is the immutable reference point for that
archived version; later documentation-only commits on `main` do not alter the
published archive.


## Current status

The current experiment framework includes five recovery methods:

- Projected Gradient Descent (**PGD**);
- Fast Iterative Hard Thresholding (**FIHT**);
- Atomic Norm Minimisation (**ANM**);
- Hankel nuclear-norm minimisation solved by **ADMM**;
- Symmetric Hankel Gradient Descent (**SHGD**).

The ADMM solver and EMaC correspond to the same Hankel nuclear-norm convex
program in this implementation. They are therefore **not reported as two
independent algorithms**. Experiment A labels the method `ADMM` to make clear
that the repository uses its own ADMM implementation rather than the CVX-based
solver used in the original paper.

All principal experiments use signal length

```text
n = 127,
```

which produces a square Hankel matrix of size `64 x 64`. The shared empirical
success criterion is

```text
relative reconstruction error <= 1e-3.
```

The framework now provides:

- deterministic, paired trial generation across algorithms;
- explicit recording of all solver and protocol parameters;
- cell-level checkpointing and exact resumption;
- atomic per-cell trial logs;
- merged trial-level JSONL output after a regime completes;
- separate statistics for successful trials and all attempted trials;
- robust exception handling so that one failed solver call does not terminate
  the entire experiment;
- a conservative SVD fallback for FIHT and shared truncated-SVD operations;
- one script for generating all Chapter 5 figures.

The completed thesis result files under `results/` are retained in version
control to provide an audit trail. Temporary smoke-test, scratch, and other
newly generated outputs should be written to separate directories and are
excluded where appropriate by `.gitignore`.

---

## Optimisation problem

The experiments study recovery of a spectrally sparse signal

```math
x_t = \sum_{k=1}^{r} d_k e^{(2\pi i f_k-\tau_k)t},
\qquad t=0,\ldots,n-1,
```

from only a subset of its time-domain samples.

The key structural observation is that the Hankel lifting

```math
\mathcal{H}x =
\begin{bmatrix}
x_0 & x_1 & \cdots \\
x_1 & x_2 & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
```

has rank $r$ when the signal contains $r$ distinct spectral components.

Because different Hankel anti-diagonals have different multiplicities, the
implementation uses the normalised lifting

```math
\mathcal{G} = \mathcal{H}D^{-1},
\qquad
\mathcal{G}^*\mathcal{G} = I,
```

where $D$ is the diagonal anti-diagonal weighting operator. Writing

```math
y = Dx
```

gives

```math
\mathcal{G}y = \mathcal{H}x.
```

### PGD formulation

Projected Gradient Descent represents the rank-$r$ Hankel matrix by two
low-rank factors,

```math
L(Z)=Z_UZ_V^*,
\qquad
Z_U\in\mathbb{C}^{n_1\times r},
\qquad
Z_V\in\mathbb{C}^{n_2\times r}.
```

The recovery problem is

```math
\min_{Z\in\mathcal{C}}
\left\{
\left\|(I-\mathcal{G}\mathcal{G}^*)L(Z)\right\|_F^2
+
\frac{1}{p}
\left\|
R_\Omega\!\left(\mathcal{G}^*L(Z)-y\right)
\right\|_2^2
+
\frac{\lambda}{2}
\left\|
Z_U^*Z_U-Z_V^*Z_V
\right\|_F^2
\right\},
```

with

```math
\lambda=\frac{1}{4},
\qquad
p=\frac{m}{n}.
```

The three terms have the following roles.

**Hankel-consistency term**

```math
\left\|(I-\mathcal{G}\mathcal{G}^*)L(Z)\right\|_F^2
```

penalises departure of the factorised matrix from the Hankel subspace.

**Data-fidelity term**

```math
\frac{1}{p}
\left\|
R_\Omega\!\left(\mathcal{G}^*L(Z)-y\right)
\right\|_2^2
```

penalises disagreement with the observed samples.

**Balancing term**

```math
\frac{\lambda}{2}
\left\|
Z_U^*Z_U-Z_V^*Z_V
\right\|_F^2
```

prevents one factor from growing while the other shrinks.

Low rank is imposed directly by the factorisation $Z_UZ_V^*$. The feasible
set

```math
\mathcal{C}
=
\left\{
Z:
\|Z\|_{2,\infty}
\le
\sqrt{\frac{\mu c_s r\sigma}{n}}
\right\}
```

enforces the factor incoherence condition through row-wise projection after
each gradient step.

The PGD iteration therefore has the form

```math
Z^{k+1}
=
P_{\mathcal{C}}
\left(
Z^k-\eta_k\nabla F(Z^k)
\right),
```

where the implementation uses a deterministic projected backtracking line
search.

The recovered time-domain signal is obtained from the final factor product by

```math
\widehat{x}
=
D^{-2}\mathcal{H}^*
\left(
Z_UZ_V^*
\right).
```

The implementation and derivation follow Cai, Wang, and Wei (2018), with the
normalised Hankel operator, spectral initialisation, Wirtinger gradient, and
incoherence projection implemented explicitly in `pgd_algorithm.py` and
`utils.py`.


### Symmetric Hankel extension

For the experiments in this repository, $n=127$, so the Hankel lifting is
$64\times64$ and complex symmetric. SHGD exploits this structure by replacing
the asymmetric factorisation

```math
Z_UZ_V^*
```

with the single-factor representation

```math
ZZ^T.
```

This removes the PGD balancing regulariser and provides the basis for
Experiment B, which compares PGD, SHGD, and FIHT on identical problem
instances.



## Installation

Python **3.10 or newer** is recommended.

The repository provides three environment descriptions for different
reproducibility purposes:

- `requirements.txt` contains the minimal pinned dependencies of the original
  environment used for the thesis results;
- `requirements-full.txt` is the complete `pip freeze` snapshot of that
  original results environment, including development-related packages;
- `environment.yml` provides the Mamba/Conda environment used for independent
  validation on the Institute for Numerical and Applied Mathematics
  `maryam` server.

For the fixed thesis-submission archive and exact release provenance, see
[Archived thesis release](#archived-thesis-release).

### Original thesis-results environment

The numerical results reported in the thesis were generated with:

```text
Python: 3.14.5
NumPy: 2.5.1
SciPy: 1.18.0
Matplotlib: 3.11.1
```

For a minimal recreation of the numerical dependencies, use:

```bash
python -m pip install -r requirements.txt
```

For the complete frozen package environment used on the original development
machine, use:

```bash
python -m pip install -r requirements-full.txt
```

The full requirements file intentionally contains additional packages used by
the development environment (for example IPython/Jupyter and debugging
utilities) that are not required by the experiment scripts themselves.

### Recommended for independent validation: Mamba environment

The repository includes `environment.yml`, which defines the numerical
environment used for the final server-side validation.

Create and activate the environment with:

```bash
mamba env create -f environment.yml
mamba activate spectral_cs
```

The committed environment specifies:

```yaml
name: spectral_cs

channels:
  - conda-forge

dependencies:
  - python=3.11
  - numpy=2.4.6
  - scipy=1.17.1
  - matplotlib=3.11.1
```

The same file can also be used with Conda if Mamba is unavailable:

```bash
conda env create -f environment.yml
conda activate spectral_cs
```

### Alternative: Python virtual environment

A standard Python virtual environment can also be used.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib
```

Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib
```

---

## Quick validation

Compile all Python modules:

```bash
python -m compileall -q .
```

Inspect the available command-line options:

```bash
python experiment_a_reproduction.py --help
python experiment_b_shgd_extension.py --help
python figures.py --help
```

A small smoke test should use a separate output directory so that it cannot be
confused with the full experiment:

```bash
python experiment_a_reproduction.py \
    --regime unseparated \
    --p-grid 0.30 0.50 \
    --n-trials 2 \
    --r-max 2 \
    --out-dir smoke_results

python experiment_b_shgd_extension.py \
    --regime unseparated \
    --r-values 3 7 \
    --p-grid 0.30 0.50 \
    --n-trials 2 \
    --out-dir smoke_results
```

These smoke-test files are not suitable for generating the complete thesis
figures.

---
### Mamba / Maryam server validation

The repository was additionally validated on the Institute
for Numerical and Applied Mathematics **Maryam** computing server using a
dedicated Mamba environment.

The validation environment was:

```text
Host: maryam
Python: 3.11.16
Mamba: 2.1.1
NumPy: 2.4.6
SciPy: 1.17.1
Matplotlib: 3.11.1
CPU threads visible: 72
```

The following checks completed successfully:

- all project modules imported without error;
- Experiment A PGD smoke test;
- Experiment B SHGD smoke test;
- one-cell smoke test of **PGD, FIHT, SHGD, ADMM, and ANM**;
- separated- and unseparated-frequency signal regimes;
- JSON and JSONL result writing to scratch storage;
- complete Chapter 5 figure generation (Figures 5.1--5.5);
- Mamba dependency resolution from `environment.yml` using a dry run.

The all-algorithm smoke test was deliberately small (`r=1`, `p=0.70`,
one trial) and is intended only to verify that each implementation executes
correctly in the server environment. These validation outputs are **not**
thesis results and should not be interpreted as replacements for the completed
experimental grids.

For server-side runs, large or temporary outputs should be directed to
appropriate scratch storage rather than the source-code directory. For example:

```bash
python experiment_a_reproduction.py \
    --regime unseparated \
    --out-dir /scratch/<user>/spectral_cs_results
```

Important validation records should be copied from scratch storage to a
persistent or backed-up location after the run.

---


## Running Experiment A

`experiment_a_reproduction.py` performs the Cai--Wang--Wei Figure 1-style
phase-transition reproduction.

### Default protocol

| Parameter | Default |
|---|---:|
| Signal length | `n = 127` |
| Hankel dimensions | `64 x 64` |
| Nominal sampling grid | `p = 0.10, 0.15, ..., 0.95` |
| Sampling rule | without replacement |
| Samples per cell | `m = round(p n)` |
| Trials per `(r,p)` cell | `50` |
| Rank schedule | starts at `r=1`, step size `1` |
| Structural maximum rank | `64` |
| Stopping patience | two consecutive all-zero rank rows for every algorithm |
| Frequency regimes | unseparated and separated |
| Separated minimum distance | `1.5/n`, including wraparound |
| Algorithms | PGD, FIHT, ANM, ADMM |
| Success criterion | relative error `<= 1e-3` |

Run the two regimes separately:

```bash
python experiment_a_reproduction.py \
    --regime unseparated \
    --time-budget 240

python experiment_a_reproduction.py \
    --regime separated \
    --time-budget 240
```

The time budget is measured in seconds. Re-run the **same command** until the
program reports that the regime is complete. The checkpoint records the exact
protocol, rank schedule, seed, solver parameters, and completed cells; changing
these settings while resuming an existing output file is rejected to prevent
silent corruption.

To run without a call-level time limit, omit `--time-budget`:

```bash
python experiment_a_reproduction.py --regime unseparated
python experiment_a_reproduction.py --regime separated
```

Useful customisations include:

```bash
python experiment_a_reproduction.py \
    --regime unseparated \
    --algorithms PGD FIHT \
    --n-trials 10 \
    --r-max 12 \
    --out-dir custom_results
```

Use a new output directory whenever the protocol is changed.

### Experiment A outputs

During execution:

```text
results/experiment_a_unseparated.json
results/experiment_a_unseparated_trials/
results/experiment_a_separated.json
results/experiment_a_separated_trials/
```

After a regime completes, its per-cell files are merged into:

```text
results/experiment_a_unseparated_trials.jsonl
results/experiment_a_separated_trials.jsonl
```

The checkpoint JSON contains aggregate cell statistics and the extracted
80%-success rank boundary. The JSONL files retain individual paired trial
records for later auditing and statistical analysis.

---

## Running Experiment B

`experiment_b_shgd_extension.py` compares asymmetric PGD, symmetric-factor
SHGD, and FIHT on a fixed grid.

### Default protocol

| Parameter | Default |
|---|---:|
| Signal length | `n = 127` |
| Rank grid | `r in {1, 3, 5, 7, 9, 11}` |
| Nominal sampling grid | `p = 0.10, 0.15, ..., 0.95` |
| Trials per `(r,p)` cell | `30` |
| Frequency regimes | unseparated and separated |
| Algorithms | PGD, SHGD, FIHT |
| Success criterion | relative error `<= 1e-3` |

Run both regimes:

```bash
python experiment_b_shgd_extension.py \
    --regime unseparated \
    --time-budget 240

python experiment_b_shgd_extension.py \
    --regime separated \
    --time-budget 240
```

Re-run each command until the corresponding regime is complete.

A custom fixed grid can be supplied explicitly:

```bash
python experiment_b_shgd_extension.py \
    --regime separated \
    --r-values 1 3 5 7 9 11 \
    --p-grid 0.20 0.30 0.40 0.50 0.60 \
    --n-trials 20 \
    --out-dir custom_results
```

### Experiment B outputs

During execution:

```text
results/experiment_b_unseparated.json
results/experiment_b_unseparated_trials/
results/experiment_b_separated.json
results/experiment_b_separated_trials/
```

After completion:

```text
results/experiment_b_unseparated_trials.jsonl
results/experiment_b_separated_trials.jsonl
```

---

## Generating the thesis figures

After all four merged JSONL files exist, generate Figures 5.1--5.5 in one run:

```bash
python figures.py --results-dir results --output-dir images
```

The default formats are PNG and PDF at 300 dpi. SVG is also supported:

```bash
python figures.py \
    --results-dir results \
    --output-dir images \
    --formats png pdf svg \
    --dpi 300
```

Expected inputs:

```text
results/experiment_a_unseparated_trials.jsonl
results/experiment_a_separated_trials.jsonl
results/experiment_b_unseparated_trials.jsonl
results/experiment_b_separated_trials.jsonl
```

Generated figures:

| Thesis figure | Output stem | Content |
|---|---|---|
| Figure 5.1 | `figure_5_1_experiment_a_separated` | Experiment A separated-frequency heatmaps |
| Figure 5.2 | `figure_5_2_experiment_a_unseparated` | Experiment A unseparated-frequency heatmaps |
| Figure 5.3 | `figure_5_3_experiment_a_80%_boundary` | 80%-success rank boundaries |
| Figure 5.4 | `figure_5_4_experiment_b_success_curves` | PGD, SHGD, and FIHT success curves at `r=3,7,9,11` |
| Figure 5.5 | `figure_5_5_experiment_b_computational_cost` | Successful-trial iterations at `r=9` and wall-clock time at `r=11` |

Figure 5.4 requires Experiment B data for ranks `3`, `7`, `9`, and `11`.
Figure 5.5 requires separated-frequency data for ranks `9` and `11`. The
default Experiment B grid includes all required ranks.

---

## Repository Structure

| Path | Purpose |
|---|---|
| `utils.py` | Signal generation, Hankel operators, deterministic seed construction, and shared numerical utilities |
| `pgd_algorithm.py` | Projected gradient descent implementation for factorised low-rank Hankel recovery |
| `fiht_baseline.py` | Fast iterative hard thresholding baseline |
| `admm_baseline.py` | Hankel nuclear-norm recovery using ADMM |
| `anm_emac_baselines.py` | Atomic norm minimisation and EMaC-compatible recovery interfaces |
| `shgd_algorithm.py` | Symmetric Hankel gradient descent implementation |
| `comparison_engine.py` | Shared paired-trial execution, algorithm registry, statistics, checkpointing, and result storage |
| `experiment_a_reproduction.py` | Experiment A: reproduction and extended comparison of PGD, FIHT, ANM, and EMaC/ADMM |
| `experiment_b_shgd_extension.py` | Experiment B: comparison of PGD, SHGD, and FIHT |
| `figures.py` | Generates the Chapter 5 figures from completed experiment records |
| `requirements.txt` | Minimal pinned Python dependencies matching the original thesis-results environment |
| `requirements-full.txt` | Complete `pip freeze` snapshot of the original development/results environment |
| `environment.yml` | Reproducible Mamba/Conda environment used for final independent server-side validation |
| `results/` | Version-controlled JSON and JSONL outputs from the completed numerical experiments |

The repository contains the source code, environment specifications, and
numerical records required for the thesis experiments. The earlier
demonstration notebook has been removed because it is not part of the
experiment pipeline and was not used to generate the reported results.

---

## Results and Reproducibility

The completed numerical results are stored in the version-controlled `results/`
directory. Including these files in the repository provides an audit trail
between the experiment code, the numerical findings, and the figures and tables
reported in the thesis.

The fixed archival version of the thesis software and records is available at
https://doi.org/10.25625/NFBEGZ and corresponds to Git tag `v1.0-thesis`,
commit `4089eb6b40b8c3cad68115926f0c33478c98317d`.

The reported thesis results were generated in the original Python environment
documented by `requirements.txt` and `requirements-full.txt` (Python 3.14.5,
NumPy 2.5.1, SciPy 1.18.0, Matplotlib 3.11.1).

The codebase was subsequently and independently smoke-tested on the Maryam
server using the separate Mamba environment defined in `environment.yml`
(Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, Matplotlib 3.11.1). This provides an
additional check that the repository executes outside the original development
machine and is not intended to imply that the Maryam environment generated the
reported thesis results.

Depending on which experiment regimes have been completed, the directory may
contain:

```text
results/
├── experiment_a_unseparated.json
├── experiment_a_unseparated_trials.jsonl
├── experiment_a_separated.json
├── experiment_a_separated_trials.jsonl
├── experiment_b_unseparated.json
├── experiment_b_unseparated_trials.jsonl
├── experiment_b_separated.json
└── experiment_b_separated_trials.jsonl
```

---

## Implemented methods

### Projected Gradient Descent

PGD factorises the lifted Hankel matrix asymmetrically as

```text
H(x) approximately equals Z_U Z_V^*,
```

and minimises a regularised non-convex objective in the factor matrices. The
implementation uses:

- normalised Hankel lifting;
- one-step spectral initialisation;
- an inflated leading-singular-value estimate;
- row-wise projection onto the incoherence set;
- the balancing regulariser;
- a monotone backtracking line search;
- stopping rules based on relative iterate and objective changes.

The unknown ground truth is used only for post-run diagnostics and never for
step acceptance or stopping.

### Fast Iterative Hard Thresholding

FIHT alternates a data-consistency update with rank-`r` truncation of the
Hankel lift. SVD failures are handled without aborting the complete experiment,
and the shared SVD utility falls back from the fast `gesdd` LAPACK driver to the
more conservative `gesvd` driver when necessary.

### Hankel nuclear-norm ADMM

The convex baseline solves

```text
minimise ||H(x)||_* subject to x[Omega] = y_obs.
```

This is the Hankel nuclear-norm program associated with EMaC. The repository
reports it as `ADMM` because it is solved by the custom implementation in
`admm_baseline.py`, not by the original paper's CVX configuration.

### Atomic Norm Minimisation

ANM is implemented through an ADMM splitting of the positive-semidefinite
Toeplitz moment formulation. It does not require `cvxpy` or another external
semidefinite-programming package.

### Symmetric Hankel Gradient Descent

For odd `n`, the Hankel lift is square and complex symmetric. SHGD exploits the
factorisation

```text
H(x) approximately equals Z Z^T,
```

using one factor matrix and no asymmetric balancing term. The initial factor is
constructed from a Takagi factorisation, with deterministic seeding for the
truncated sparse-SVD path.

---

## Reproducibility design

The two experiment drivers share the same execution engine so that their
protocols cannot silently drift apart.

For every `(regime, r, p-index, trial)` tuple, the engine derives one integer
seed. All algorithms in that cell therefore receive the same signal and the
same sampling pattern. SHGD also receives a deterministic initialisation seed.

Each checkpoint stores:

- the base seed;
- signal length and rank grid;
- nominal and realised sampling ratios;
- sampling convention and rounding rule;
- success tolerance;
- damping and separation settings;
- NumPy and SciPy versions;
- every numerical solver parameter;
- completed cells and aggregate statistics.

Trial records preserve the algorithm status, reconstruction error, runtime,
iteration count, success indicator, and seed. Timing and iteration summaries
are reported separately for successful trials because averages that combine
successful runs with failed or maximum-iteration runs can be misleading.

---

## Reproduction scope and documented deviations

Experiment A is a **reproduction attempt**, not a claim of bit-for-bit identity
with the published MATLAB experiment. The following differences are explicit:

1. ANM and Hankel nuclear-norm recovery are custom ADMM implementations rather
   than the original CVX solver configuration.
2. Algorithms are advanced on one shared rank schedule to keep all rows paired;
   the original paper describes escalation at a more method-specific level.
3. PGD and FIHT use stopping tests evaluated on the projected candidate iterate.
4. The implementation uses dense NumPy/SciPy linear algebra and does not claim
   the FFT/PROPACK runtime complexity of the original implementation.
5. Sampling uses `m = round(p n)` distinct indices without replacement, and the
   realised ratio `m/n` is stored alongside the nominal grid value.

Accordingly, qualitative phase-transition structure is the primary comparison.
Absolute boundaries and runtimes should not be presented as exact replications
of the original software environment.

---

## Important implementation corrections

Earlier development versions contained errors that made several methods fail
systematically. The current implementation includes the following corrections:

1. **Hankel adjoint:** the true adjoint sums along anti-diagonals; averaging is
   retained only as a separately identified pseudo-inverse operation where the
   ADMM least-squares update requires it.
2. **Signal reconstruction:** factorised Hankel estimates are mapped back with
   the required `D^-2 H*` reconstruction.
3. **Coordinate consistency:** the data-fidelity target uses the preconditioned
   observed coordinates `D[Omega] * y_obs`.
4. **Incoherence threshold:** the projection threshold is based on the inflated
   singular-value estimate used by the paper's initialisation analysis.
5. **Balancing gradient:** the derivative of the PGD balancing regulariser is
   included in both factor gradients.

The PGD Wirtinger gradient and the corrected operators were checked numerically
on small test problems during development before the full grids were run.

---

## Mathematical model

The signal is a sum of `r` complex exponentials:

```text
x[t] = sum_{k=1}^r d_k exp((2 pi i f_k - tau_k)t),
       t = 0, ..., n-1.
```

Its Hankel lift is defined by

```text
H(x)[i,j] = x[i+j],
```

with dimensions satisfying `n1 + n2 = n + 1`. For `n=127`,
`n1=n2=64`. Under distinct non-zero spectral components, the Hankel rank is the
model order `r`.

The normalised lifting operator is

```text
G = H D^-1,
```

where `D^2` contains the anti-diagonal multiplicities. This gives the isometry
property

```text
G* G = I.
```

---

## Citation

If you use the source code, numerical results, or figure-generation scripts from
this repository, please cite the archived thesis software dataset:

> Duah, Michael Hubert (2026). *Spectral Compressed Sensing via Projected
> Gradient Descent: Numerical Simulation and Analysis — Source Code and
> Experiment Records*. GRO.data, Version 1.0.
> https://doi.org/10.25625/NFBEGZ

The archived release corresponds to Git tag `v1.0-thesis` and commit
`4089eb6b40b8c3cad68115926f0c33478c98317d`.

The mathematical methods implemented in this repository should additionally be
cited through their respective original publications listed in the
**Main references** section above.
