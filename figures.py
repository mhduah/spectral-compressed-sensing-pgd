"""
Generate all Chapter 5 figures in one run.

This script regenerates Figures 5.1--5.5 from the merged trial-level
JSONL files stored in the results directory.

Expected input files
--------------------
results/experiment_a_unseparated_trials.jsonl
results/experiment_a_separated_trials.jsonl
results/experiment_b_unseparated_trials.jsonl
results/experiment_b_separated_trials.jsonl

Generated outputs
-----------------
Figure 5.1 : Experiment A, separated success-probability heatmaps
Figure 5.2 : Experiment A, unseparated success-probability heatmaps
Figure 5.3 : Experiment A, 80%-success boundaries
Figure 5.4 : Experiment B, success-probability curves (r = 3, 7, 9, 11)
Figure 5.5 : Experiment B, computational cost
             - mean iterations at r = 9
             - mean wall-clock time at r = 11

Run from the project root:
    python figures.py

Optional:
    python figures.py --results-dir results --output-dir images
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


SUCCESS_THRESHOLD = 0.80

EXPERIMENT_FILES = {
    ("A", "unseparated"): "experiment_a_unseparated_trials.jsonl",
    ("A", "separated"): "experiment_a_separated_trials.jsonl",
    ("B", "unseparated"): "experiment_b_unseparated_trials.jsonl",
    ("B", "separated"): "experiment_b_separated_trials.jsonl",
}

EXPERIMENT_A_ORDER = ("PGD", "FIHT", "ANM", "ADMM")
EXPERIMENT_B_ORDER = ("PGD", "SHGD", "FIHT")

DISPLAY_NAMES = {
    "PGD": "PGD",
    "FIHT": "FIHT",
    "SHGD": "SHGD",
    "ANM": "ANM (atomic norm, ADMM)",
    "ADMM": "ADMM (Hankel nuclear norm)",
}

MARKERS = {
    "PGD": "o",
    "FIHT": "s",
    "SHGD": "^",
    "ANM": "D",
    "ADMM": "v",
}


@dataclass(frozen=True)
class TrialData:
    algorithms: tuple[str, ...]
    ranks: np.ndarray
    p_grid: np.ndarray
    success_prob: dict[str, np.ndarray]
    trial_counts: dict[str, np.ndarray]
    success_counts: dict[str, np.ndarray]
    mean_iterations_success: dict[str, np.ndarray]
    mean_runtime_success: dict[str, np.ndarray]


def get_first(record: dict[str, Any], names: tuple[str, ...], *, default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def canonical_algorithm(name: Any) -> str:
    compact = "".join(ch for ch in str(name).upper() if ch.isalnum())

    if compact == "PGD":
        return "PGD"
    if compact == "FIHT":
        return "FIHT"
    if compact == "SHGD":
        return "SHGD"
    if "ANM" in compact or "ATOMIC" in compact:
        return "ANM"
    if (
        compact == "ADMM"
        or "EMAC" in compact
        or "HANKELADMM" in compact
        or "NUCLEARNORMADMM" in compact
        or "HANKELNUCLEARNORM" in compact
    ):
        return "ADMM"

    raise ValueError(
        f"Unrecognised algorithm label {name!r}. "
        "Update canonical_algorithm() to include it."
    )


def parse_success(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "success", "successful"}:
        return True
    if text in {"false", "0", "no", "failure", "failed"}:
        return False

    raise ValueError(f"Cannot interpret success value {value!r}.")


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}."
                )
            yield record


def aggregate_trials(path: Path) -> TrialData:
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n  {path}\n"
            "Check the results directory or filename."
        )

    success_counts_raw: defaultdict[tuple[str, int, float], int] = defaultdict(int)
    trial_counts_raw: defaultdict[tuple[str, int, float], int] = defaultdict(int)
    iteration_sums_raw: defaultdict[tuple[str, int, float], float] = defaultdict(float)
    iteration_success_raw: defaultdict[tuple[str, int, float], int] = defaultdict(int)
    runtime_sums_raw: defaultdict[tuple[str, int, float], float] = defaultdict(float)
    runtime_success_raw: defaultdict[tuple[str, int, float], int] = defaultdict(int)

    observed_algorithms: set[str] = set()
    observed_ranks: set[int] = set()
    observed_p: set[float] = set()

    for line_number, record in enumerate(read_jsonl(path), start=1):
        algorithm_value = get_first(record, ("algorithm", "algo", "method"))
        rank_value = get_first(record, ("r", "rank"))
        p_value = get_first(record, ("p", "sampling_ratio", "nominal_p", "p_nominal"))
        success_value = get_first(record, ("success", "succeeded", "is_success"))

        if algorithm_value is None or rank_value is None or p_value is None or success_value is None:
            raise KeyError(
                f"{path}, record {line_number}: missing one of the required fields "
                "(algorithm, r/rank, p/sampling_ratio, success)."
            )

        algorithm = canonical_algorithm(algorithm_value)
        rank = int(rank_value)
        p = round(float(p_value), 10)
        success = parse_success(success_value)

        key = (algorithm, rank, p)
        trial_counts_raw[key] += 1
        success_counts_raw[key] += int(success)

        observed_algorithms.add(algorithm)
        observed_ranks.add(rank)
        observed_p.add(p)

        if success:
            iterations = parse_optional_float(
                get_first(record, ("iterations", "iter", "n_iter", "num_iterations"))
            )
            runtime = parse_optional_float(
                get_first(record, ("time", "runtime", "wall_time", "elapsed", "elapsed_time"))
            )

            if iterations is not None:
                iteration_sums_raw[key] += iterations
                iteration_success_raw[key] += 1

            if runtime is not None:
                runtime_sums_raw[key] += runtime
                runtime_success_raw[key] += 1

    if not trial_counts_raw:
        raise ValueError(f"No trial records found in {path}.")

    algorithms = tuple(sorted(observed_algorithms))
    ranks = np.array(sorted(observed_ranks), dtype=int)
    p_grid = np.array(sorted(observed_p), dtype=float)

    rank_index = {rank: index for index, rank in enumerate(ranks)}
    p_index = {p: index for index, p in enumerate(p_grid)}

    success_prob: dict[str, np.ndarray] = {}
    trial_counts: dict[str, np.ndarray] = {}
    success_counts: dict[str, np.ndarray] = {}
    mean_iterations_success: dict[str, np.ndarray] = {}
    mean_runtime_success: dict[str, np.ndarray] = {}

    for algorithm in algorithms:
        trials_matrix = np.zeros((len(ranks), len(p_grid)), dtype=int)
        success_matrix = np.zeros((len(ranks), len(p_grid)), dtype=int)
        iter_mean_matrix = np.full((len(ranks), len(p_grid)), np.nan, dtype=float)
        time_mean_matrix = np.full((len(ranks), len(p_grid)), np.nan, dtype=float)

        for rank in ranks:
            for p in p_grid:
                key = (algorithm, int(rank), round(float(p), 10))
                i = rank_index[int(rank)]
                j = p_index[round(float(p), 10)]

                trials_matrix[i, j] = trial_counts_raw.get(key, 0)
                success_matrix[i, j] = success_counts_raw.get(key, 0)

                iter_n = iteration_success_raw.get(key, 0)
                if iter_n > 0:
                    iter_mean_matrix[i, j] = iteration_sums_raw[key] / iter_n

                time_n = runtime_success_raw.get(key, 0)
                if time_n > 0:
                    time_mean_matrix[i, j] = runtime_sums_raw[key] / time_n

        with np.errstate(divide="ignore", invalid="ignore"):
            prob_matrix = np.divide(
                success_matrix,
                trials_matrix,
                out=np.full(success_matrix.shape, np.nan, dtype=float),
                where=trials_matrix > 0,
            )

        success_prob[algorithm] = prob_matrix
        trial_counts[algorithm] = trials_matrix
        success_counts[algorithm] = success_matrix
        mean_iterations_success[algorithm] = iter_mean_matrix
        mean_runtime_success[algorithm] = time_mean_matrix

    return TrialData(
        algorithms=algorithms,
        ranks=ranks,
        p_grid=p_grid,
        success_prob=success_prob,
        trial_counts=trial_counts,
        success_counts=success_counts,
        mean_iterations_success=mean_iterations_success,
        mean_runtime_success=mean_runtime_success,
    )


def require_algorithms(data: TrialData, expected: Iterable[str], source: str) -> None:
    missing = [algorithm for algorithm in expected if algorithm not in data.success_prob]
    if missing:
        raise ValueError(
            f"{source} does not contain the required algorithms: {missing}. "
            f"Found: {list(data.algorithms)}"
        )


def sampling_extent(p_grid: np.ndarray) -> tuple[float, float]:
    if len(p_grid) == 1:
        half_step = 0.025
    else:
        half_step = float(np.median(np.diff(p_grid))) / 2.0
    return float(p_grid[0] - half_step), float(p_grid[-1] + half_step)


def rank_extent(ranks: np.ndarray) -> tuple[float, float]:
    if len(ranks) == 1:
        half_step = 0.5
    else:
        half_step = float(np.median(np.diff(ranks))) / 2.0
    return float(ranks[0] - half_step), float(ranks[-1] + half_step)


def sensible_rank_ticks(ranks: np.ndarray, maximum_ticks: int = 7) -> np.ndarray:
    if len(ranks) <= maximum_ticks:
        return ranks
    locations = np.linspace(0, len(ranks) - 1, maximum_ticks)
    indices = np.unique(np.rint(locations).astype(int))
    return ranks[indices]

def fixed_rank_ticks(ranks: np.ndarray) -> list[int]:
    ticks = [1, 10, 20, 30, 40, 50, 60]
    return [t for t in ticks if ranks[0] <= t <= ranks[-1]]

def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        output_path = output_dir / f"{stem}.{extension}"
        save_kwargs = {"bbox_inches": "tight"}
        if extension.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs["dpi"] = dpi
        figure.savefig(output_path, **save_kwargs)
        print(f"[SAVED] {output_path}")


def largest_rank_at_threshold(
    data: TrialData,
    algorithm: str,
    threshold: float = SUCCESS_THRESHOLD,
) -> np.ndarray:
    matrix = data.success_prob[algorithm]
    boundary = np.zeros(len(data.p_grid), dtype=float)
    for column in range(len(data.p_grid)):
        qualifying = data.ranks[matrix[:, column] >= threshold]
        boundary[column] = float(np.max(qualifying)) if len(qualifying) else 0.0
    return boundary


def plot_experiment_a_heatmaps(
    data: TrialData,
    regime: str,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    require_algorithms(data, EXPERIMENT_A_ORDER, f"Experiment A ({regime})")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.5, 8.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    p_left, p_right = sampling_extent(data.p_grid)
    r_bottom, r_top = rank_extent(data.ranks)
    image = None

    for axis, algorithm in zip(axes.flat, EXPERIMENT_A_ORDER):
        image = axis.imshow(
            data.success_prob[algorithm],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(p_left, p_right, r_bottom, r_top),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        axis.set_title(DISPLAY_NAMES[algorithm])
        axis.set_xlabel(r"sampling ratio $p$")
        axis.set_ylabel(r"rank $r$")
        axis.set_xticks(np.arange(0.10, 1.00, 0.20))
        axis.set_yticks(fixed_rank_ticks(data.ranks))
        axis.grid(False)

    colorbar = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
    colorbar.set_label("empirical success probability")

    if regime == "separated":
        fig.suptitle(r"Experiment A, separated frequencies ($n=127$, 50 trials/cell)")
        stem = "figure_5_1_experiment_a_separated"
    else:
        fig.suptitle(r"Experiment A, unseparated frequencies ($n=127$, 50 trials/cell)")
        stem = "figure_5_2_experiment_a_unseparated"

    save_figure(fig, output_dir, stem, formats, dpi)
    plt.close(fig)


def plot_experiment_a_boundaries(
    unseparated: TrialData,
    separated: TrialData,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    require_algorithms(separated, EXPERIMENT_A_ORDER, "Experiment A (separated)")
    require_algorithms(unseparated, EXPERIMENT_A_ORDER, "Experiment A (unseparated)")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.5),
        sharey=True,
        constrained_layout=True,
    )

    regimes = (
        ("separated frequencies", separated),
        ("unseparated frequencies", unseparated),
    )

    common_max_rank = max(
        int(np.max(separated.ranks)),
        int(np.max(unseparated.ranks)),
    )

    for axis, (title, data) in zip(axes, regimes):
        for algorithm in EXPERIMENT_A_ORDER:
            boundary = largest_rank_at_threshold(data, algorithm)
            axis.plot(
                data.p_grid,
                boundary,
                marker=MARKERS[algorithm],
                markersize=4,
                linewidth=1.4,
                label=algorithm,
            )

        axis.set_title(title)
        axis.set_xlabel(r"sampling ratio $p$")
        axis.set_xlim(float(data.p_grid[0]) - 0.02, float(data.p_grid[-1]) + 0.02)
        axis.set_ylim(0, common_max_rank + 2)
        axis.set_xticks(np.arange(0.10, 1.00, 0.10))
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"80%-success boundary $r_{0.8}(p)$")
    axes[0].legend(loc="upper left", frameon=True)
    fig.suptitle(r"Experiment A: 80% success boundary vs. rank")

    save_figure(
        fig,
        output_dir,
        "figure_5_3_experiment_a_80%_boundary",
        formats,
        dpi,
    )
    plt.close(fig)


def plot_experiment_b_success_curves(
    separated: TrialData,
    unseparated: TrialData,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    require_algorithms(separated, EXPERIMENT_B_ORDER, "Experiment B (separated)")
    require_algorithms(unseparated, EXPERIMENT_B_ORDER, "Experiment B (unseparated)")

    ranks_to_plot = (3, 7, 9, 11)
    for rank in ranks_to_plot:
        if rank not in separated.ranks or rank not in unseparated.ranks:
            raise ValueError(
                f"Figure 5.4 requires rank {rank}, but it is missing from the data."
            )

    if not np.allclose(separated.p_grid, unseparated.p_grid):
        raise ValueError("Experiment B separated/unseparated p-grids do not match.")

    p_grid = separated.p_grid

    fig, axes = plt.subplots(
        2,
        len(ranks_to_plot),
        figsize=(13.0, 6.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    regime_rows = (
        ("separated", separated),
        ("unseparated", unseparated),
    )

    rank_index_sep = {rank: i for i, rank in enumerate(separated.ranks)}
    rank_index_unsep = {rank: i for i, rank in enumerate(unseparated.ranks)}

    for row, (regime_name, data) in enumerate(regime_rows):
        rank_index = rank_index_sep if regime_name == "separated" else rank_index_unsep

        for column, rank in enumerate(ranks_to_plot):
            axis = axes[row, column]
            i = rank_index[rank]

            for algorithm in EXPERIMENT_B_ORDER:
                y = data.success_prob[algorithm][i, :]
                axis.plot(
                    p_grid,
                    y,
                    marker=MARKERS[algorithm],
                    markersize=3.5,
                    linewidth=1.2,
                    label=algorithm,
                )

            axis.set_title(rf"$r={rank}$")
            x_pad = 0.03
            y_pad_lower = 0.03
            y_pad_upper = 0.04
            axis.set_xlim(float(p_grid[0]) - x_pad, float(p_grid[-1]) + x_pad)
            axis.set_ylim(0.0 - y_pad_lower, 1.0 + y_pad_upper)
            axis.set_xticks(np.arange(0.10, 1.00, 0.20))
            axis.set_yticks(np.arange(0.0, 1.01, 0.2))
            axis.grid(True, alpha=0.3)

            if row == 1:
                axis.set_xlabel(r"sampling ratio $p$")

            if column == 0:
                axis.set_ylabel(
                    f"{regime_name}\n" + "success probability"
                )

    axes[0, 0].legend(loc="lower right", frameon=True)
    fig.suptitle(
        r"Experiment B: PGD vs. SHGD vs. FIHT success probability "
        r"($n=127$, 30 trials/cell)"
    )

    save_figure(
        fig,
        output_dir,
        "figure_5_4_experiment_b_success_curves",
        formats,
        dpi,
    )
    plt.close(fig)


def find_rank_row(data: TrialData, rank: int) -> int:
    locations = np.where(data.ranks == rank)[0]
    if len(locations) == 0:
        raise ValueError(f"Required rank {rank} is missing from the data.")
    return int(locations[0])


def plot_experiment_b_cost(
    separated: TrialData,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    require_algorithms(separated, EXPERIMENT_B_ORDER, "Experiment B (separated)")

    r_iter = 9
    r_time = 11

    idx_iter = find_rank_row(separated, r_iter)
    idx_time = find_rank_row(separated, r_time)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.4),
        constrained_layout=True,
    )

    # Left panel: mean iterations to convergence at r = 9
    ax = axes[0]
    for algorithm in EXPERIMENT_B_ORDER:
        y = separated.mean_iterations_success[algorithm][idx_iter, :]
        ax.plot(
            separated.p_grid,
            y,
            marker=MARKERS[algorithm],
            markersize=4,
            linewidth=1.4,
            label=algorithm,
        )

    ax.set_title(rf"Mean iterations to convergence, $r={r_iter}$")
    ax.set_xlabel(r"sampling ratio $p$")
    ax.set_ylabel("iterations (successful trials only)")
    ax.set_xticks(np.arange(0.10, 1.00, 0.10))
    ax.grid(True, alpha=0.3)

    # Right panel: mean wall-clock time to convergence at r = 11
    ax = axes[1]
    for algorithm in EXPERIMENT_B_ORDER:
        y = separated.mean_runtime_success[algorithm][idx_time, :]
        ax.plot(
            separated.p_grid,
            y,
            marker=MARKERS[algorithm],
            markersize=4,
            linewidth=1.4,
            label=algorithm,
        )

    ax.set_title(rf"Mean runtime to convergence, $r={r_time}$")
    ax.set_xlabel(r"sampling ratio $p$")
    ax.set_ylabel("wall-clock time (s)")
    ax.set_xticks(np.arange(0.10, 1.00, 0.10))
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", frameon=True)
    fig.suptitle(
        r"Experiment B computational cost, separated frequencies ($n=127$)"
    )

    save_figure(
        fig,
        output_dir,
        "figure_5_5_experiment_b_computational_cost",
        formats,
        dpi,
    )
    plt.close(fig)


def validate_trial_counts(
    data: TrialData,
    expected_algorithms: Iterable[str],
    expected_trials: int,
    label: str,
) -> None:
    for algorithm in expected_algorithms:
        if algorithm not in data.trial_counts:
            continue

        observed = data.trial_counts[algorithm]
        completed = observed[observed > 0]

        if completed.size == 0:
            print(f"[WARNING] {label}, {algorithm}: no completed cells.")
            continue

        unexpected = completed[completed != expected_trials]
        if unexpected.size:
            unique_counts = sorted(set(int(value) for value in unexpected))
            print(
                f"[WARNING] {label}, {algorithm}: some non-empty cells have "
                f"trial counts {unique_counts}; expected {expected_trials}."
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all Chapter 5 thesis figures in one run."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing the merged JSONL trial files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("images"),
        help="Directory where the figures will be saved.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "pdf"),
        choices=("png", "pdf", "svg"),
        help="Output formats. Default: png pdf.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG resolution. Default: 300.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    formats = tuple(args.formats)

    loaded: dict[tuple[str, str], TrialData] = {}

    print("Loading trial-level JSONL files")
    print("-" * 72)

    for key, filename in EXPERIMENT_FILES.items():
        path = args.results_dir / filename
        print(f"[LOAD] {path}")
        loaded[key] = aggregate_trials(path)

    a_unseparated = loaded[("A", "unseparated")]
    a_separated = loaded[("A", "separated")]
    b_unseparated = loaded[("B", "unseparated")]
    b_separated = loaded[("B", "separated")]

    validate_trial_counts(
        a_unseparated, EXPERIMENT_A_ORDER, 50, "Experiment A, unseparated"
    )
    validate_trial_counts(
        a_separated, EXPERIMENT_A_ORDER, 50, "Experiment A, separated"
    )
    validate_trial_counts(
        b_unseparated, EXPERIMENT_B_ORDER, 30, "Experiment B, unseparated"
    )
    validate_trial_counts(
        b_separated, EXPERIMENT_B_ORDER, 30, "Experiment B, separated"
    )

    print("\nGenerating figures")
    print("-" * 72)

    plot_experiment_a_heatmaps(
        a_unseparated,
        regime="unseparated",
        output_dir=args.output_dir,
        formats=formats,
        dpi=args.dpi,
    )
    plot_experiment_a_heatmaps(
        a_separated,
        regime="separated",
        output_dir=args.output_dir,
        formats=formats,
        dpi=args.dpi,
    )
    plot_experiment_a_boundaries(
        a_unseparated,
        a_separated,
        output_dir=args.output_dir,
        formats=formats,
        dpi=args.dpi,
    )
    plot_experiment_b_success_curves(
        b_separated,
        b_unseparated,
        output_dir=args.output_dir,
        formats=formats,
        dpi=args.dpi,
    )
    plot_experiment_b_cost(
        b_separated,
        output_dir=args.output_dir,
        formats=formats,
        dpi=args.dpi,
    )

    print("\nAll Chapter 5 figures were generated successfully.")


if __name__ == "__main__":
    main()
