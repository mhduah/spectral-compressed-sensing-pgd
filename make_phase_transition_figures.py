#!/usr/bin/env python3
"""Generate phase-transition heatmaps from reproduction JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    required = {"p_grid", "r_values"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing required keys in {path}: {sorted(missing)}")
    return data


def available_algorithms(data: dict) -> list[str]:
    return [
        key for key, value in data.items()
        if isinstance(value, dict) and "success_prob" in value
    ]


def plot_heatmap(data: dict, algorithm: str, output_path: Path, title_suffix: str) -> None:
    success = np.asarray(data[algorithm]["success_prob"], dtype=float)
    p_grid = np.asarray(data["p_grid"], dtype=float)
    r_values = np.asarray(data["r_values"], dtype=int)

    expected = (len(r_values), len(p_grid))
    if success.shape != expected:
        raise ValueError(
            f"{algorithm} success grid has shape {success.shape}; expected {expected}."
        )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    image = ax.imshow(
        success,
        origin="lower",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        extent=[p_grid.min(), p_grid.max(), r_values.min(), r_values.max()],
    )
    ax.set_xlabel("Sampling ratio m/n")
    ax.set_ylabel("Model order r")
    ax.set_title(f"{algorithm} phase transition — {title_suffix}")
    fig.colorbar(image, ax=ax, label="Recovery success probability")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_file(input_path: str, outdir: Path, label: str) -> None:
    data = load_results(input_path)
    algorithms = available_algorithms(data)
    if not algorithms:
        raise ValueError(f"No algorithm success grids found in {input_path}.")

    outdir.mkdir(parents=True, exist_ok=True)
    for algorithm in algorithms:
        filename = f"phase_transition_{label}_{algorithm.lower()}.png"
        plot_heatmap(data, algorithm, outdir / filename, label)
        print(f"Saved {outdir / filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unsep", help="Unseparated-frequency results JSON")
    parser.add_argument("--sep", help="Separated-frequency results JSON")
    parser.add_argument("--outdir", default="images", help="Output directory")
    args = parser.parse_args()

    if not args.unsep and not args.sep:
        parser.error("Provide at least one of --unsep or --sep.")

    outdir = Path(args.outdir)
    if args.unsep:
        render_file(args.unsep, outdir, "unseparated")
    if args.sep:
        render_file(args.sep, outdir, "separated")


if __name__ == "__main__":
    main()
