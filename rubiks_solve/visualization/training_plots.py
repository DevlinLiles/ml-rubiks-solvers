"""Training metric plots for ML solvers."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rubiks_solve.visualization.cube_3d import save_or_show


def plot_loss_curve(
    metrics_df: pd.DataFrame,
    solver_name: str,
    output_path: "Path | None" = None,
    interactive: bool = False,
) -> plt.Figure:
    """Plot training loss over epochs.

    Expects a ``metrics_df`` with at least an ``epoch`` column and a ``loss``
    column.  An optional ``mae`` column is plotted on a secondary y-axis when
    present.

    Args:
        metrics_df:   DataFrame of per-epoch metrics.
        solver_name:  Human-readable solver label used in the title.
        output_path:  If given, the figure is saved as PNG to this path.
        interactive:  When *True*, ``plt.show()`` is called.

    Returns:
        The rendered :class:`matplotlib.figure.Figure`.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    x_col = "epoch" if "epoch" in metrics_df.columns else metrics_df.index

    ax.plot(
        metrics_df[x_col] if "epoch" in metrics_df.columns else metrics_df.index,
        metrics_df["loss"],
        label="Loss (MSE)",
        color="steelblue",
        linewidth=2,
    )

    if "mae" in metrics_df.columns:
        ax2 = ax.twinx()
        ax2.plot(
            metrics_df[x_col] if "epoch" in metrics_df.columns else metrics_df.index,
            metrics_df["mae"],
            label="MAE",
            color="darkorange",
            linewidth=1.5,
            linestyle="--",
        )
        ax2.set_ylabel("MAE", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")
        lines2, labels2 = ax2.get_legend_handles_labels()
    else:
        lines2, labels2 = [], []

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{solver_name} — Training Loss")
    ax.grid(True, alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    save_or_show(fig, output_path, interactive)
    return fig


def plot_solve_rate(
    metrics_df: pd.DataFrame,
    solver_name: str,
    output_path: "Path | None" = None,
    interactive: bool = False,
) -> plt.Figure:
    """Plot solve rate over epochs with an optional scramble depth overlay.

    Expects a ``metrics_df`` with at least an ``epoch`` column and a
    ``solve_rate`` column (values in ``[0, 1]``).  If a ``scramble_depth``
    column is present it is rendered on a secondary y-axis.

    Args:
        metrics_df:   DataFrame of per-epoch metrics.
        solver_name:  Human-readable solver label used in the title.
        output_path:  If given, the figure is saved as PNG to this path.
        interactive:  When *True*, ``plt.show()`` is called.

    Returns:
        The rendered :class:`matplotlib.figure.Figure`.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    x_col = "epoch" if "epoch" in metrics_df.columns else metrics_df.index
    x_vals = metrics_df[x_col] if "epoch" in metrics_df.columns else metrics_df.index

    ax.plot(
        x_vals,
        metrics_df["solve_rate"],
        label="Solve Rate",
        color="seagreen",
        linewidth=2,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Solve Rate")
    ax.set_title(f"{solver_name} — Solve Rate")
    ax.grid(True, alpha=0.3)

    lines2: list = []
    labels2: list = []
    if "scramble_depth" in metrics_df.columns:
        ax2 = ax.twinx()
        ax2.plot(
            x_vals,
            metrics_df["scramble_depth"],
            label="Scramble Depth",
            color="mediumpurple",
            linewidth=1.5,
            linestyle=":",
        )
        ax2.set_ylabel("Scramble Depth", color="mediumpurple")
        ax2.tick_params(axis="y", labelcolor="mediumpurple")
        lines2, labels2 = ax2.get_legend_handles_labels()

    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    fig.tight_layout()
    save_or_show(fig, output_path, interactive)
    return fig


def plot_fitness_curve(
    metrics_df: pd.DataFrame,
    output_path: "Path | None" = None,
    interactive: bool = False,
    stagnation_events: "list[int] | None" = None,
) -> plt.Figure:
    """Plot genetic algorithm best, mean, and control fitness per generation.

    Expects a ``metrics_df`` with at least a ``generation`` (or index) column
    and a ``fitness_best`` column.  Optional columns:

    - ``fitness_mean``: plotted as a dashed line with shaded fill to best.
    - ``control_fitness``: random-individual baseline plotted as a dotted gray line.

    Args:
        metrics_df:        DataFrame of per-generation metrics.
        output_path:       If given, the figure is saved as PNG to this path.
        interactive:       When *True*, ``plt.show()`` is called.
        stagnation_events: Optional list of generation indices where stagnation
                           injection fired; rendered as vertical dashed lines.

    Returns:
        The rendered :class:`matplotlib.figure.Figure`.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    x_col = "generation" if "generation" in metrics_df.columns else None
    x_vals = metrics_df[x_col] if x_col else metrics_df.index

    ax.plot(
        x_vals,
        metrics_df["fitness_best"],
        label="Best Fitness",
        color="crimson",
        linewidth=2,
    )

    if "fitness_mean" in metrics_df.columns and metrics_df["fitness_mean"].notna().any():
        ax.plot(
            x_vals,
            metrics_df["fitness_mean"],
            label="Mean Fitness",
            color="salmon",
            linewidth=1.5,
            linestyle="--",
        )
        ax.fill_between(
            x_vals,
            metrics_df["fitness_mean"],
            metrics_df["fitness_best"],
            alpha=0.2,
            color="salmon",
        )

    if "control_fitness" in metrics_df.columns and metrics_df["control_fitness"].notna().any():
        ax.plot(
            x_vals,
            metrics_df["control_fitness"],
            label="Control (random)",
            color="gray",
            linewidth=1.2,
            linestyle=":",
            alpha=0.8,
        )

    if stagnation_events:
        for i, gen in enumerate(stagnation_events):
            ax.axvline(
                gen,
                color="steelblue",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                label="Stagnation injection" if i == 0 else None,
            )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Genetic Algorithm — Fitness Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_or_show(fig, output_path, interactive)
    return fig


def plot_solution_length_distribution(
    solve_results: list,
    solver_name: str,
    output_path: "Path | None" = None,
    interactive: bool = False,
) -> plt.Figure:
    """Plot a histogram of solution lengths from a batch of solve results.

    Args:
        solve_results: List of :class:`~rubiks_solve.solvers.base.SolveResult`
                       objects.  Only successfully solved results are included
                       in the histogram.
        solver_name:   Human-readable solver label used in the title.
        output_path:   If given, the figure is saved as PNG to this path.
        interactive:   When *True*, ``plt.show()`` is called.

    Returns:
        The rendered :class:`matplotlib.figure.Figure`.
    """
    lengths = [r.move_count for r in solve_results if r.solved]

    fig, ax = plt.subplots(figsize=(8, 5))

    if lengths:
        ax.hist(
            lengths,
            bins=max(1, max(lengths) - min(lengths) + 1),
            color="steelblue",
            edgecolor="black",
            alpha=0.85,
        )
        ax.axvline(
            float(pd.Series(lengths).mean()),
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean = {pd.Series(lengths).mean():.1f}",
        )
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No successful solves",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color="gray",
        )

    n_total = len(solve_results)
    n_solved = len(lengths)
    ax.set_xlabel("Solution Length (moves)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{solver_name} — Solution Length Distribution\n"
        f"(solved {n_solved}/{n_total})"
    )
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_or_show(fig, output_path, interactive)
    return fig
