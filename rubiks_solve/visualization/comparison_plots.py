"""Cross-solver comparison plots."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from rubiks_solve.visualization.cube_3d import save_or_show

# Metrics that can be extracted from a list of SolveResult objects.
_METRIC_LABELS: dict[str, str] = {
    "solve_rate": "Solve Rate",
    "mean_moves": "Mean Move Count",
    "mean_time": "Mean Solve Time (s)",
}


def _compute_metric(results: list, metric: str) -> float:
    """Compute a scalar metric from a list of SolveResult objects.

    Args:
        results: List of :class:`~rubiks_solve.solvers.base.SolveResult`.
        metric:  One of ``"solve_rate"``, ``"mean_moves"``, or ``"mean_time"``.

    Returns:
        Scalar float value for the requested metric.

    Raises:
        ValueError: If *metric* is not recognised.
    """
    if not results:
        return 0.0

    if metric == "solve_rate":
        return sum(1 for r in results if r.solved) / len(results)
    if metric == "mean_moves":
        solved = [r.move_count for r in results if r.solved]
        return float(np.mean(solved)) if solved else float("nan")
    if metric == "mean_time":
        return float(np.mean([r.solve_time_seconds for r in results]))
    raise ValueError(
        f"Unknown metric {metric!r}. Choose from {list(_METRIC_LABELS)}."
    )


def plot_solver_comparison(
    results: dict[str, list],
    metric: str = "solve_rate",
    output_path: "Path | None" = None,
    interactive: bool = False,
) -> plt.Figure:
    """Bar chart comparing solvers on a single metric.

    Solvers are sorted descending by solve rate; ties are broken by ascending
    mean move count.

    Args:
        results:      Mapping from solver name to its list of
                      :class:`~rubiks_solve.solvers.base.SolveResult` objects.
        metric:       Which metric to display on the y-axis.  One of
                      ``"solve_rate"``, ``"mean_moves"``, or ``"mean_time"``.
        output_path:  If given, save the figure as PNG to this path.
        interactive:  When *True*, ``plt.show()`` is called.

    Returns:
        The rendered :class:`matplotlib.figure.Figure`.
    """
    if metric not in _METRIC_LABELS:
        raise ValueError(
            f"Unknown metric {metric!r}. Choose from {list(_METRIC_LABELS)}."
        )

    # Compute primary and tiebreak metric per solver
    rows: list[tuple[str, float, float]] = []
    for solver_name, solver_results in results.items():
        primary = _compute_metric(solver_results, metric)
        solve_r = _compute_metric(solver_results, "solve_rate")
        mean_m = _compute_metric(solver_results, "mean_moves")
        rows.append((solver_name, primary, solve_r, mean_m))

    # Sort: descending solve_rate, then ascending mean_moves
    rows.sort(key=lambda x: (-x[2], x[3]))

    solver_names = [r[0] for r in rows]
    values = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(solver_names) * 1.4), 6))

    x = np.arange(len(solver_names))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(solver_names)))  # type: ignore[attr-defined]
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.7)

    # Annotate bars
    for bar, val in zip(bars, values):
        if np.isfinite(val):
            label = f"{val:.2f}" if metric != "solve_rate" else f"{val:.1%}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 * max(values or [1]),
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(solver_names, rotation=20, ha="right")
    ax.set_ylabel(_METRIC_LABELS[metric])
    ax.set_title(f"Solver Comparison — {_METRIC_LABELS[metric]}")
    ax.grid(True, axis="y", alpha=0.3)

    if metric == "solve_rate":
        ax.set_ylim(0, 1.1)

    fig.tight_layout()
    save_or_show(fig, output_path, interactive)
    return fig


def plot_performance_over_scramble_depth(
    results: dict[str, dict[int, list]],
    metric: str = "solve_rate",
    output_path: "Path | None" = None,
    interactive: bool = False,
) -> plt.Figure:
    """Line chart of solver performance vs. scramble depth.

    One line per solver; the x-axis shows scramble depth values sorted
    numerically.

    Args:
        results:      Nested mapping ``solver_name -> depth -> [SolveResult]``.
        metric:       Which metric to plot.  One of ``"solve_rate"``,
                      ``"mean_moves"``, or ``"mean_time"``.
        output_path:  If given, save the figure as PNG to this path.
        interactive:  When *True*, ``plt.show()`` is called.

    Returns:
        The rendered :class:`matplotlib.figure.Figure`.
    """
    if metric not in _METRIC_LABELS:
        raise ValueError(
            f"Unknown metric {metric!r}. Choose from {list(_METRIC_LABELS)}."
        )

    # Collect all depths across solvers
    all_depths: list[int] = sorted(
        {depth for solver_results in results.values() for depth in solver_results}
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = [c["color"] for c in prop_cycle]

    for i, (solver_name, depth_map) in enumerate(results.items()):
        color = colors[i % len(colors)]
        y_vals: list[float] = []
        x_vals: list[int] = []

        for depth in all_depths:
            if depth in depth_map:
                val = _compute_metric(depth_map[depth], metric)
                x_vals.append(depth)
                y_vals.append(val)

        ax.plot(
            x_vals,
            y_vals,
            marker="o",
            label=solver_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Scramble Depth")
    ax.set_ylabel(_METRIC_LABELS[metric])
    ax.set_title(f"Performance vs Scramble Depth — {_METRIC_LABELS[metric]}")

    if all_depths:
        ax.set_xticks(all_depths)

    if metric == "solve_rate":
        ax.set_ylim(0, 1.05)

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_or_show(fig, output_path, interactive)
    return fig
