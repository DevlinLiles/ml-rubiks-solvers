"""rubiks_solve.visualization — matplotlib-based rendering for puzzles and metrics."""
from rubiks_solve.visualization.cube_3d import (
    CUBE_COLORS,
    render_cube_3d,
    render_cube_unfolded,
    render_megaminx,
    save_or_show,
)
from rubiks_solve.visualization.training_plots import (
    plot_fitness_curve,
    plot_loss_curve,
    plot_solve_rate,
    plot_solution_length_distribution,
)
from rubiks_solve.visualization.comparison_plots import (
    plot_solver_comparison,
    plot_performance_over_scramble_depth,
)
from rubiks_solve.visualization.solution_replay import animate_solution

__all__ = [
    # cube_3d
    "CUBE_COLORS",
    "render_cube_3d",
    "render_cube_unfolded",
    "render_megaminx",
    "save_or_show",
    # training_plots
    "plot_loss_curve",
    "plot_solve_rate",
    "plot_fitness_curve",
    "plot_solution_length_distribution",
    # comparison_plots
    "plot_solver_comparison",
    "plot_performance_over_scramble_depth",
    # solution_replay
    "animate_solution",
]
