"""CLI: benchmark all solvers on all puzzle types.

Usage examples::

    rubiks-benchmark --time-budget 30 --n-puzzles 100 --output results/
    rubiks-benchmark --puzzles 2x2 3x3 --solvers genetic mcts --n-puzzles 50
    rubiks-benchmark --config config.json --output results/

Output:

    results/benchmark_results.csv    — per-run tabular data
    results/comparison_plot.png      — grouped bar chart
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _all_puzzle_classes() -> dict[str, type]:
    from rubiks_solve.core import Cube2x2, Cube3x3, Cube4x4, Cube5x5

    return {"2x2": Cube2x2, "3x3": Cube3x3, "4x4": Cube4x4, "5x5": Cube5x5}


def _build_solver(solver_name: str, puzzle_cls: type, time_budget: float, seed: int):
    """Instantiate a solver for a given puzzle class.

    Args:
        solver_name: One of ``"genetic"``, ``"mcts"``.
        puzzle_cls:  Concrete puzzle class.
        time_budget: Per-puzzle time budget in seconds (used by MCTS).
        seed:        Random seed.

    Returns:
        An :class:`~rubiks_solve.solvers.base.AbstractSolver` instance.
    """
    if solver_name == "genetic":
        from rubiks_solve.solvers.genetic.solver import GeneticConfig, GeneticSolver

        cfg = GeneticConfig(
            seed=seed,
            max_chromosome_length=puzzle_cls.move_limit(),
        )
        return GeneticSolver(puzzle_cls, cfg)

    if solver_name == "mcts":
        from rubiks_solve.solvers.mcts.solver import MCTSConfig, MCTSSolver

        cfg = MCTSConfig(seed=seed, time_limit_seconds=time_budget)
        return MCTSSolver(puzzle_cls, cfg)

    else:
        raise ValueError(f"Unsupported solver for benchmarking: {solver_name!r}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark script.

    Returns:
        Populated :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="rubiks-benchmark",
        description="Benchmark solver algorithms on Rubik's cube puzzle types.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--puzzles",
        nargs="+",
        choices=["2x2", "3x3", "4x4", "5x5"],
        default=["3x3"],
        metavar="PUZZLE",
        help="Puzzle type(s) to benchmark.",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=["genetic", "mcts"],
        default=["genetic", "mcts"],
        metavar="SOLVER",
        help="Solver algorithm(s) to benchmark.",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="Wall-clock time budget per puzzle per solver.",
    )
    parser.add_argument(
        "--n-puzzles",
        type=int,
        default=100,
        help="Number of scrambled puzzles to benchmark per configuration.",
    )
    parser.add_argument(
        "--scramble-depths",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        metavar="DEPTH",
        help="Scramble depths to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        metavar="DIR",
        help="Output directory for CSV and plots.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help="Optional AppConfig JSON/YAML file.  CLI flags override config values.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_benchmark(
    puzzle_name: str,
    puzzle_cls: type,
    solver_name: str,
    scramble_depth: int,
    n_puzzles: int,
    time_budget: float,
    seed: int,
    logger: object,
) -> list[dict]:
    """Benchmark a single (puzzle, solver, depth) combination.

    Generates ``n_puzzles`` scrambled states at ``scramble_depth`` and runs the
    solver on each, respecting ``time_budget`` per puzzle (best-effort via the
    solver's own config; the outer loop does not enforce the budget independently).

    Args:
        puzzle_name:    Human-readable puzzle name for logging.
        puzzle_cls:     Concrete puzzle class.
        solver_name:    Solver identifier.
        scramble_depth: Number of random moves used to scramble each puzzle.
        n_puzzles:      How many puzzles to attempt.
        time_budget:    Per-puzzle time budget (seconds).
        seed:           Base seed; each puzzle uses ``seed + i`` for variety.
        logger:         structlog logger.

    Returns:
        List of result dicts, one per puzzle attempt.
    """
    from rubiks_solve.utils.rng import make_rng
    from rubiks_solve.utils.timer import timer

    solver = _build_solver(solver_name, puzzle_cls, time_budget, seed)
    rng = make_rng(seed)
    solved_base = puzzle_cls.solved_state()

    rows: list[dict] = []
    for i in range(n_puzzles):
        puzzle = solved_base.scramble(scramble_depth, rng)
        with timer() as _t:
            result = solver.solve(puzzle)

        rows.append(
            {
                "puzzle": puzzle_name,
                "solver": solver_name,
                "scramble_depth": scramble_depth,
                "run_idx": i,
                "solved": result.solved,
                "move_count": result.move_count,
                "solve_time_s": result.solve_time_seconds,
                "iterations": result.iterations,
            }
        )

    n_solved = sum(1 for r in rows if r["solved"])
    logger.info(  # type: ignore[union-attr]
        "Benchmark run complete",
        puzzle=puzzle_name,
        solver=solver_name,
        depth=scramble_depth,
        solve_rate=f"{n_solved}/{n_puzzles}",
    )
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full benchmark suite, emit CSV and a comparison bar chart."""
    args = parse_args()

    from rubiks_solve.utils.logging_config import configure_logging, get_logger

    configure_logging(level=args.log_level)
    logger = get_logger(__name__)

    # Apply AppConfig overrides
    if args.config is not None:
        from rubiks_solve.utils.config import AppConfig

        app_cfg = AppConfig.from_file(args.config)
        bm = app_cfg.benchmark
        if args.time_budget == 30.0:
            args.time_budget = bm.time_budget_seconds
        if args.n_puzzles == 100:
            args.n_puzzles = bm.n_puzzles
        if args.scramble_depths == [5, 10, 15, 20]:
            args.scramble_depths = bm.scramble_depths

    args.output.mkdir(parents=True, exist_ok=True)

    puzzle_registry = _all_puzzle_classes()
    all_rows: list[dict] = []

    total_runs = len(args.puzzles) * len(args.solvers) * len(args.scramble_depths)
    run_idx = 0

    for puzzle_name in args.puzzles:
        puzzle_cls = puzzle_registry[puzzle_name]
        for solver_name in args.solvers:
            for depth in args.scramble_depths:
                run_idx += 1
                logger.info(
                    f"Run {run_idx}/{total_runs}",
                    puzzle=puzzle_name,
                    solver=solver_name,
                    depth=depth,
                )
                try:
                    rows = _run_benchmark(
                        puzzle_name=puzzle_name,
                        puzzle_cls=puzzle_cls,
                        solver_name=solver_name,
                        scramble_depth=depth,
                        n_puzzles=args.n_puzzles,
                        time_budget=args.time_budget,
                        seed=args.seed,
                        logger=logger,
                    )
                    all_rows.extend(rows)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Benchmark run failed",
                        puzzle=puzzle_name,
                        solver=solver_name,
                        depth=depth,
                        error=str(exc),
                    )

    if not all_rows:
        logger.error("No benchmark results collected — nothing to save.")
        sys.exit(1)

    # ---- Save CSV ----
    csv_path = args.output / "benchmark_results.csv"
    fieldnames = list(all_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("Benchmark CSV saved", path=str(csv_path))

    # ---- Comparison plots ----
    try:
        import pandas as pd
        from rubiks_solve.visualization.comparison_plots import (
            plot_solver_comparison,
            plot_performance_over_scramble_depth,
        )
        from rubiks_solve.solvers.base import SolveResult

        df = pd.DataFrame(all_rows)

        # Build solver_name -> [SolveResult] for solve rate bar chart
        solver_results: dict[str, list] = {}
        for (pname, sname), grp in df.groupby(["puzzle", "solver"]):
            key = f"{sname} ({pname})"
            solver_results[key] = [
                SolveResult(
                    solved=bool(row["solved"]),
                    moves=[],  # moves not tracked in aggregate
                    solve_time_seconds=float(row["solve_time_s"]),
                    iterations=int(row["iterations"]),
                )
                for _, row in grp.iterrows()
            ]

        bar_path = args.output / "comparison_plot.png"
        plot_solver_comparison(solver_results, metric="solve_rate", output_path=bar_path)
        logger.info("Comparison bar chart saved", path=str(bar_path))

        # Build solver_name -> depth -> [SolveResult] for depth-vs-performance chart
        depth_results: dict[str, dict[int, list]] = {}
        for (pname, sname, depth), grp in df.groupby(["puzzle", "solver", "scramble_depth"]):
            key = f"{sname} ({pname})"
            depth_results.setdefault(key, {})[int(depth)] = [
                SolveResult(
                    solved=bool(row["solved"]),
                    moves=[],
                    solve_time_seconds=float(row["solve_time_s"]),
                    iterations=int(row["iterations"]),
                )
                for _, row in grp.iterrows()
            ]

        depth_plot_path = args.output / "depth_performance_plot.png"
        plot_performance_over_scramble_depth(
            depth_results, metric="solve_rate", output_path=depth_plot_path
        )
        logger.info("Depth performance chart saved", path=str(depth_plot_path))

    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not generate comparison plots", error=str(exc))

    logger.info("Benchmark complete", output=str(args.output))


if __name__ == "__main__":
    main()
