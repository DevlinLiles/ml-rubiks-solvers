"""CLI: solve a scrambled puzzle using a chosen algorithm.

Usage examples::

    rubiks-solve --puzzle 3x3 --scramble "R U R' U'" --solver genetic
    rubiks-solve --puzzle 3x3 --random-scramble 20 --solver mcts
    rubiks-solve --puzzle 2x2 --random-scramble 10 --solver genetic --visualize
    rubiks-solve --puzzle 3x3 --random-scramble 15 --solver cnn --model models/cnn/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def _puzzle_registry() -> dict[str, type]:
    from rubiks_solve.core import Cube2x2, Cube3x3, Cube4x4, Cube5x5

    return {"2x2": Cube2x2, "3x3": Cube3x3, "4x4": Cube4x4, "5x5": Cube5x5}


def _parse_move_sequence(sequence: str, puzzle) -> list:
    """Parse a whitespace-separated move string into a list of Move objects.

    Args:
        sequence: Space-separated move names, e.g. ``"R U R' U'"``.
        puzzle:   A puzzle instance whose ``legal_moves()`` is used as the
                  lookup table.

    Returns:
        List of :class:`~rubiks_solve.core.base.Move` objects.

    Raises:
        SystemExit: If any move name is not found in the legal move set.
    """
    move_map = {m.name: m for m in puzzle.legal_moves()}
    names = sequence.split()
    moves = []
    for name in names:
        if name not in move_map:
            print(
                f"Error: move '{name}' not recognised for {puzzle.puzzle_name()}.\n"
                f"Legal moves: {', '.join(sorted(move_map))}",
                file=sys.stderr,
            )
            sys.exit(1)
        moves.append(move_map[name])
    return moves


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the solve script.

    Returns:
        Populated :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="rubiks-solve",
        description="Solve a scrambled Rubik's cube puzzle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--puzzle",
        choices=["2x2", "3x3", "4x4", "5x5"],
        default="3x3",
        help="Puzzle type to solve.",
    )

    scramble_group = parser.add_mutually_exclusive_group(required=True)
    scramble_group.add_argument(
        "--scramble",
        type=str,
        metavar="MOVES",
        help='Space-separated move sequence to scramble from solved, e.g. "R U R\' U\'".',
    )
    scramble_group.add_argument(
        "--random-scramble",
        type=int,
        metavar="N",
        help="Apply N random moves to a solved state.",
    )

    parser.add_argument(
        "--solver",
        choices=["genetic", "cnn", "mcts"],
        default="genetic",
        help="Solver algorithm to use.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        metavar="DIR",
        help="Path to a trained model directory (required for --solver cnn).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random scramble and solver.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show before/after cube visualizations using matplotlib.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save before/after visualization to this PNG file path.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------


def _build_solver(args: argparse.Namespace, puzzle_cls: type):
    """Instantiate the requested solver.

    Args:
        args:       Parsed CLI arguments.
        puzzle_cls: Concrete puzzle class.

    Returns:
        An :class:`~rubiks_solve.solvers.base.AbstractSolver` instance.
    """
    if args.solver == "genetic":
        from rubiks_solve.solvers.genetic.solver import GeneticConfig, GeneticSolver

        cfg = GeneticConfig(
            seed=args.seed,
            max_chromosome_length=puzzle_cls.move_limit(),
        )
        return GeneticSolver(puzzle_cls, cfg)

    if args.solver == "mcts":
        from rubiks_solve.solvers.mcts.solver import MCTSConfig, MCTSSolver

        cfg = MCTSConfig(seed=args.seed)
        return MCTSSolver(puzzle_cls, cfg)

    if args.solver == "cnn":
        from rubiks_solve.solvers.cnn.solver import CNNConfig, CNNSolver
        from rubiks_solve.encoding.registry import get_encoder

        model_path: Path | None = args.model
        if model_path is not None:
            # Find latest checkpoint
            checkpoints = sorted(Path(model_path).glob("ckpt_epoch_*.npz"))
            if checkpoints:
                model_path = checkpoints[-1]

        cfg = CNNConfig(model_path=model_path)
        encoder = get_encoder("one_hot", puzzle_cls)
        return CNNSolver(puzzle_cls, encoder, cfg)

    print(f"Unknown solver: {args.solver}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, scramble or load a puzzle, run the solver, and print results."""
    args = parse_args()

    from rubiks_solve.utils.logging_config import configure_logging, get_logger
    from rubiks_solve.utils.rng import make_rng

    configure_logging(level=args.log_level)
    logger = get_logger(__name__)

    puzzle_registry = _puzzle_registry()
    puzzle_cls = puzzle_registry[args.puzzle]
    rng = make_rng(args.seed)

    # Build the scrambled puzzle
    solved = puzzle_cls.solved_state()
    if args.scramble:
        moves_applied = _parse_move_sequence(args.scramble, solved)
        scrambled = solved.apply_moves(moves_applied)
        scramble_desc = args.scramble
    else:
        scrambled = solved.scramble(args.random_scramble, rng)
        scramble_desc = f"random {args.random_scramble} moves"

    logger.info(
        "Puzzle scrambled",
        puzzle=args.puzzle,
        scramble=scramble_desc,
        solver=args.solver,
    )

    if scrambled.is_solved:
        print("The puzzle is already solved — no moves needed.")
        return

    solver = _build_solver(args, puzzle_cls)

    from rubiks_solve.utils.timer import timer

    with timer() as t:
        result = solver.solve(scrambled)

    # ---- Print results ----
    print()
    print(f"Puzzle  : {args.puzzle}")
    print(f"Scramble: {scramble_desc}")
    print(f"Solver  : {args.solver}")
    print(f"Solved  : {result.solved}")
    print(f"Moves   : {result.move_count}")
    print(f"Time    : {t.elapsed_seconds:.3f}s")
    if result.moves:
        print(f"Sequence: {' '.join(m.name for m in result.moves)}")

    if not result.solved:
        print(
            "\nWarning: solver did not find a complete solution within its budget.",
            file=sys.stderr,
        )

    # ---- Optional visualization ----
    if args.visualize or args.output:
        try:
            import matplotlib.pyplot as plt
            from rubiks_solve.visualization.cube_3d import (
                render_cube_unfolded,
                save_or_show,
            )

            fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 5))
            render_cube_unfolded(scrambled, ax=ax_before, title="Before (Scrambled)")

            final_state = scrambled.apply_moves(result.moves) if result.moves else scrambled
            render_cube_unfolded(
                final_state,
                ax=ax_after,
                title="After (Solver Output)" + (" ✓" if result.solved else " ✗"),
            )

            fig.suptitle(
                f"{args.puzzle} — {args.solver} solver — {result.move_count} moves",
                fontsize=13,
            )
            fig.tight_layout()

            save_or_show(fig, args.output, interactive=args.visualize)
        except ImportError as exc:
            logger.warning("Could not render visualization", error=str(exc))


if __name__ == "__main__":
    main()
