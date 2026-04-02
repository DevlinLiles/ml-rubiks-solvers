"""CLI: render a cube state or animate a solution sequence.

Usage examples::

    rubiks-viz --puzzle 3x3 --state solved --output cube.png
    rubiks-viz --puzzle 3x3 --state scrambled --scramble-depth 15 --output scrambled.png
    rubiks-viz --puzzle 3x3 --replay solution.json --output solution.gif --interactive
    rubiks-viz --puzzle 3x3 --state solved --mode 3d --output cube_3d.png

Solution JSON format (``--replay``)::

    {
        "scramble": "R U R' U'",          // or omit for solved initial state
        "moves": ["R", "U", "R'", "U'"]  // solution move sequence
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the visualize script.

    Returns:
        Populated :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        prog="rubiks-viz",
        description="Render cube states or animate solution sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--puzzle",
        choices=["2x2", "3x3", "4x4", "5x5"],
        default="3x3",
        help="Puzzle type to render.",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--state",
        choices=["solved", "scrambled"],
        metavar="STATE",
        help="Render a static cube state.  'solved' shows the goal state; "
             "'scrambled' applies --scramble-depth random moves.",
    )
    mode_group.add_argument(
        "--replay",
        type=Path,
        metavar="FILE",
        help="JSON file containing 'scramble' and 'moves' keys.  "
             "Animates the solution sequence.  Saves a GIF when --output ends in .gif.",
    )

    parser.add_argument(
        "--scramble-depth",
        type=int,
        default=10,
        metavar="N",
        help="Number of random moves for --state scrambled.",
    )
    parser.add_argument(
        "--mode",
        choices=["unfolded", "3d"],
        default="unfolded",
        help="Rendering style for static states.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Save rendered image or animation to this path.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Display the rendered figure interactively (blocks until closed).",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=500,
        metavar="MS",
        help="Frame interval in milliseconds for GIF animations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for scramble generation.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Optional title for the rendered figure.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_puzzle_cls(puzzle_name: str) -> type:
    """Return the concrete puzzle class for a given name.

    Args:
        puzzle_name: One of ``"2x2"``, ``"3x3"``, ``"4x4"``, ``"5x5"``.

    Returns:
        Concrete puzzle class.
    """
    from rubiks_solve.core import Cube2x2, Cube3x3, Cube4x4, Cube5x5

    registry = {"2x2": Cube2x2, "3x3": Cube3x3, "4x4": Cube4x4, "5x5": Cube5x5}
    return registry[puzzle_name]


def _resolve_move(name: str, move_map: dict) -> object:
    """Look up a move by name in a legal-move dict.

    Args:
        name:     Move name string, e.g. ``"R'"``.
        move_map: Dict mapping name → Move built from ``puzzle.legal_moves()``.

    Returns:
        The :class:`~rubiks_solve.core.base.Move` object.

    Raises:
        SystemExit: If *name* is not in *move_map*.
    """
    if name not in move_map:
        print(
            f"Error: move '{name}' not recognised.\n"
            f"Legal moves: {', '.join(sorted(move_map))}",
            file=sys.stderr,
        )
        sys.exit(1)
    return move_map[name]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to static render or animation."""
    args = parse_args()

    from rubiks_solve.utils.rng import make_rng

    puzzle_cls = _load_puzzle_cls(args.puzzle)
    rng = make_rng(args.seed)

    # ------------------------------------------------------------------ #
    # Static state rendering
    # ------------------------------------------------------------------ #
    if args.state is not None:
        if args.state == "solved":
            puzzle = puzzle_cls.solved_state()
            title = args.title or f"{args.puzzle} — Solved State"
        else:  # scrambled
            puzzle = puzzle_cls.solved_state().scramble(args.scramble_depth, rng)
            title = args.title or f"{args.puzzle} — Scrambled ({args.scramble_depth} moves)"

        from rubiks_solve.visualization.cube_3d import (
            render_cube_3d,
            render_cube_unfolded,
            save_or_show,
        )

        if args.mode == "3d":
            fig = render_cube_3d(puzzle, title=title)
        else:
            fig = render_cube_unfolded(puzzle, title=title)

        if args.output is None and not args.interactive:
            print(
                "Tip: pass --output <file.png> to save, or --interactive to display.",
                file=sys.stderr,
            )

        save_or_show(fig, args.output, interactive=args.interactive)

        if args.output:
            print(f"Saved: {args.output}")

    # ------------------------------------------------------------------ #
    # Solution replay animation
    # ------------------------------------------------------------------ #
    elif args.replay is not None:
        replay_path = Path(args.replay)
        if not replay_path.exists():
            print(f"Error: replay file not found: {replay_path}", file=sys.stderr)
            sys.exit(1)

        try:
            replay_data: dict = json.loads(replay_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"Error: failed to parse replay JSON: {exc}", file=sys.stderr)
            sys.exit(1)

        base_puzzle = puzzle_cls.solved_state()
        move_map = {m.name: m for m in base_puzzle.legal_moves()}

        # Apply scramble if present
        scramble_str: str = replay_data.get("scramble", "")
        if scramble_str.strip():
            scramble_moves = [
                _resolve_move(name, move_map)
                for name in scramble_str.split()
            ]
            initial_puzzle = base_puzzle.apply_moves(scramble_moves)
        else:
            initial_puzzle = base_puzzle

        # Parse solution moves
        raw_moves: list[str] = replay_data.get("moves", [])
        solution_moves = [_resolve_move(name, move_map) for name in raw_moves]

        from rubiks_solve.visualization.solution_replay import animate_solution

        title = args.title or f"{args.puzzle} — Solution Replay ({len(solution_moves)} moves)"

        anim = animate_solution(
            initial_puzzle=initial_puzzle,
            moves=solution_moves,
            interval_ms=args.interval_ms,
            output_path=args.output,
            interactive=args.interactive,
        )

        if args.output:
            print(f"Saved: {args.output}")

        # Keep a reference to prevent GC before show()
        _ = anim

    else:
        print("Error: specify --state or --replay.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
