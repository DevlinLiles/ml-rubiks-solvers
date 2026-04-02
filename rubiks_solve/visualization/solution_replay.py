"""Animated solution replay using matplotlib FuncAnimation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from rubiks_solve.core.base import AbstractPuzzle, Move
from rubiks_solve.visualization.cube_3d import render_cube_unfolded, save_or_show


def animate_solution(
    initial_puzzle: AbstractPuzzle,
    moves: list[Move],
    interval_ms: int = 500,
    output_path: "Path | None" = None,
    interactive: bool = False,
) -> animation.FuncAnimation:
    """Animate a solution sequence step by step.

    Each animation frame shows the cube state *after* applying the next move in
    ``moves``.  Frame 0 shows the initial (scrambled) state before any move is
    applied.

    The animation is saved as a GIF when ``output_path`` ends with ``.gif``.
    Other extensions supported by matplotlib writers (e.g. ``.mp4``) are also
    accepted if the appropriate writer is installed.

    Args:
        initial_puzzle: The scrambled puzzle to animate from.
        moves:          Ordered list of :class:`~rubiks_solve.core.base.Move`
                        objects forming the solution.
        interval_ms:    Delay between frames in milliseconds.
        output_path:    Optional file path to save the animation.  GIF and MP4
                        are the most common formats.  Parent directories are
                        created automatically.
        interactive:    When *True*, ``plt.show()`` is called after setup.

    Returns:
        The :class:`matplotlib.animation.FuncAnimation` object.  Keep a
        reference to prevent garbage collection before saving.
    """
    # Build the list of states: frame 0 is initial, then one per move
    states: list[AbstractPuzzle] = [initial_puzzle]
    current = initial_puzzle
    for move in moves:
        current = current.apply_move(move)
        states.append(current)

    n_frames = len(states)
    move_labels: list[str] = ["Initial"] + [m.name for m in moves]

    # Create figure and initial render
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    def _draw_frame(frame_idx: int) -> list:
        ax.cla()
        puzzle_state = states[frame_idx]
        label = move_labels[frame_idx]

        # Use the unfolded net renderer, drawing into our axes
        _fig_tmp = render_cube_unfolded(
            puzzle_state,
            ax=ax,
            title=f"Move {frame_idx}/{len(moves)}: {label}",
        )
        # render_cube_unfolded returns a figure; we only need the axes content
        del _fig_tmp
        return ax.get_children()

    # Initialise first frame
    _draw_frame(0)

    anim = animation.FuncAnimation(
        fig,
        _draw_frame,
        frames=n_frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".gif":
            writer = animation.PillowWriter(fps=max(1, 1000 // interval_ms))
        elif suffix == ".mp4":
            writer = animation.FFMpegWriter(fps=max(1, 1000 // interval_ms))  # type: ignore[assignment]
        else:
            writer = animation.PillowWriter(fps=max(1, 1000 // interval_ms))
        anim.save(str(output_path), writer=writer)

    if interactive:
        plt.show()

    return anim
