"""3-D and 2-D cube rendering via matplotlib."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches  # pylint: disable=consider-using-from-import
import numpy as np

# Lazy import: mpl_toolkits is bundled with matplotlib but the 3-D axes are
# registered as a side-effect of the import.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # pylint: disable=unused-import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from rubiks_solve.core.base import AbstractPuzzle

# ---------------------------------------------------------------------------
# Color map: face index → matplotlib color string
# Face indices follow CubeNNN convention:
#   0=U (white), 1=D (yellow), 2=F (green), 3=B (blue), 4=L (orange), 5=R (red)
# ---------------------------------------------------------------------------
CUBE_COLORS: dict[int, str] = {
    0: "white",   # U
    1: "yellow",  # D
    2: "green",   # F
    3: "blue",    # B
    4: "orange",  # L
    5: "red",     # R
}

# Megaminx colors (12 faces)
MEGAMINX_COLORS: dict[int, str] = {
    0: "white",
    1: "yellow",
    2: "green",
    3: "blue",
    4: "orange",
    5: "red",
    6: "purple",
    7: "pink",
    8: "gray",
    9: "lightblue",
    10: "lime",
    11: "beige",
}


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _face_quads_3d(
    state_2d: np.ndarray,
    origin: np.ndarray,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    color_map: dict[int, str],
) -> list[tuple[list[list[float]], str]]:
    """Build a list of (vertices, color) quads for a single cube face.

    The face stickers are laid out on the plane defined by ``origin``,
    ``u_vec`` (row direction), and ``v_vec`` (column direction).

    Args:
        state_2d:   2-D array of color indices, shape ``(n, n)``.
        origin:     3-D point of the top-left corner of the face.
        u_vec:      Unit vector along rows (pointing down the face).
        v_vec:      Unit vector along columns (pointing right across the face).
        color_map:  Mapping from color index to matplotlib color string.

    Returns:
        List of ``(quad_vertices, color_string)`` tuples.
    """
    n = state_2d.shape[0]
    cell_size = 1.0 / n
    quads: list[tuple[list[list[float]], str]] = []

    for row in range(n):
        for col in range(n):
            color_idx = int(state_2d[row, col])
            color = color_map.get(color_idx, "gray")

            # Four corners of the sticker (slightly inset)
            pad = 0.02 * cell_size
            r0 = row * cell_size + pad
            r1 = (row + 1) * cell_size - pad
            c0 = col * cell_size + pad
            c1 = (col + 1) * cell_size - pad

            corners = [
                origin + r0 * u_vec + c0 * v_vec,
                origin + r0 * u_vec + c1 * v_vec,
                origin + r1 * u_vec + c1 * v_vec,
                origin + r1 * u_vec + c0 * v_vec,
            ]
            verts = [[pt.tolist() for pt in corners]]
            quads.append((verts, color))

    return quads


def render_cube_3d(
    puzzle: AbstractPuzzle,
    ax: "Axes3D | None" = None,
    title: str = "",
) -> plt.Figure:
    """Render a 3-D view of an NxN cube using ``Poly3DCollection``.

    Each of the six visible faces is drawn as a grid of colored squares on the
    surface of the unit cube.  The viewing angle is set to a comfortable
    isometric perspective.

    Args:
        puzzle: An ``AbstractPuzzle`` instance whose ``state`` has shape
                ``(6, n, n)``.
        ax:     Existing 3-D axes to draw on.  When *None* a new figure is
                created automatically.
        title:  Optional figure title.

    Returns:
        The :class:`matplotlib.figure.Figure` containing the rendered cube.
    """
    state = puzzle.state  # shape (6, n, n)
    if state.ndim != 3 or state.shape[0] != 6:
        raise ValueError(
            f"render_cube_3d expects state shape (6, n, n), got {state.shape}"
        )

    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # Face definitions: (face_index, origin, u_vec, v_vec)
    # The unit cube spans [0,1]^3.  We map:
    #   U (top)    → z=1 plane,  rows toward -y, cols toward +x
    #   D (bottom) → z=0 plane,  rows toward +y, cols toward +x
    #   F (front)  → y=0 plane,  rows toward -z, cols toward +x
    #   B (back)   → y=1 plane,  rows toward -z, cols toward -x
    #   L (left)   → x=0 plane,  rows toward -z, cols toward +y
    #   R (right)  → x=1 plane,  rows toward -z, cols toward -y
    o = np.array
    face_defs: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = [
        (0, o([0.0, 1.0, 1.0]), o([0.0, -1.0, 0.0]), o([1.0, 0.0, 0.0])),  # U
        (1, o([0.0, 0.0, 0.0]), o([0.0, 1.0, 0.0]),  o([1.0, 0.0, 0.0])),  # D
        (2, o([0.0, 0.0, 1.0]), o([0.0, 0.0, -1.0]), o([1.0, 0.0, 0.0])),  # F
        (3, o([1.0, 1.0, 1.0]), o([0.0, 0.0, -1.0]), o([-1.0, 0.0, 0.0])), # B
        (4, o([0.0, 0.0, 1.0]), o([0.0, 0.0, -1.0]), o([0.0, 1.0, 0.0])),  # L
        (5, o([1.0, 1.0, 1.0]), o([0.0, 0.0, -1.0]), o([0.0, -1.0, 0.0])), # R
    ]

    all_verts: list[list[list[float]]] = []
    all_colors: list[str] = []

    for face_idx, origin, u_vec, v_vec in face_defs:
        face_state = state[face_idx]
        quads = _face_quads_3d(face_state, origin, u_vec, v_vec, CUBE_COLORS)
        for verts, color in quads:
            all_verts.extend(verts)
            all_colors.append(color)

    collection = Poly3DCollection(
        all_verts,
        facecolors=all_colors,
        edgecolors="black",
        linewidths=0.5,
        alpha=1.0,
    )
    ax.add_collection3d(collection)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_box_aspect([1, 1, 1])  # type: ignore[arg-type]
    ax.set_axis_off()
    ax.view_init(elev=25, azim=35)

    if title:
        ax.set_title(title, pad=10)

    fig.tight_layout()
    return fig


def render_cube_unfolded(
    puzzle: AbstractPuzzle,
    ax: "plt.Axes | None" = None,
    title: str = "",
) -> plt.Figure:
    """Render the cube as a 2-D unfolded cross (standard cube net layout).

    The net uses the standard layout::

             [U]
        [L] [F] [R] [B]
             [D]

    Each sticker cell is colored according to its current color index.

    Args:
        puzzle: An ``AbstractPuzzle`` with ``state`` shape ``(6, n, n)``.
        ax:     Existing 2-D axes.  When *None* a new figure is created.
        title:  Optional figure title.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    state = puzzle.state  # (6, n, n)
    if state.ndim != 3 or state.shape[0] != 6:
        raise ValueError(
            f"render_cube_unfolded expects state shape (6, n, n), got {state.shape}"
        )
    n = state.shape[1]

    # Net grid: 4 columns × 3 rows (each cell is n×n stickers)
    # Column/row offsets in sticker units:
    #   U at col=1, row=0
    #   L at col=0, row=1 ; F at col=1, row=1 ; R at col=2, row=1 ; B at col=3, row=1
    #   D at col=1, row=2
    face_positions: list[tuple[int, int, int]] = [
        (0, 1, 0),  # U: face_idx, col_offset, row_offset
        (4, 0, 1),  # L
        (2, 1, 1),  # F
        (5, 2, 1),  # R
        (3, 3, 1),  # B
        (1, 1, 2),  # D
    ]

    total_cols = 4 * n
    total_rows = 3 * n
    cell_size = 0.9  # each sticker is 0.9 units wide (gap = 0.1)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(total_cols * 0.6, total_rows * 0.6)
        )
    else:
        fig = ax.get_figure()

    ax.set_xlim(0, total_cols)
    ax.set_ylim(0, total_rows)
    ax.set_aspect("equal")
    ax.axis("off")

    for face_idx, col_off, row_off in face_positions:
        face_state = state[face_idx]
        x_base = col_off * n
        # Flip y so row 0 is at top of net
        y_base = total_rows - (row_off + 1) * n

        for row in range(n):
            for col in range(n):
                color_idx = int(face_state[row, col])
                color = CUBE_COLORS.get(color_idx, "gray")
                # y position: row 0 at top → highest y value
                x = x_base + col + (1 - cell_size) / 2
                y = y_base + (n - 1 - row) + (1 - cell_size) / 2
                rect = patches.Rectangle(
                    (x, y),
                    cell_size,
                    cell_size,
                    linewidth=0.5,
                    edgecolor="black",
                    facecolor=color,
                )
                ax.add_patch(rect)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def render_megaminx(
    puzzle: AbstractPuzzle,
    ax: "plt.Axes | None" = None,
    title: str = "",
) -> plt.Figure:
    """Render a Megaminx as a 2-D pentagon layout showing all 12 faces.

    Each face is represented as a regular pentagon.  The 12 pentagons are
    arranged in two rings of 5 surrounding a top and bottom center face,
    matching the standard Megaminx net.

    Args:
        puzzle: A Megaminx puzzle instance with ``state`` shape ``(12, 11)``.
        ax:     Existing 2-D axes.  When *None* a new figure is created.
        title:  Optional figure title.

    Returns:
        The :class:`matplotlib.figure.Figure`.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    ax.set_aspect("equal")
    ax.axis("off")

    state = puzzle.state  # (12, 11) — each face has 11 sticker slots
    n_faces = state.shape[0]

    # Compute dominant color per face (mode of sticker colors on that face)
    face_colors: list[str] = []
    for fi in range(n_faces):
        face_stickers = state[fi]
        counts = np.bincount(face_stickers.astype(int), minlength=n_faces)
        dominant_idx = int(counts.argmax())
        face_colors.append(MEGAMINX_COLORS.get(dominant_idx, "gray"))

    # Layout: center top (face 0), ring of 5 around top (faces 1-5),
    # ring of 5 around bottom (faces 6-10), center bottom (face 11)
    radius_inner = 1.5
    radius_outer = 3.5
    pentagon_radius = 0.7

    # Center positions for each face
    centers: list[tuple[float, float]] = []

    # Top center
    centers.append((0.0, radius_outer * 0.6))

    # Inner ring (5 faces)
    for i in range(5):
        angle = np.radians(90 + i * 72)
        cx = radius_inner * np.cos(angle)
        cy = radius_inner * np.sin(angle)
        centers.append((cx, cy))

    # Outer ring (5 faces)
    for i in range(5):
        angle = np.radians(90 + 36 + i * 72)
        cx = radius_outer * np.cos(angle)
        cy = radius_outer * np.sin(angle)
        centers.append((cx, cy))

    # Bottom center
    centers.append((0.0, -radius_outer * 0.6))

    all_x = [c[0] for c in centers]
    all_y = [c[1] for c in centers]
    margin = pentagon_radius * 2
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    for fi in range(min(n_faces, len(centers))):
        cx, cy = centers[fi]
        color = face_colors[fi] if fi < len(face_colors) else "gray"
        pentagon = patches.RegularPolygon(
            (cx, cy),
            numVertices=5,
            radius=pentagon_radius,
            orientation=np.radians(18),
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.add_patch(pentagon)
        ax.text(
            cx,
            cy,
            str(fi),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="black" if color not in {"black", "blue", "purple"} else "white",
        )

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig


def save_or_show(
    fig: plt.Figure,
    path: "Path | None" = None,
    interactive: bool = False,
) -> None:
    """Save a figure to disk and/or display it interactively.

    Args:
        fig:         The matplotlib figure to save or show.
        path:        Destination file path.  The format is inferred from the
                     extension (PNG, PDF, SVG, etc.).  Parent directories are
                     created automatically.  When *None*, nothing is saved.
        interactive: When *True*, ``plt.show()`` is called after saving.
    """
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if interactive:
        plt.show()
    elif path is not None:
        plt.close(fig)
