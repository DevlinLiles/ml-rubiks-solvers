/**
 * colors.js — Shared color palettes and puzzle display constants
 *
 * Loaded before all puzzle renderers. Exposes globals used by:
 *   cube3d.js, megaminx3d.js, skewb3d.js, solve-controller.js
 */

// ---------------------------------------------------------------------------
// Megaminx (12-face dodecahedron)
// ---------------------------------------------------------------------------

// One color per face index 0-11, index 12 = inner/dark
const MEGA_COLORS = [
  '#FFFFFF', // 0  U
  '#FFD500', // 1
  '#009B48', // 2
  '#0046AD', // 3
  '#FF5800', // 4
  '#C41E3A', // 5
  '#A020F0', // 6
  '#FF69B4', // 7
  '#00CED1', // 8
  '#8B4513', // 9
  '#7CFC00', // 10
  '#FF8C00', // 11
  '#1a1a1a', // 12 inner
];

const MEGA_FACE_NAMES = ['U', 'BL', 'BR', 'L', 'R', 'DL', 'DR', 'DBL', 'DBR', 'DB', 'F', 'DF'];

// ---------------------------------------------------------------------------
// Skewb Ultimate (cube-shaped, 8 corner pieces + 6 face pieces)
// ---------------------------------------------------------------------------

// One color per corner piece (8 corners, each has a unique "home" color)
const SKEWB_CORNER_COLORS = [
  '#FFFFFF', // 0  LBD
  '#FFD500', // 1  RBD
  '#009B48', // 2  LFD
  '#0046AD', // 3  RFD
  '#FF5800', // 4  LBU
  '#C41E3A', // 5  RBU
  '#9B59B6', // 6  LFU
  '#00CED1', // 7  RFU
];

// One color per face piece (6 faces of the inscribed cube)
const SKEWB_FACE_COLORS = [
  '#FF5800', // 0  L
  '#C41E3A', // 1  R
  '#0046AD', // 2  B
  '#009B48', // 3  F
  '#FFD500', // 4  D
  '#FFFFFF', // 5  U
];

const SKEWB_FACE_LABELS = ['L', 'R', 'B', 'F', 'D', 'U'];

// face index → [TL, TR, BL, BR] corner slot indices (0-7)
const SKEWB_FACE_CORNERS = [
  [4, 6, 0, 2], // 0 L: LBU, LFU, LBD, LFD
  [7, 5, 3, 1], // 1 R: RFU, RBU, RFD, RBD
  [5, 4, 1, 0], // 2 B: RBU, LBU, RBD, LBD
  [6, 7, 2, 3], // 3 F: LFU, RFU, LFD, RFD
  [2, 3, 0, 1], // 4 D: LFD, RFD, LBD, RBD
  [4, 5, 6, 7], // 5 U: LBU, RBU, LFU, RFU
];

// ---------------------------------------------------------------------------
// Puzzle display names (dropdown labels)
// ---------------------------------------------------------------------------

const PUZZLE_DISPLAY_NAMES = {
  megaminx:       'Megaminx',
  skewb_ultimate: 'Skewb Ultimate',
};
