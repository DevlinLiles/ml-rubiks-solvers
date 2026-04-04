# 3D Render Improvement Plan — Megaminx & Skewb Ultimate

## Root cause of the current look

The Rubik's cube renderer (`cube3d.js`) builds **individual BoxGeometry cubies** — each
cubie is its own 3D box with 6 material slots.  The dark gaps between pieces are the
natural gaps between separate meshes.  Nothing is "painted on"; the geometry IS the piece.

The current Megaminx and Skewb renderers do the opposite: they paint flat
`PlaneGeometry` squares on top of a single solid body.  That produces the floating-sticker
look seen in image 5.

The fix for both puzzles is the same as the Rubik's cube: **replace the solid body +
floating stickers with individual per-piece 3D meshes whose colored surface IS the top
face and whose dark sides create natural piece boundaries.**

---

## Important shape correction

The Skewb Ultimate docstring says:
> "dodecahedral twisty puzzle built on the Skewb mechanism"

The current `skewb3d.js` renders it as a cube — that is wrong.
Both puzzles must be rendered as **dodecahedra**.

---

## Megaminx — Implementation Plan

### Piece count (62 movable + 12 fixed centers)

| Type    | Count | Faces visible from outside |
|---------|-------|---------------------------|
| Centers | 12    | 1 (the pentagonal face)    |
| Edges   | 30    | 2 (the two adjacent faces) |
| Corners | 20    | 3 (three meeting faces)    |

### Step 1 — Compute dodecahedron topology

Use the existing `_DODEC_RAW_VERTS` (20 vertices) and `_DODEC_FACES` (12 × 5 indices).

Derive:
- **30 edges**: iterate all adjacent face-pairs; if they share exactly 2 vertex indices,
  record `{vertA, vertB, faceA, faceB, edgeIdx-on-faceA, edgeIdx-on-faceB}`.
- **20 corners**: each vertex appears in exactly 3 faces; record
  `{vertIdx, face0, face1, face2}`.

### Step 2 — Face subdivision geometry

For each pentagonal face (centroid C, vertices V[0..4]):

```
innerScale = 0.50        // inner pentagon fraction
inner[i]   = C + innerScale*(V[i]-C)   // 5 inner-pentagon vertices
outerMid[i]= (V[i]+V[(i+1)%5])/2      // 5 edge midpoints
innerMid[i]= C + innerScale*(outerMid[i]-C)
```

**11 piece tile shapes (2D, in face-local UV space):**

| Sticker index | Type    | 2D polygon vertices                               |
|---------------|---------|---------------------------------------------------|
| 0             | Center  | inner[0..4]  (regular pentagon)                   |
| 1 (edge 0)    | Edge    | inner[0], inner[1], outerMid[0-right], outerMid[0-left] ← trapezoid |
| 2–5           | Edge    | same pattern for edges 1–4                        |
| 6 (corner 0)  | Corner  | V[0], midBetween(inner[0],V[0]→toward-M[4]), midBetween(inner[0],V[0]→toward-M[0]) ← triangle |
| 7–10          | Corner  | same for corners 1–4                              |

Exact trapezoid for edge i:
```
[inner[i], inner[(i+1)%5], outerMid[i] + epsilon_right, outerMid[i] + epsilon_left]
```

More precisely — the full clean tiling that leaves no gaps:
```
Corner piece i:
  V[i],  point_on_inner_between_V[i] and V[(i-1)%5],  inner[i],  point_on_inner_between_V[i] and V[(i+1)%5]
  → a kite/diamond shape at each vertex

Edge piece i  (between corner i and corner (i+1)%5):
  inner[i], inner[(i+1)%5], V[(i+1)%5], V[i]  trimmed by the corner kites
  → trapezoid between inner ring and outer edge
```

Use `THREE.Shape` + `THREE.ExtrudeGeometry` with depth ~0.08 world units.
The extruded direction is the face outward normal.
Top face material = sticker color.  Side material = `#1a1a1a`.

### Step 3 — 3D mesh placement

For each tile:
1. Build its `THREE.Shape` in the face's local (u,v) 2D coordinate system.
2. Extrude along the face normal (depth = 0.08).
3. Translate so the extruded base sits flush with the dodecahedron inradius surface.
4. Rotate to match the face orientation (same quaternion already used for sticker placement).

Result: each tile sits proud of the body surface like a real puzzle piece.

### Step 4 — State → piece color mapping

The existing state `(12, 11)` maps directly:
- `state[faceIdx][0]`   → center tile color on face `faceIdx`
- `state[faceIdx][1..5]`→ edge tile colors on face `faceIdx`
- `state[faceIdx][6..10]`→ corner tile colors on face `faceIdx`

`setState(colors)` just loops over all 132 tile meshes and sets `material[0].color`.

### Step 5 — Face turn animation

For a move `{face, direction}`:

1. **Rotation axis**: the outward normal of that face = `normalize(faceCenter)`.
   Pre-compute and store `MEGA_FACE_NORMALS[0..11]` at init time from the
   12 face centroids.

2. **Rotation angle**: `direction * (2π/5)` (72°).

3. **Which tiles rotate**: all 11 tiles on the turning face **plus** the 5 border
   tiles on each of the 5 adjacent faces (the sticker that abuts the edge of the
   turning face).  Total ≈ 11 + 5×3 = 26 tiles per turn.

   To find adjacent border tiles: pre-compute for each face the 5 neighboring
   face/sticker-index pairs during init.

4. **Animation**: same pivot-group pattern as `cube3d.js` —
   move affected meshes into a pivot `THREE.Group`, rotate the group over
   `durationMs` with ease-in-out, detach meshes, call `setState(newState)`.

### Step 6 — Data structures stored at init

```javascript
this.tiles       // [{mesh, faceIdx, stickerIdx}]  — 132 entries
this.faceNormals // [THREE.Vector3 × 12]
this.faceNeighbors // [{face, stickerIdx}[][12]] — 5 border-sticker refs per face
this.pivotGroup  // THREE.Group for animation
```

---

## Skewb Ultimate — Implementation Plan

### Shape correction: render as dodecahedron

The Skewb Ultimate IS a dodecahedron.  The 8 corners and 6 face pieces of its Skewb
mechanism correspond to the inscribed cube's vertices and faces, but the physical
**outer surface** is a dodecahedron.  Render it that way.

### Piece layout on the dodecahedron surface

The Skewb mechanism cuts along 4 axes, each passing through opposite inscribed-cube
vertices.  These cut planes divide the dodecahedron into two halves.  The result is:

- **8 corner pieces**: each covers a "cap" region near one of the 8 inscribed-cube
  vertices (= 8 dodecahedron vertices).  Each is visible across 3 dodecahedron faces.
- **6 face pieces**: rhombus-shaped pieces, one centred on each inscribed-cube face
  direction.  Each is visible across multiple dodecahedron faces.

For a practical visual (avoids computing exact Skewb cut geometry through the
dodecahedron surface), use the following observable fact:

> On each of the 12 pentagonal faces, the Skewb cut divides the pentagon into exactly
> **one face-piece region** (centre rhombus) and **two or three corner-piece regions**.

However, this exact cut geometry per face requires non-trivial plane–pentagon
intersection math.

### Simplified but accurate approach

Use the same face-subdivision as the Megaminx but with only **6 tiles per face**
matching the Skewb piece layout:

```
Each pentagonal face shows:
  1 large center-diamond tile  → belongs to one of the 6 face pieces
  5 corner-triangle tiles      → each belongs to one of the 8 corner pieces
```

This maps directly to the existing state: `colors[8+faceIdx]` for the center,
and `SKEWB_FACE_CORNERS[faceIdx][cornerIdx]` for each of the 5 corners.

**Tile shapes (2D, in face-local UV):**

```
Corner tile i:  triangle  [C, midpoint(C, V[i]), midpoint(C, V[(i+1)%5])]
  scaled out so outer edge aligns with the face boundary near V[i]

Center tile:    pentagon  inner[0..4] at scale 0.55
  (the remaining region after the 5 corner triangles are removed)
```

Each tile extruded with `THREE.ExtrudeGeometry` depth 0.08 along face normal.

**Color mapping:**
```javascript
// center tile on face fi:
const [pid] = colors[8 + fi];
color = SKEWB_FACE_COLORS[pid];

// corner tile ci (0..4) on face fi:
const slot = SKEWB_FACE_CORNERS[fi][ci];   // ci maps to TL/TR/BL/BR/top
const [pid] = colors[slot];
color = SKEWB_CORNER_COLORS[pid];
```

Note: SKEWB_FACE_CORNERS only defines 4 corner indices per face but the pentagon
has 5 corners.  The mapping for the 5th corner must be derived from the puzzle
topology (each pentagon vertex is shared by 3 faces; use the adjacent-face corner
lookups).  This is a one-time init computation.

### Step — Animation

For a move `{face: 'L'|'R'|'F'|'B', direction}`:

The 4 Skewb axes pass through inscribed-cube corner pairs:
```
L: LBD(0) ↔ RFU(7)  axis = normalize((-1,-1,-1) ↔ (+1,+1,+1)) = normalize(1,1,1)
R: RBD(1) ↔ LFU(6)  axis = normalize((+1,-1,-1) ↔ (-1,+1,+1)) = normalize(-1,1,1)
F: LFD(2) ↔ RBU(5)  axis = normalize((-1,-1,+1) ↔ (+1,+1,-1)) = normalize(1,1,-1)
B: RFD(3) ↔ LBU(4)  axis = normalize((+1,-1,+1) ↔ (-1,+1,-1)) = normalize(-1,1,-1)
```
(All normalized to unit vectors.)

Rotation angle: `direction * (2π/3)` (120°).

Which tiles rotate: half the puzzle.  Specifically, the "near" half (containing the
lower-indexed corner) rotates.  Pre-compute the set of tile meshes in each half for
all 4 axes at init time using dot-product against the axis to classify each tile's
face centroid.

---

## Implementation Sequence

### Phase 1 — Megaminx (highest visual impact, cleaner geometry)
1. Add `_buildTiles()` to `megaminx3d.js` replacing `_buildStickers()`
   - Compute tile shapes using `THREE.Shape` + `THREE.ExtrudeGeometry`
   - Store as `this.tiles`
2. Update `setState()` to set tile top-face material color
3. Add `MEGA_FACE_NORMALS` and `_faceTileGroups` pre-computation
4. Implement `animateMove(moveData, newState, durationMs)` with pivot-group animation

### Phase 2 — Skewb Ultimate (shape correction + tile geometry)
1. Rename `skewb3d.js` renderer to use dodecahedron body (same as Megaminx)
2. Replace 5-sticker cube faces with 6-tile pentagon faces using Skewb piece layout
3. Implement `animateMove()` with 120° body-diagonal rotation

### Phase 3 — Hook animations into `solve-controller.js`
Currently `_stepForward()` and `_stepBack()` skip animation for Megaminx/Skewb.
Once `animateMove()` exists on both classes, add:
```javascript
} else if (this.isMegaminx && this.megaminx3d) {
  await this.megaminx3d.animateMove(moveData, newState.colors, durationMs);
} else if (this.isSkewb && this.skewb3d) {
  await this.skewb3d.animateMove(moveData, newState.colors, durationMs);
}
```

---

## Key geometry helpers needed (shared utility)

```javascript
// Build a THREE.Shape from an array of 2D {x,y} points
function shapeFrom2D(pts) { ... }

// Extrude a shape along a direction vector, return BufferGeometry
function extrudeAlongNormal(shape, normal, depth) { ... }

// Multi-material mesh: slot 0 = top (colored), slot 1 = sides (dark)
function tileMesh(extrudedGeo, topColor) { ... }
```

These can live in a small `geometry-utils.js` file loaded before the puzzle renderers.
