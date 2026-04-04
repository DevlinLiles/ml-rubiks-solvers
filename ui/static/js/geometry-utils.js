/**
 * geometry-utils.js — shared Three.js geometry helpers for extruded puzzle tiles.
 *
 * buildTileGeometry(topPts, basePts)
 *   Create a BufferGeometry from two parallel rings of THREE.Vector3 world
 *   positions.  topPts is the visible (outward-facing) polygon; basePts is
 *   the same polygon at the body surface.
 *
 *   Material groups:
 *     group 0 (materialIndex 0): top face (colored sticker surface)
 *     group 1 (materialIndex 1): side walls (dark)
 *
 * createTileMesh(geo, topColor)
 *   Wrap a tile geometry in a Mesh with [colorMat, darkMat].
 */

function buildTileGeometry(topPts, basePts) {
  const n = topPts.length;
  const positions = [];

  // Top ring: indices 0 … n-1
  for (const p of topPts)  positions.push(p.x, p.y, p.z);
  // Base ring: indices n … 2n-1
  for (const p of basePts) positions.push(p.x, p.y, p.z);

  const topIdx  = [];
  const sideIdx = [];

  // Top face — fan from vertex 0.
  // Winding order: 0 → i → i+1 (adjust below if face renders dark).
  for (let i = 1; i < n - 1; i++) {
    topIdx.push(0, i, i + 1);
  }

  // Side walls — one quad per edge, split into 2 triangles.
  for (let i = 0; i < n; i++) {
    const a = i;
    const b = (i + 1) % n;
    const c = n + i;
    const d = n + (i + 1) % n;
    sideIdx.push(a, c, b);
    sideIdx.push(b, c, d);
  }

  const allIdx = [...topIdx, ...sideIdx];
  const geo = new THREE.BufferGeometry();
  geo.setIndex(allIdx);
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.computeVertexNormals();
  geo.addGroup(0,            topIdx.length,  0);  // top
  geo.addGroup(topIdx.length, sideIdx.length, 1);  // sides
  return geo;
}

function createTileMesh(geo, topColor) {
  return new THREE.Mesh(geo, [
    new THREE.MeshPhongMaterial({ color: topColor, shininess: 60, side: THREE.DoubleSide }),
    new THREE.MeshPhongMaterial({ color: 0x1a1a1a, shininess: 10 }),
  ]);
}
