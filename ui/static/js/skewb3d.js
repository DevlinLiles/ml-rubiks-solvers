/**
 * skewb3d.js — Three.js 3D Skewb Ultimate (dodecahedron) renderer
 *
 * Renders the Skewb Ultimate as a dodecahedron (its true physical shape).
 * Each of the 12 pentagonal faces is subdivided into 11 extruded tiles using
 * the same geometry as Megaminx.  Color assignment follows the Skewb state:
 *
 *   - Tiles at the 2 "cube-corner" vertices per face → corner piece color
 *   - All other tiles (center, edges, 3 non-cube-corner kites) → face piece color
 *
 * All tile geometry is in world space with identity mesh transforms.
 *
 * State format: colors[slotIdx] = [piece_id, orientation]  shape (14, 2)
 *   rows 0-7  = corner slots (piece_id 0-7)
 *   rows 8-13 = face piece slots (piece_id 0-5)
 *
 * Public API:
 *   setState(colors)
 *   resetTransforms()
 *   animateMove(moveData, newColors, durationMs)  — async
 *   destroy()
 *
 * Requires: THREE (r128+), geometry-utils.js,
 *           SKEWB_CORNER_COLORS, SKEWB_FACE_COLORS from colors.js
 */

// ---------------------------------------------------------------------------
// Dodecahedron geometry (shared constants — identical to megaminx3d.js)
// ---------------------------------------------------------------------------

// _M_PHI, _M_R, _DODEC_RAW_VERTS, _DODEC_FACES, _DODEC_RADIUS, _DODEC_SCALE
// are already declared by megaminx3d.js which loads first.

const _SK_TILE_DEPTH  = 0.10;
const _SK_INNER_SCALE = 0.50;

// ---------------------------------------------------------------------------
// Cube-face normals for the 6 Skewb face pieces.
// Order matches face-piece indices: L=0 R=1 B=2 F=3 D=4 U=5
// Coordinate system: L=-X R=+X  B=-Z F=+Z  D=-Y U=+Y
// ---------------------------------------------------------------------------
const _SK_CUBE_FACE_NORMALS = [
  new THREE.Vector3(-1, 0,  0), // 0 L
  new THREE.Vector3( 1, 0,  0), // 1 R
  new THREE.Vector3( 0, 0, -1), // 2 B
  new THREE.Vector3( 0, 0,  1), // 3 F
  new THREE.Vector3( 0,-1,  0), // 4 D
  new THREE.Vector3( 0, 1,  0), // 5 U
];

// Mapping: dodecahedron vertex index (0-7, the ±1,±1,±1 cube corners)
// → Skewb corner piece slot index.
// Dodec vertex coords: 0=(-1,-1,-1)=LBD, 1=(-1,-1,1)=LFD, 2=(-1,1,-1)=LBU,
//                      3=(-1,1,1)=LFU, 4=(1,-1,-1)=RBD, 5=(1,-1,1)=RFD,
//                      6=(1,1,-1)=RBU, 7=(1,1,1)=RFU
// Skewb corner indices: 0=LBD 1=RBD 2=LFD 3=RFD 4=LBU 5=RBU 6=LFU 7=RFU
const _SK_DODEC_VERT_TO_CORNER = [0, 2, 4, 6, 1, 3, 5, 7];

// ---------------------------------------------------------------------------
// Skewb rotation axes (body diagonals of inscribed cube).
// Positive direction = CW when viewed from the lower-indexed corner.
// ---------------------------------------------------------------------------
const _SK_AXES = {
  L: new THREE.Vector3( 1,  1,  1).normalize(),
  R: new THREE.Vector3(-1,  1,  1).normalize(),
  F: new THREE.Vector3( 1,  1, -1).normalize(),
  B: new THREE.Vector3(-1,  1, -1).normalize(),
};

// ---------------------------------------------------------------------------
// SkewbUltimate3D
// ---------------------------------------------------------------------------

class SkewbUltimate3D {
  /**
   * @param {HTMLElement} container
   */
  constructor(container) {
    this.container  = container;
    this.tiles      = [];   // [{mesh, faceIdx, stickerIdx, tileType, pieceRef}]
    this.faceNormals = [];  // THREE.Vector3[12]

    // Per-face metadata: which cube-face piece, which corner slots for kite tiles
    this._facePieceSlot  = new Array(12); // index into colors[8..13]
    this._cornerKiteSlot = new Array(12); // [5] elements: null or cornerSlotIdx (0..7)

    this._initScene();
    this._buildBody();
    this._buildTiles();
    this._precomputeMoveSets();
    this._startRenderLoop();
    this._bindResize();
  }

  // -------------------------------------------------------------------------
  // Scene setup
  // -------------------------------------------------------------------------

  _initScene() {
    const w = this.container.clientWidth  || 600;
    const h = this.container.clientHeight || 600;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    this.container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d1117);

    this.camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
    this.camera.position.set(0, 2.0, 6.5);
    this.camera.lookAt(0, 0, 0);

    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.minDistance   = 4;
      this.controls.maxDistance   = 14;
      this.controls.enablePan     = false;
    }

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.50));
    const sun = new THREE.DirectionalLight(0xffffff, 0.80);
    sun.position.set(5, 8, 6);
    this.scene.add(sun);
    const fill = new THREE.DirectionalLight(0x88aaff, 0.25);
    fill.position.set(-4, -3, -5);
    this.scene.add(fill);
  }

  // -------------------------------------------------------------------------
  // Geometry construction
  // -------------------------------------------------------------------------

  _buildBody() {
    const geo = new THREE.DodecahedronGeometry(_DODEC_RADIUS * 0.97, 0);
    const mat = new THREE.MeshPhongMaterial({ color: 0x1a1a1a, shininess: 20 });
    this.scene.add(new THREE.Mesh(geo, mat));
  }

  /**
   * Build 132 extruded tile meshes with Skewb piece metadata.
   *
   * For each dodecahedron face fi:
   *   - Find nearest cube-face normal → _facePieceSlot[fi] ∈ 0-5
   *   - For each of the 5 pentagon vertices: if it's a cube corner (dodec vert 0-7),
   *     store the Skewb corner slot; otherwise null.
   *   - 11 tiles: stickers 0-5 (center+edges) + stickers 6-10 (corner kites).
   *     The corner kite at vertex i maps to _cornerKiteSlot[fi][i]:
   *       non-null → corner piece;  null → face piece.
   */
  _buildTiles() {
    const scale = _DODEC_SCALE;
    const verts = _DODEC_RAW_VERTS.map(([x,y,z]) =>
      new THREE.Vector3(x, y, z).multiplyScalar(scale)
    );

    for (let fi = 0; fi < 12; fi++) {
      const faceVertIdxs = _DODEC_FACES[fi];
      const fv           = faceVertIdxs.map(vi => verts[vi]);

      // Centroid and normal
      const C = new THREE.Vector3();
      fv.forEach(v => C.add(v));
      C.divideScalar(5);
      const normal = C.clone().normalize();
      this.faceNormals.push(normal.clone());

      // Nearest cube face → face piece slot
      let bestSlot = 0, bestDot = -Infinity;
      for (let k = 0; k < 6; k++) {
        const d = normal.dot(_SK_CUBE_FACE_NORMALS[k]);
        if (d > bestDot) { bestDot = d; bestSlot = k; }
      }
      this._facePieceSlot[fi] = bestSlot;

      // Corner kite slots: which of the 5 pentagon vertex positions is a cube corner?
      const kiteSlots = faceVertIdxs.map(vi =>
        vi < 8 ? _SK_DODEC_VERT_TO_CORNER[vi] : null
      );
      this._cornerKiteSlot[fi] = kiteSlots;

      // Local UV frame
      const uAxis = fv[0].clone().sub(C).normalize();
      const vAxis = normal.clone().cross(uAxis).normalize();

      const proj = fv.map(p => {
        const d = p.clone().sub(C);
        return { x: d.dot(uAxis), y: d.dot(vAxis) };
      });

      const inner    = proj.map(p => ({ x: p.x * _SK_INNER_SCALE, y: p.y * _SK_INNER_SCALE }));
      const outerMid = proj.map((p, i) => {
        const q = proj[(i + 1) % 5];
        return { x: (p.x + q.x) * 0.5, y: (p.y + q.y) * 0.5 };
      });

      // Same 11-tile subdivision as Megaminx
      const shapes = [
        [...inner],
        ...Array.from({ length: 5 }, (_, i) => [inner[i], inner[(i + 1) % 5], outerMid[i]]),
        ...Array.from({ length: 5 }, (_, i) => [
          inner[i],
          outerMid[(i - 1 + 5) % 5],
          proj[i],
          outerMid[i],
        ]),
      ];

      for (let si = 0; si < 11; si++) {
        const pts2d   = shapes[si];
        const topPts  = pts2d.map(p =>
          C.clone().addScaledVector(uAxis, p.x).addScaledVector(vAxis, p.y).addScaledVector(normal, _SK_TILE_DEPTH)
        );
        const basePts = pts2d.map(p =>
          C.clone().addScaledVector(uAxis, p.x).addScaledVector(vAxis, p.y)
        );

        // Compute tile centroid (avg of top vertices) for move-set classification
        const centroid = new THREE.Vector3();
        topPts.forEach(p => centroid.add(p));
        centroid.divideScalar(topPts.length);

        const geo  = buildTileGeometry(topPts, basePts);
        const mesh = createTileMesh(geo, 0x888888); // placeholder, setState fills real color

        this.scene.add(mesh);
        this.tiles.push({ mesh, faceIdx: fi, stickerIdx: si, centroid });
      }
    }
  }

  // -------------------------------------------------------------------------
  // Precompute which tiles move for each of the 4 Skewb moves
  // -------------------------------------------------------------------------

  _precomputeMoveSets() {
    this._moveSets = {};
    for (const [moveName, axis] of Object.entries(_SK_AXES)) {
      this._moveSets[moveName] = this.tiles.filter(t => t.centroid.dot(axis) < 0);
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Update all tile colors from Skewb state.
   * @param {number[][]} colors  shape (14, 2) — [piece_id, orientation]
   *   rows 0-7  = corner slots
   *   rows 8-13 = face piece slots
   */
  setState(colors) {
    for (const { mesh, faceIdx, stickerIdx } of this.tiles) {
      let color;

      if (stickerIdx >= 6) {
        // Corner kite — vertex position index within face = stickerIdx - 6
        const vi        = stickerIdx - 6;
        const cornerSlot = this._cornerKiteSlot[faceIdx][vi];

        if (cornerSlot !== null) {
          // This kite is at a cube-corner vertex → show corner piece color
          const [pid] = colors[cornerSlot] ?? [cornerSlot, 0];
          color = SKEWB_CORNER_COLORS[pid] ?? '#333333';
        } else {
          // Non-cube-corner vertex → show face piece color
          const fp    = this._facePieceSlot[faceIdx];
          const [pid] = colors[8 + fp] ?? [fp, 0];
          color = SKEWB_FACE_COLORS[pid] ?? '#333333';
        }
      } else {
        // Center (si=0) or edge (si=1..5) → face piece color
        const fp    = this._facePieceSlot[faceIdx];
        const [pid] = colors[8 + fp] ?? [fp, 0];
        color = SKEWB_FACE_COLORS[pid] ?? '#333333';
      }

      mesh.material[0].color.set(color);
    }
  }

  /**
   * Reset all tile transforms to identity.  Call before setState when jumping steps.
   */
  resetTransforms() {
    for (const { mesh } of this.tiles) {
      mesh.position.set(0, 0, 0);
      mesh.quaternion.set(0, 0, 0, 1);
      mesh.scale.set(1, 1, 1);
      mesh.updateMatrix();
    }
  }

  /**
   * Animate a Skewb half-puzzle rotation (120°), then snap to new state.
   * @param {object}   moveData   — {face: 'L'|'R'|'F'|'B', direction: ±1}
   * @param {number[][]} newColors
   * @param {number}   durationMs
   * @returns {Promise<void>}
   */
  async animateMove(moveData, newColors, durationMs) {
    const axis    = _SK_AXES[moveData.face];
    const moving  = this._moveSets[moveData.face];

    if (!axis || !moving) {
      this.resetTransforms();
      this.setState(newColors);
      return;
    }

    const totalAngle = moveData.direction * (2 * Math.PI / 3);

    const pivot = new THREE.Group();
    this.scene.add(pivot);
    for (const { mesh } of moving) pivot.attach(mesh);

    await new Promise(resolve => {
      const start = performance.now();
      const tick = () => {
        const raw  = (performance.now() - start) / durationMs;
        const t    = Math.min(raw, 1);
        const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        pivot.quaternion.setFromAxisAngle(axis, ease * totalAngle);

        if (t < 1) {
          requestAnimationFrame(tick);
        } else {
          for (const { mesh } of moving) this.scene.attach(mesh);
          this.scene.remove(pivot);
          this.resetTransforms();
          this.setState(newColors);
          resolve();
        }
      };
      requestAnimationFrame(tick);
    });
  }

  // -------------------------------------------------------------------------
  // Render loop / lifecycle
  // -------------------------------------------------------------------------

  _startRenderLoop() {
    const tick = () => {
      this._rafId = requestAnimationFrame(tick);
      if (this.controls) this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    tick();
  }

  _bindResize() {
    this._resizeObserver = new ResizeObserver(() => {
      const w = this.container.clientWidth;
      const h = this.container.clientHeight;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    });
    this._resizeObserver.observe(this.container);
  }

  destroy() {
    if (this._rafId) cancelAnimationFrame(this._rafId);
    if (this._resizeObserver) this._resizeObserver.disconnect();
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}
