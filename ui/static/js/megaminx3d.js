/**
 * megaminx3d.js — Three.js 3D Megaminx (dodecahedron) renderer
 *
 * Each of the 12 pentagonal faces is subdivided into 11 extruded tile meshes:
 *   sticker 0        — center pentagon
 *   stickers 1-5     — 5 edge triangles (CW order)
 *   stickers 6-10    — 5 corner kites   (CW order)
 *
 * All tile geometry is in world space with identity mesh transforms so that
 * pivot-group animation works correctly.
 *
 * Public API:
 *   setState(colors)                           — colors[fi][si] = colorIndex 0-11
 *   resetTransforms()                          — snap all tiles back to world-space positions
 *   animateMove(moveData, newColors, duration) — async face-turn animation
 *   destroy()
 *
 * Requires: THREE (r128+), geometry-utils.js, MEGA_COLORS from colors.js
 */

// ---------------------------------------------------------------------------
// Dodecahedron geometry constants
// ---------------------------------------------------------------------------

const _M_PHI = (1 + Math.sqrt(5)) / 2;
const _M_R   = 1 / _M_PHI;

const _DODEC_RAW_VERTS = [
  [-1,-1,-1],[-1,-1, 1],[-1, 1,-1],[-1, 1, 1],
  [ 1,-1,-1],[ 1,-1, 1],[ 1, 1,-1],[ 1, 1, 1],
  [ 0,-_M_R,-_M_PHI],[ 0,-_M_R, _M_PHI],
  [ 0, _M_R,-_M_PHI],[ 0, _M_R, _M_PHI],
  [-_M_R,-_M_PHI, 0],[-_M_R, _M_PHI, 0],
  [ _M_R,-_M_PHI, 0],[ _M_R, _M_PHI, 0],
  [-_M_PHI, 0,-_M_R],[ _M_PHI, 0,-_M_R],
  [-_M_PHI, 0, _M_R],[ _M_PHI, 0, _M_R],
];

const _DODEC_FACES = [
  [ 3,11, 7,15,13], //  0 U
  [ 7,19,17, 6,15], //  1 F
  [17, 4, 8,10, 6], //  2 R
  [ 8, 0,16, 2,10], //  3 BR
  [ 0,12, 1,18,16], //  4 BL
  [ 6,10, 2,13,15], //  5 L
  [ 2,16,18, 3,13], //  6 D
  [18, 1, 9,11, 3], //  7 DF
  [ 4,14,12, 0, 8], //  8 DR
  [11, 9, 5,19, 7], //  9 DBR
  [19, 5,14, 4,17], // 10 DBL
  [ 1,12,14, 5, 9], // 11 DL
];

const _DODEC_RADIUS = 2.2;
const _DODEC_SCALE  = _DODEC_RADIUS / Math.sqrt(3);

// Face-name → face index
const _MEGA_FACE_IDX = {
  U:0,F:1,R:2,BR:3,BL:4,L:5,D:6,DF:7,DR:8,DBR:9,DBL:10,DL:11,
};

// ---------------------------------------------------------------------------
// Face adjacency: which neighbouring stickers rotate with a face turn.
// _MEGA_FACE_ADJ[fi] = [[neighbourFace, [s0,s1,s2]], ...] × 5
// (Translated 1-to-1 from Python FACE_ADJACENCY in megaminx.py)
// ---------------------------------------------------------------------------
const _MEGA_FACE_ADJ = [
  // 0 U
  [[1,[10,1,6]],[2,[10,1,6]],[3,[10,1,6]],[4,[10,1,6]],[5,[10,1,6]]],
  // 1 F
  [[0,[7,3,8]],[2,[9,5,10]],[7,[6,1,10]],[11,[8,3,7]],[5,[9,5,10]]],
  // 2 R
  [[0,[8,4,9]],[3,[9,5,10]],[8,[6,1,10]],[7,[8,3,7]],[1,[9,5,10]]],
  // 3 BR
  [[0,[9,5,10]],[4,[9,5,10]],[9,[6,1,10]],[8,[8,3,7]],[2,[9,5,10]]],
  // 4 BL
  [[0,[6,1,10]],[5,[9,5,10]],[10,[6,1,10]],[9,[8,3,7]],[3,[9,5,10]]],
  // 5 L
  [[0,[10,2,7]],[1,[9,5,10]],[11,[6,1,10]],[10,[8,3,7]],[4,[9,5,10]]],
  // 6 D
  [[7,[7,3,8]],[8,[7,3,8]],[9,[7,3,8]],[10,[7,3,8]],[11,[7,3,8]]],
  // 7 DF
  [[6,[10,1,6]],[8,[9,5,10]],[2,[8,4,9]],[1,[8,3,7]],[11,[9,5,10]]],
  // 8 DR
  [[6,[6,2,7]],[9,[9,5,10]],[3,[8,4,9]],[2,[8,3,7]],[7,[9,5,10]]],
  // 9 DBR
  [[6,[7,3,8]],[10,[9,5,10]],[4,[8,4,9]],[3,[8,3,7]],[8,[9,5,10]]],
  // 10 DBL
  [[6,[8,4,9]],[11,[9,5,10]],[5,[8,4,9]],[4,[8,3,7]],[9,[9,5,10]]],
  // 11 DL
  [[6,[9,5,10]],[7,[9,5,10]],[1,[8,4,9]],[5,[8,3,7]],[10,[9,5,10]]],
];

const _TILE_DEPTH   = 0.10;  // extrusion above body surface
const _INNER_SCALE  = 0.50;  // inner-pentagon fraction of outer vertex distance

// ---------------------------------------------------------------------------
// Megaminx3D
// ---------------------------------------------------------------------------

class Megaminx3D {
  /**
   * @param {HTMLElement} container
   */
  constructor(container) {
    this.container  = container;
    this.tiles      = [];        // [{mesh, faceIdx, stickerIdx}]
    this.faceNormals = [];       // THREE.Vector3[12]
    this._initScene();
    this._buildBody();
    this._buildTiles();
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
   * Build 132 extruded tile meshes (12 faces × 11 tiles each).
   * All geometry is in world space; mesh transforms are identity.
   */
  _buildTiles() {
    const scale = _DODEC_SCALE;
    const verts = _DODEC_RAW_VERTS.map(([x,y,z]) =>
      new THREE.Vector3(x, y, z).multiplyScalar(scale)
    );

    for (let fi = 0; fi < 12; fi++) {
      const fv = _DODEC_FACES[fi].map(vi => verts[vi]);

      // Face centroid
      const C = new THREE.Vector3();
      fv.forEach(v => C.add(v));
      C.divideScalar(5);

      // Outward normal
      const normal = C.clone().normalize();
      this.faceNormals.push(normal.clone());

      // Local UV frame on the face plane
      const uAxis = fv[0].clone().sub(C).normalize();
      const vAxis = normal.clone().cross(uAxis).normalize();

      // Project face vertices to (u, v)
      const proj = fv.map(p => {
        const d = p.clone().sub(C);
        return { x: d.dot(uAxis), y: d.dot(vAxis) };
      });

      // Inner pentagon and edge midpoints in UV space
      const inner    = proj.map(p => ({ x: p.x * _INNER_SCALE, y: p.y * _INNER_SCALE }));
      const outerMid = proj.map((p, i) => {
        const q = proj[(i + 1) % 5];
        return { x: (p.x + q.x) * 0.5, y: (p.y + q.y) * 0.5 };
      });

      // 11 tile polygon definitions in UV space:
      //   si=0      : center pentagon (inner[0..4])
      //   si=1..5   : edge triangle i-1  [inner[i-1], inner[i], outerMid[i-1]]
      //   si=6..10  : corner kite  i-6   [inner[i-6], outerMid[(i-7+5)%5], proj[i-6], outerMid[i-6]]
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

      // Build each tile
      for (let si = 0; si < 11; si++) {
        const pts2d = shapes[si];

        // Convert UV polygon to world-space top/base rings
        const topPts  = pts2d.map(p =>
          C.clone()
            .addScaledVector(uAxis,   p.x)
            .addScaledVector(vAxis,   p.y)
            .addScaledVector(normal,  _TILE_DEPTH)
        );
        const basePts = pts2d.map(p =>
          C.clone()
            .addScaledVector(uAxis,  p.x)
            .addScaledVector(vAxis,  p.y)
        );

        const geo  = buildTileGeometry(topPts, basePts);
        const mesh = createTileMesh(geo, MEGA_COLORS[fi]);

        this.scene.add(mesh);
        this.tiles.push({ mesh, faceIdx: fi, stickerIdx: si });
      }
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Update all sticker colors from puzzle state.
   * @param {number[][]} colors  shape (12, 11) — each value = MEGA_COLORS index
   */
  setState(colors) {
    for (const { mesh, faceIdx, stickerIdx } of this.tiles) {
      const ci = colors[faceIdx]?.[stickerIdx] ?? faceIdx;
      mesh.material[0].color.set(MEGA_COLORS[ci] ?? '#333333');
    }
  }

  /**
   * Reset all tile mesh transforms to identity so world-space geometry renders
   * at the correct positions.  Call before setState when jumping steps.
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
   * Animate a face turn, then snap to the new state.
   * @param {object}   moveData    — {face: string, direction: ±1}
   * @param {number[][]} newColors — new puzzle state colors (12×11)
   * @param {number}   durationMs
   * @returns {Promise<void>}
   */
  async animateMove(moveData, newColors, durationMs) {
    const fi = _MEGA_FACE_IDX[moveData.face];
    if (fi === undefined) {
      this.resetTransforms();
      this.setState(newColors);
      return;
    }

    const totalAngle = moveData.direction * (2 * Math.PI / 5);
    const axis       = this.faceNormals[fi];
    const adj        = _MEGA_FACE_ADJ[fi];

    // Collect the 11 face tiles + 3 border strips from each of 5 neighbours (26 total)
    const moving = [];
    for (const tile of this.tiles) {
      if (tile.faceIdx === fi) { moving.push(tile); continue; }
      for (const [nf, strip] of adj) {
        if (tile.faceIdx === nf && strip.includes(tile.stickerIdx)) {
          moving.push(tile);
          break;
        }
      }
    }

    // Pivot group
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
