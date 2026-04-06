/**
 * cube3d.js — Three.js 3D Rubik's cube renderer
 *
 * Face index conventions (from cube_nnn.py):
 *   0=U (White), 1=D (Yellow), 2=F (Green), 3=B (Blue), 4=L (Orange), 5=R (Red)
 *
 * Cubie grid position (ix, iy, iz) ∈ [0, n-1]^3:
 *   ix=n-1 → right (+X),  ix=0 → left  (-X)
 *   iy=n-1 → top   (+Y),  iy=0 → bottom(-Y)
 *   iz=n-1 → front (+Z),  iz=0 → back  (-Z)
 *
 * BoxGeometry material face order: 0=+X(R), 1=-X(L), 2=+Y(U), 3=-Y(D), 4=+Z(F), 5=-Z(B)
 */

// Color palette: index matches face index 0-5, index 6 = inner dark
const FACE_COLORS = [
  '#FFFFFF', // 0 = U = White
  '#FFD500', // 1 = D = Yellow
  '#009B48', // 2 = F = Green
  '#0046AD', // 3 = B = Blue
  '#FF5800', // 4 = L = Orange
  '#C41E3A', // 5 = R = Red
  '#1a1a1a', // 6 = inner
];

// BoxGeometry face → axis/sign for move animation
const FACE_AXIS = { U: 'y', D: 'y', F: 'z', B: 'z', L: 'x', R: 'x' };
const FACE_SIGN = { U: +1, D: -1, F: -1, B: +1, L: +1, R: -1 };

/**
 * Given a face and moveLayer, return which grid coordinate value cubies must
 * have to belong to that layer.
 */
function layerGridIndex(face, moveLayer, n) {
  switch (face) {
    case 'U': return n - 1 - moveLayer;
    case 'D': return moveLayer;
    case 'F': return n - 1 - moveLayer;
    case 'B': return moveLayer;
    case 'L': return moveLayer;
    case 'R': return n - 1 - moveLayer;
    default:  return 0;
  }
}

/**
 * Compute the color index for a given material slot on a cubie at (ix,iy,iz).
 * Returns 0-5 for face colors, 6 for inner (hidden) faces.
 *
 * @param {number} matSlot  0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z
 * @param {number} ix
 * @param {number} iy
 * @param {number} iz
 * @param {number} n
 * @param {number[][][]} faceColors  state[face][row][col]  (already a plain JS array)
 */
function getCubieColor(matSlot, ix, iy, iz, n, faceColors) {
  switch (matSlot) {
    case 2: // +Y = U face
      if (iy === n - 1) return faceColors[0][iz][ix];
      return 6;
    case 3: // -Y = D face
      if (iy === 0) return faceColors[1][(n - 1) - iz][ix];
      return 6;
    case 4: // +Z = F face
      if (iz === n - 1) return faceColors[2][(n - 1) - iy][ix];
      return 6;
    case 5: // -Z = B face
      if (iz === 0) return faceColors[3][(n - 1) - iy][(n - 1) - ix];
      return 6;
    case 1: // -X = L face
      if (ix === 0) return faceColors[4][(n - 1) - iy][iz];
      return 6;
    case 0: // +X = R face
      if (ix === n - 1) return faceColors[5][(n - 1) - iy][(n - 1) - iz];
      return 6;
    default:
      return 6;
  }
}

class RubiksCube3D {
  /**
   * @param {HTMLElement} container  DOM element to render into
   * @param {number} n               Cube dimension (2, 3, 4, 5)
   */
  constructor(container, n = 3) {
    this.container = container;
    this.n = n;
    this.cubies = [];      // flat array of { mesh, gridPos: {ix,iy,iz} }
    this._animating = false;

    this._initScene();
    this._buildCube();
    this._startRenderLoop();
    this._bindResize();
  }

  // ---------------------------------------------------------------------------
  // Scene setup
  // ---------------------------------------------------------------------------

  _initScene() {
    const { clientWidth: w, clientHeight: h } = this.container;

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputEncoding = THREE.sRGBEncoding;
    this.container.appendChild(this.renderer.domElement);

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d1117);
    this.scene.fog = new THREE.Fog(0x0d1117, 20, 60);

    // Camera
    const aspect = w / h;
    this.camera = new THREE.PerspectiveCamera(40, aspect, 0.1, 100);
    this.camera.position.set(4.5, 4.0, 6.5);
    this.camera.lookAt(0, 0, 0);

    // Orbit controls
    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.minDistance = 3;
      this.controls.maxDistance = 18;
      this.controls.enablePan = false;
    }

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.45);
    this.scene.add(ambient);

    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(6, 10, 8);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 1024;
    sun.shadow.mapSize.height = 1024;
    this.scene.add(sun);

    const fill = new THREE.DirectionalLight(0x88aaff, 0.3);
    fill.position.set(-5, -3, -6);
    this.scene.add(fill);

    // Pivot group used for layer animations
    this.pivotGroup = new THREE.Group();
    this.scene.add(this.pivotGroup);
  }

  _bindResize() {
    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this.container);
  }

  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  _startRenderLoop() {
    const tick = () => {
      requestAnimationFrame(tick);
      if (this.controls) this.controls.update();
      this.renderer.render(this.scene, this.camera);
    };
    tick();
  }

  // ---------------------------------------------------------------------------
  // Cube construction
  // ---------------------------------------------------------------------------

  _buildCube() {
    // Remove any existing cubies
    for (const { mesh } of this.cubies) {
      this.scene.remove(mesh);
    }
    this.cubies = [];

    const n = this.n;
    const gap = 1.02;                    // spacing between cubies
    const half = (n - 1) / 2;

    const geo = new THREE.BoxGeometry(0.95, 0.95, 0.95);

    for (let ix = 0; ix < n; ix++) {
      for (let iy = 0; iy < n; iy++) {
        for (let iz = 0; iz < n; iz++) {
          // Skip purely internal cubies (only corners/edges/faces are visible)
          if (n > 2 && ix > 0 && ix < n-1 && iy > 0 && iy < n-1 && iz > 0 && iz < n-1) continue;

          const mats = Array.from({ length: 6 }, (_, slot) => {
            const colorIdx = getCubieColor(slot, ix, iy, iz, n, _solvedFaceColors(n));
            return new THREE.MeshPhongMaterial({
              color: new THREE.Color(FACE_COLORS[colorIdx]),
              shininess: colorIdx === 6 ? 0 : 60,
              specular: colorIdx === 6 ? new THREE.Color(0x000000) : new THREE.Color(0x444444),
            });
          });

          const mesh = new THREE.Mesh(geo, mats);
          mesh.castShadow = true;
          mesh.receiveShadow = true;
          mesh.position.set(
            (ix - half) * gap,
            (iy - half) * gap,
            (iz - half) * gap,
          );

          this.scene.add(mesh);
          this.cubies.push({ mesh, gridPos: { ix, iy, iz }, _ix0: ix, _iy0: iy, _iz0: iz });
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // State update
  // ---------------------------------------------------------------------------

  /**
   * Update all cubie face colors from a state colors array.
   * @param {number[][][]} faceColors  shape (6, n, n) as nested JS arrays
   */
  setState(faceColors) {
    const n = this.n;
    for (const { mesh, gridPos: { ix, iy, iz } } of this.cubies) {
      for (let slot = 0; slot < 6; slot++) {
        const colorIdx = getCubieColor(slot, ix, iy, iz, n, faceColors);
        mesh.material[slot].color.set(FACE_COLORS[colorIdx]);
        mesh.material[slot].shininess = colorIdx === 6 ? 0 : 60;
      }
    }
  }

  /**
   * Reset every cubie to its initial solved-state world position and orientation.
   * Must be called before setState() whenever jumping to an arbitrary step, so
   * that gridPos values are valid for the next animateMove() call.
   */
  resetPositions() {
    // Flush any cubies still parented to the pivot group from an interrupted animation.
    while (this.pivotGroup.children.length > 0) {
      this.scene.attach(this.pivotGroup.children[0]);
    }
    this.pivotGroup.rotation.set(0, 0, 0);

    const gap = 1.02;
    const half = (this.n - 1) / 2;
    for (const cubie of this.cubies) {
      const { mesh, _ix0: ix, _iy0: iy, _iz0: iz } = cubie;
      if (mesh.parent !== this.scene) this.scene.attach(mesh);
      mesh.position.set((ix - half) * gap, (iy - half) * gap, (iz - half) * gap);
      mesh.quaternion.set(0, 0, 0, 1);
      cubie.gridPos.ix = ix;
      cubie.gridPos.iy = iy;
      cubie.gridPos.iz = iz;
    }
  }

  // ---------------------------------------------------------------------------
  // Move animation
  // ---------------------------------------------------------------------------

  /**
   * Animate a single move, snapping positions afterwards.
   * @param {object} moveData    {face, layer, direction, double}
   * @param {object} newState    {colors: number[][][]}
   * @param {number} durationMs
   * @returns {Promise<void>}
   */
  animateMove(moveData, newState, durationMs = 300) {
    return new Promise((resolve) => {
      const { face, layer, direction, double: isDouble } = moveData;
      const n = this.n;

      const axis = FACE_AXIS[face] || 'y';
      const sign = FACE_SIGN[face] !== undefined ? FACE_SIGN[face] : +1;
      const targetAngle = sign * (isDouble ? Math.PI : Math.PI / 2) * direction;

      // layer === -1 means whole-cube rotation (x/y/z) — animate every cubie.
      let layerCubies;
      if (layer === -1) {
        layerCubies = [...this.cubies];
      } else {
        const gridVal = layerGridIndex(face, layer, n);
        layerCubies = this.cubies.filter(({ gridPos: { ix, iy, iz } }) => {
          if (axis === 'x') return ix === gridVal;
          if (axis === 'y') return iy === gridVal;
          if (axis === 'z') return iz === gridVal;
          return false;
        });
      }

      // Diagnostic: log move details and cubie count so direction mismatches are visible.
      const moveName = moveData.name || `${face}${direction < 0 ? "'" : isDouble ? '2' : ''}`;
      console.log(
        `[cube3d] ${moveName}  axis=${axis}  angle=${(targetAngle * 180 / Math.PI).toFixed(0)}°` +
        `  layer=${layer}  cubies=${layerCubies.length}/${this.cubies.length}`
      );

      // Capture pre-animation grid positions for post-move verification logging.
      const prePos = layerCubies.map(c => ({ ix: c.gridPos.ix, iy: c.gridPos.iy, iz: c.gridPos.iz }));

      if (layerCubies.length === 0) {
        if (newState) {
          const colors = Array.isArray(newState) ? newState : newState.colors;
          if (colors) this.setState(colors);
        }
        resolve();
        return;
      }

      // Move cubies into pivot group
      for (const cubie of layerCubies) {
        this.pivotGroup.attach(cubie.mesh);
      }
      this.pivotGroup.rotation.set(0, 0, 0);

      const startTime = performance.now();

      const animate = () => {
        const elapsed = performance.now() - startTime;
        const t = Math.min(elapsed / durationMs, 1);
        const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; // ease-in-out quad
        const angle = targetAngle * eased;

        this.pivotGroup.rotation[axis] = angle;

        if (t < 1) {
          requestAnimationFrame(animate);
        } else {
          // Snap to final angle
          this.pivotGroup.rotation[axis] = targetAngle;

          // Move cubies back to scene
          for (const cubie of layerCubies) {
            this.scene.attach(cubie.mesh);
          }
          this.pivotGroup.rotation.set(0, 0, 0);

          // Snap positions to integers and reset quaternion.
          // Resetting the quaternion is critical: BoxGeometry material slots are
          // in local space, so a rotated mesh maps slot 2 (+Y local) to a
          // different world direction.  setState() assumes local axes == world
          // axes, so we must realign before repainting colors.
          for (const cubie of layerCubies) {
            const p = cubie.mesh.position;
            const gap = 1.02;
            const half = (n - 1) / 2;
            // Snap world position → grid position
            p.x = Math.round(p.x / gap) * gap;
            p.y = Math.round(p.y / gap) * gap;
            p.z = Math.round(p.z / gap) * gap;
            cubie.gridPos.ix = Math.round(p.x / gap + half);
            cubie.gridPos.iy = Math.round(p.y / gap + half);
            cubie.gridPos.iz = Math.round(p.z / gap + half);
            // Re-align local axes with world axes so setState() paints the
            // correct material slot for each visible face.
            cubie.mesh.quaternion.set(0, 0, 0, 1);
          }

          // Verification log: compare physical cubie movement with model state.
          // For a U CW move, expect e.g. (0,n-1,0)→(0,n-1,n-1) meaning L-back→L-front.
          // Open browser console, step through a U move, and check:
          //   1. Physical moves shown here match expected CW positions.
          //   2. Model U face row 0 (back row) printed below should shift consistently.
          const sampleSize = Math.min(4, layerCubies.length);
          const sampleMoves = layerCubies.slice(0, sampleSize).map((c, i) => {
            const p = prePos[i];
            return `(${p.ix},${p.iy},${p.iz})→(${c.gridPos.ix},${c.gridPos.iy},${c.gridPos.iz})`;
          }).join('  ');
          console.log(`  [cube3d] sample gridPos: ${sampleMoves}`);

          const colors0 = newState ? (Array.isArray(newState) ? newState : newState.colors) : null;
          if (colors0) {
            // Log the U face top row (row 0 = back row) from the model state.
            console.log(`  [cube3d] model U[0] (back): ${JSON.stringify(colors0[0][0])}`);
            console.log(`  [cube3d] model U[${n-1}] (front): ${JSON.stringify(colors0[0][n - 1])}`);
          }

          // Apply final state colors (accepts either a colors array or {colors} object)
          if (newState) {
            const colors = Array.isArray(newState) ? newState : newState.colors;
            if (colors) this.setState(colors);
          }

          resolve();
        }
      };

      requestAnimationFrame(animate);
    });
  }

  /**
   * Animate a full sequence of moves.
   * @param {object[]} moveDataArray   array of move objects
   * @param {object[]} stateArray      array of state objects (length = moveDataArray.length + 1; [0] is initial)
   * @param {number}   durationMs      animation duration per move
   * @param {number}   delayMs         pause between moves
   * @returns {Promise<void>}
   */
  async animateMoveSequence(moveDataArray, stateArray, durationMs = 300, delayMs = 80) {
    for (let i = 0; i < moveDataArray.length; i++) {
      if (this._stopAnimation) break;
      const newState = stateArray[i + 1] || null;
      await this.animateMove(moveDataArray[i], newState, durationMs);
      if (delayMs > 0) {
        await _delay(delayMs);
      }
    }
  }

  /**
   * Stop any in-progress animation sequence.
   */
  stopAnimation() {
    this._stopAnimation = true;
    setTimeout(() => { this._stopAnimation = false; }, 50);
  }

  /**
   * Cleanly destroy this renderer (remove canvas, stop observer).
   */
  destroy() {
    if (this._resizeObserver) this._resizeObserver.disconnect();
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode) {
      this.renderer.domElement.parentNode.removeChild(this.renderer.domElement);
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function _delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/**
 * Generate the solved-state face colors for an n×n cube (for initial construction).
 * Returns a (6 × n × n) nested array where faceColors[f][r][c] === f.
 */
function _solvedFaceColors(n) {
  const result = [];
  for (let f = 0; f < 6; f++) {
    const face = [];
    for (let r = 0; r < n; r++) {
      const row = [];
      for (let c = 0; c < n; c++) row.push(f);
      face.push(row);
    }
    result.push(face);
  }
  return result;
}

// Expose globally
window.RubiksCube3D = RubiksCube3D;
window.FACE_COLORS = FACE_COLORS;
