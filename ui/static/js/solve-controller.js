/**
 * solve-controller.js — SolveController
 *
 * Manages the Solve tab: puzzle selection, parameter input, job submission,
 * result loading, and playback (step/play/back/forward).
 *
 * Depends on globals: RubiksCube3D (cube3d.js), Megaminx3D (megaminx3d.js),
 *   SkewbUltimate3D (skewb3d.js), PUZZLE_DISPLAY_NAMES (colors.js),
 *   delay / pollJob / showNotify / setSpinner (app.js)
 */

class SolveController {
  constructor() {
    // Active 3D renderer — only one is non-null at a time
    this.cube3d      = null;
    this.megaminx3d  = null;
    this.skewb3d     = null;

    // Which puzzle type is active
    this.isMegaminx = false;
    this.isSkewb    = false;
    this.currentN   = 3;

    // Dropdown metadata
    this.moveLimits       = {};
    this.availableSolvers = {};
    this._allSolvers      = [];

    // Playback state
    this.allMoves     = []; // combined scramble + solve moves
    this.allStates    = []; // states (length = allMoves.length + 1)
    this.currentStep  = 0;
    this.scrambleCount = 0;
    this.playing      = false;
    this.playTimer    = null;
    this._isAnimating = false;

    this._bindUI();
    this._initDropdowns();
  }

  // -------------------------------------------------------------------------
  // Initialisation
  // -------------------------------------------------------------------------

  async _initDropdowns() {
    const [puzzles, solvers, moveLimits, availableSolvers] = await Promise.all([
      fetch('/api/puzzles').then(r => r.json()),
      fetch('/api/solvers').then(r => r.json()),
      fetch('/api/move_limits').then(r => r.ok ? r.json() : {}).catch(() => ({})),
      fetch('/api/available_solvers').then(r => r.ok ? r.json() : {}).catch(() => ({})),
    ]);

    this.moveLimits       = moveLimits;
    this.availableSolvers = availableSolvers;
    this._allSolvers      = solvers;

    const pd = document.getElementById('solve-puzzle');
    puzzles.forEach(p => {
      const opt = document.createElement('option');
      opt.value       = p;
      opt.textContent = PUZZLE_DISPLAY_NAMES[p] ?? p.toUpperCase();
      pd.appendChild(opt);
    });
    pd.value = '3x3';

    const sd = document.getElementById('solve-solver');
    solvers.forEach(s => {
      const opt = document.createElement('option');
      opt.value       = s;
      opt.textContent = s.charAt(0).toUpperCase() + s.slice(1);
      sd.appendChild(opt);
    });

    this._reinitRenderer();
    pd.addEventListener('change', () => { this._reinitRenderer(); this._updateSolverDropdown(); });
    sd.addEventListener('change', () => this._updateSolverParams());
    this._updateSolverDropdown();
    this._updateSolverParams();
  }

  _updateSolverDropdown() {
    const puzzle    = document.getElementById('solve-puzzle').value;
    const available = this.availableSolvers[puzzle] || this._allSolvers || [];
    const sd        = document.getElementById('solve-solver');

    Array.from(sd.options).forEach(opt => {
      const ok   = available.includes(opt.value);
      opt.disabled = !ok;
      opt.title    = ok ? '' : 'No trained model found for this puzzle';
    });

    if (!available.includes(sd.value)) {
      const first = Array.from(sd.options).find(o => !o.disabled);
      if (first) sd.value = first.value;
    }

    this._updateSolverParams();
  }

  _updateSolverParams() {
    const solver = document.getElementById('solve-solver').value;
    const isMcts = solver === 'mcts';
    document.getElementById('solve-max-gen-label').textContent =
      isMcts ? 'Time Limit (s)' : 'Max Generations';
    document.getElementById('solve-max-gen').value = isMcts ? 30 : 2000;
    const popGroup = document.getElementById('solve-pop-size-group');
    if (popGroup) popGroup.style.display = isMcts ? 'none' : '';
  }

  // -------------------------------------------------------------------------
  // 3D renderer management
  // -------------------------------------------------------------------------

  /**
   * Destroy any existing renderer and create the correct one for the currently
   * selected puzzle. Called on startup and every time the puzzle dropdown changes.
   */
  _reinitRenderer() {
    const puzzle = document.getElementById('solve-puzzle').value;

    this.isMegaminx = puzzle === 'megaminx';
    this.isSkewb    = puzzle === 'skewb_ultimate';

    const container = document.getElementById('cube-canvas-container');
    container.innerHTML = '';

    // Destroy stale renderers
    if (this.cube3d)     { this.cube3d.destroy();     this.cube3d     = null; }
    if (this.megaminx3d) { this.megaminx3d.destroy();  this.megaminx3d = null; }
    if (this.skewb3d)    { this.skewb3d.destroy();     this.skewb3d    = null; }

    if (this.isMegaminx) {
      this.megaminx3d = new Megaminx3D(container);
      // Show solved-state colors on startup
      this.megaminx3d.setState(
        Array.from({ length: 12 }, (_, f) => Array(11).fill(f))
      );
    } else if (this.isSkewb) {
      this.skewb3d = new SkewbUltimate3D(container);
      // Show solved-state colors on startup (piece i in slot i, orientation 0)
      this.skewb3d.setState(
        Array.from({ length: 14 }, (_, i) => i < 8 ? [i, 0] : [i - 8, 0])
      );
    } else {
      const n     = parseInt(puzzle[0], 10) || 3;
      this.currentN = n;
      this.cube3d = new RubiksCube3D(container, n);
    }

    this._resetPlayback();
  }

  // -------------------------------------------------------------------------
  // UI bindings
  // -------------------------------------------------------------------------

  _bindUI() {
    document.getElementById('scramble-depth-slider').addEventListener('input', e => {
      document.getElementById('scramble-depth-val').textContent = e.target.value;
    });

    document.getElementById('solve-btn').addEventListener('click', () => this._startSolve());

    document.getElementById('play-pause-btn').addEventListener('click',  () => this._togglePlay());
    document.getElementById('step-back-btn').addEventListener('click',   () => this._stepBack());
    document.getElementById('step-fwd-btn').addEventListener('click',    () => this._stepForward());
    document.getElementById('step-start-btn').addEventListener('click',  () => this._goToStart());
    document.getElementById('step-end-btn').addEventListener('click',    () => this._goToEnd());

    document.getElementById('speed-slider').addEventListener('input', () => {
      if (this.playing) {
        clearInterval(this.playTimer);
        this._schedulePlay();
      }
    });
  }

  // -------------------------------------------------------------------------
  // Solve job
  // -------------------------------------------------------------------------

  async _startSolve() {
    const puzzle        = document.getElementById('solve-puzzle').value;
    const solver        = document.getElementById('solve-solver').value;
    const scrambleDepth = parseInt(document.getElementById('scramble-depth-slider').value, 10);
    const seed          = parseInt(document.getElementById('solve-seed').value, 10) || 42;
    const maxGen        = parseInt(document.getElementById('solve-max-gen').value, 10) || 200;
    const popSize       = parseInt(document.getElementById('solve-pop-size').value, 10) || 100;

    this._stopPlay();
    setSpinner(true);
    document.getElementById('solve-btn').disabled = true;

    try {
      const resp = await fetch('/api/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ puzzle, solver, scramble_depth: scrambleDepth,
                               seed, max_generations: maxGen, population_size: popSize }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const { job_id } = await resp.json();

      const result = await pollJob(job_id, 'solve');
      if (result.status === 'error') throw new Error(result.error || 'Unknown error');

      this._loadResult(result);
      showNotify(result.solved ? 'Solved!' : 'Best attempt shown', result.solved ? 'success' : 'error');
    } catch (err) {
      showNotify('Error: ' + err.message, 'error');
      console.error(err);
    } finally {
      setSpinner(false);
      document.getElementById('solve-btn').disabled = false;
    }
  }

  _loadResult(result) {
    this.scrambleCount = (result.scramble_moves  || []).length;
    this.allMoves      = [...(result.scramble_moves  || []), ...(result.solve_moves  || [])];
    this.allStates     = [...(result.scramble_states || []), ...(result.solve_states || []).slice(1)];
    this.currentStep   = 0;

    document.getElementById('stat-iterations').textContent = result.iterations ?? '-';
    document.getElementById('stat-time').textContent =
      result.solve_time != null ? result.solve_time.toFixed(1) + 's' : '-';
    const solvedEl = document.getElementById('stat-solved');
    solvedEl.textContent = result.solved ? 'YES' : 'NO';
    solvedEl.className   = 'stat-value ' + (result.solved ? 'success' : 'danger');
    document.getElementById('stat-moves').textContent = (result.solve_moves || []).length;

    this._buildMoveList();
    this._renderStep(this.scrambleCount);
    this._updatePlaybackButtons();
  }

  // -------------------------------------------------------------------------
  // Move list
  // -------------------------------------------------------------------------

  _buildMoveList() {
    const list = document.getElementById('move-list');
    list.innerHTML = '';

    for (let i = this.scrambleCount; i < this.allMoves.length; i++) {
      const move = this.allMoves[i];
      const item = document.createElement('div');
      item.className  = 'move-item';
      item.dataset.step = i + 1;

      const numEl  = document.createElement('span');
      numEl.className  = 'move-num';
      numEl.textContent = i - this.scrambleCount + 1;

      const nameEl = document.createElement('span');
      nameEl.className  = 'move-name';
      nameEl.textContent = move.name;

      item.appendChild(numEl);
      item.appendChild(nameEl);
      item.addEventListener('click', () => this._goToStep(i + 1));
      list.appendChild(item);
    }
  }

  // -------------------------------------------------------------------------
  // Playback — step rendering
  // -------------------------------------------------------------------------

  /** Instantly jump to a step — resets cubie positions for Rubik's cubes. */
  _renderStep(step) {
    this.currentStep = step;
    const state = this.allStates[step];
    if (!state) return;

    if (this.isMegaminx && this.megaminx3d) {
      this.megaminx3d.resetTransforms();
      this.megaminx3d.setState(state.colors);
    } else if (this.isSkewb && this.skewb3d) {
      this.skewb3d.resetTransforms();
      this.skewb3d.setState(state.colors);
    } else if (this.cube3d) {
      // Must reset cubie world-positions before setState so gridPos remains valid
      this.cube3d.resetPositions();
      this.cube3d.setState(state.colors);
    }

    this._updateStepUI(step);
  }

  _updateStepUI(step) {
    const totalSolve = this.allMoves.length - this.scrambleCount;
    const phase      = step < this.scrambleCount ? 'scramble' : 'solve';
    const solveStep  = Math.max(0, step - this.scrambleCount);
    document.getElementById('step-info').innerHTML =
      `Solve <b>${solveStep}</b>/${totalSolve} <span class="phase-badge ${phase}">${phase}</span>`;

    document.querySelectorAll('.move-item').forEach(el => {
      el.classList.toggle('active', parseInt(el.dataset.step, 10) === step);
    });

    const activeItem = document.querySelector('.move-item.active');
    if (activeItem) activeItem.scrollIntoView({ block: 'nearest' });
  }

  _moveDuration() {
    const v = parseInt(document.getElementById('speed-slider').value, 10) || 2;
    return Math.round(500 - (v - 1) * (490 / 9));
  }

  _goToStep(step) {
    this._stopPlay();
    this._isAnimating = false;
    this._renderStep(Math.max(0, Math.min(step, this.allMoves.length)));
  }

  _goToStart() { this._goToStep(0); }
  _goToEnd()   { this._goToStep(this.allMoves.length); }

  async _stepBack() {
    if (this._isAnimating || this.currentStep <= 0) return;
    const step     = this.currentStep - 1;
    const moveData = this.allMoves[step];
    const prevState = this.allStates[step];
    if (!prevState) return;

    // Animated step back for Rubik's, Megaminx, and Skewb
    const reversed = { ...moveData, direction: -moveData.direction };
    if (this.cube3d) {
      this._isAnimating = true;
      await this.cube3d.animateMove(reversed, prevState.colors, this._moveDuration());
      this._isAnimating = false;
      this.currentStep  = step;
      this._updateStepUI(step);
    } else if (this.isMegaminx && this.megaminx3d) {
      this._isAnimating = true;
      await this.megaminx3d.animateMove(reversed, prevState.colors, this._moveDuration());
      this._isAnimating = false;
      this.currentStep  = step;
      this._updateStepUI(step);
    } else if (this.isSkewb && this.skewb3d) {
      this._isAnimating = true;
      await this.skewb3d.animateMove(reversed, prevState.colors, this._moveDuration());
      this._isAnimating = false;
      this.currentStep  = step;
      this._updateStepUI(step);
    } else {
      this._renderStep(step);
    }
  }

  async _stepForward() {
    if (this._isAnimating || this.currentStep >= this.allMoves.length) return;
    const step     = this.currentStep + 1;
    const moveData = this.allMoves[step - 1];
    const newState = this.allStates[step];
    if (!moveData || !newState) return;

    if (this.cube3d) {
      this._isAnimating = true;
      await this.cube3d.animateMove(moveData, newState.colors, this._moveDuration());
      this._isAnimating = false;
      this.currentStep  = step;
      this._updateStepUI(step);
    } else if (this.isMegaminx && this.megaminx3d) {
      this._isAnimating = true;
      await this.megaminx3d.animateMove(moveData, newState.colors, this._moveDuration());
      this._isAnimating = false;
      this.currentStep  = step;
      this._updateStepUI(step);
    } else if (this.isSkewb && this.skewb3d) {
      this._isAnimating = true;
      await this.skewb3d.animateMove(moveData, newState.colors, this._moveDuration());
      this._isAnimating = false;
      this.currentStep  = step;
      this._updateStepUI(step);
    } else {
      this._renderStep(step);
    }
  }

  // -------------------------------------------------------------------------
  // Playback — play/pause/auto
  // -------------------------------------------------------------------------

  _togglePlay() {
    if (this.playing) this._stopPlay(); else this._startPlay();
  }

  _startPlay() {
    if (this.currentStep >= this.allMoves.length || this.currentStep < this.scrambleCount) {
      this._renderStep(this.scrambleCount);
    }
    this.playing = true;
    document.getElementById('play-pause-btn').textContent = '⏸';
    this._playNextAnimated();
  }

  _stopPlay() {
    this.playing = false;
    clearInterval(this.playTimer);
    document.getElementById('play-pause-btn').textContent = '▶';
  }

  async _playNextAnimated() {
    while (this.playing && this.currentStep < this.allMoves.length) {
      const step     = this.currentStep + 1;
      const moveData = this.allMoves[step - 1];
      const newState = this.allStates[step];
      if (!moveData || !newState) break;

      const durationMs = this._moveDuration();

      if (this.cube3d) {
        await this.cube3d.animateMove(moveData, newState.colors, durationMs);
      } else if (this.isMegaminx && this.megaminx3d) {
        await this.megaminx3d.animateMove(moveData, newState.colors, durationMs);
      } else if (this.isSkewb && this.skewb3d) {
        await this.skewb3d.animateMove(moveData, newState.colors, durationMs);
      } else {
        this._renderStep(step);
        await delay(durationMs);
      }

      this.currentStep = step;
      this._updateStepUI(step);

      if (this.playing) await delay(80);
    }
    if (this.playing) this._stopPlay();
  }

  // -------------------------------------------------------------------------
  // Reset
  // -------------------------------------------------------------------------

  _resetPlayback() {
    this._stopPlay();
    this.allMoves     = [];
    this.allStates    = [];
    this.currentStep  = 0;
    this.scrambleCount = 0;
    document.getElementById('move-list').innerHTML = '';
    document.getElementById('step-info').textContent = '';
    ['stat-iterations', 'stat-time', 'stat-moves'].forEach(id => {
      document.getElementById(id).textContent = '-';
    });
    const solvedEl = document.getElementById('stat-solved');
    solvedEl.textContent = '-';
    solvedEl.className   = 'stat-value';
  }

  _updatePlaybackButtons() {
    const hasData = this.allMoves.length > 0;
    ['play-pause-btn', 'step-back-btn', 'step-fwd-btn',
     'step-start-btn', 'step-end-btn'].forEach(id => {
      document.getElementById(id).disabled = !hasData;
    });
  }
}
