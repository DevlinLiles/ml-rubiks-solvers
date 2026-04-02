/**
 * app.js — Main application logic for the Rubik's Cube ML Solver UI
 */

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function delay(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function pollJob(jobId, endpoint, intervalMs = 800) {
  while (true) {
    const resp = await fetch(`/api/${endpoint}/${jobId}`);
    if (!resp.ok) throw new Error(`Poll failed: ${resp.status}`);
    const data = await resp.json();
    if (data.status !== 'running') return data;
    await delay(intervalMs);
  }
}

function showNotify(msg, type = 'info') {
  const el = document.getElementById('notify');
  el.textContent = msg;
  el.className = `notify show ${type}`;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove('show'), 3500);
}

function setSpinner(active) {
  const sp = document.getElementById('global-spinner');
  const lb = document.getElementById('status-label');
  if (active) {
    sp.classList.add('active');
    lb.textContent = 'Running…';
  } else {
    sp.classList.remove('active');
    lb.textContent = '';
  }
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

function initTabs() {
  const buttons = document.querySelectorAll('.tab-btn');
  const panels  = document.querySelectorAll('.tab-panel');

  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      buttons.forEach((b) => b.classList.remove('active'));
      panels.forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.tab;
      document.getElementById(target).classList.add('active');
    });
  });
}

// ---------------------------------------------------------------------------
// Megaminx flat renderer (12-face net as colored grids)
// ---------------------------------------------------------------------------

const MEGA_COLORS = [
  '#FFFFFF','#FFD500','#009B48','#0046AD','#FF5800','#C41E3A',
  '#A020F0','#FF69B4','#00CED1','#8B4513','#7CFC00','#FF8C00',
  '#1a1a1a',
];

const MEGA_FACE_NAMES = ['U','BL','BR','L','R','DL','DR','DBL','DBR','DB','F','DF'];

function renderMegaminx(container, colors) {
  container.innerHTML = '';
  const wrapper = document.createElement('div');
  wrapper.className = 'megaminx-net';

  // Layout: 4 rows of faces in a net arrangement
  const layout = [
    [null,  0, null, null],   // row 0: U
    [ 4,    2,   1,   3 ],   // row 1: R F BL BR
    [ 6,    10,  5,   null], // row 2: DR F DL
    [null,  7,   9,   8 ],   // row 3: DBL DB DBR
  ];

  for (let row = 0; row < layout.length; row++) {
    const rowDiv = document.createElement('div');
    rowDiv.style.cssText = 'display:flex;gap:4px;justify-content:center;';

    for (let col = 0; col < layout[row].length; col++) {
      const faceIdx = layout[row][col];
      const cell = document.createElement('div');
      cell.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:1px;';

      if (faceIdx === null) {
        cell.style.width = '90px';
        rowDiv.appendChild(cell);
        continue;
      }

      const label = document.createElement('div');
      label.className = 'mega-face-label';
      label.textContent = MEGA_FACE_NAMES[faceIdx] || `F${faceIdx}`;
      cell.appendChild(label);

      const faceData = colors[faceIdx]; // shape (11,) or row data
      // Megaminx state shape is (12, 11)
      const grid = document.createElement('div');
      grid.style.cssText = 'display:grid;grid-template-columns:repeat(11,12px);gap:1px;';

      if (Array.isArray(faceData)) {
        for (const colorIdx of faceData) {
          const sq = document.createElement('div');
          sq.style.cssText = `width:12px;height:12px;border-radius:2px;background:${MEGA_COLORS[colorIdx] || '#333'};border:1px solid rgba(0,0,0,.3);`;
          grid.appendChild(sq);
        }
      }
      cell.appendChild(grid);
      rowDiv.appendChild(cell);
    }
    wrapper.appendChild(rowDiv);
  }

  container.appendChild(wrapper);
}

// ---------------------------------------------------------------------------
// SolveController
// ---------------------------------------------------------------------------

class SolveController {
  constructor() {
    this.cube3d = null;
    this.currentN = 3;
    this.isMegaminx = false;

    // Playback state
    this.allMoves  = [];  // combined scramble + solve moves
    this.allStates = [];  // combined states (length = allMoves.length + 1)
    this.currentStep = 0;
    this.scrambleCount = 0;
    this.playing = false;
    this.playTimer = null;

    this._bindUI();
    this._initDropdowns();
  }

  async _initDropdowns() {
    // Populate puzzle dropdown
    const [puzzles, solvers] = await Promise.all([
      fetch('/api/puzzles').then((r) => r.json()),
      fetch('/api/solvers').then((r) => r.json()),
    ]);

    const pd = document.getElementById('solve-puzzle');
    puzzles.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p === 'megaminx' ? 'Megaminx' : p.toUpperCase();
      pd.appendChild(opt);
    });
    pd.value = '3x3';

    const sd = document.getElementById('solve-solver');
    solvers.forEach((s) => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = s.charAt(0).toUpperCase() + s.slice(1);
      sd.appendChild(opt);
    });

    // Initialize cube
    this._reinitCube();
    pd.addEventListener('change', () => this._reinitCube());
    sd.addEventListener('change', () => this._updateSolverParams());
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

  _reinitCube() {
    const puzzle = document.getElementById('solve-puzzle').value;
    this.isMegaminx = puzzle === 'megaminx';
    const container = document.getElementById('cube-canvas-container');
    container.innerHTML = '';

    if (this.isMegaminx) {
      // Show placeholder megaminx net
      renderMegaminx(container, Array.from({ length: 12 }, (_, f) => Array(11).fill(f)));
      this.cube3d = null;
    } else {
      const n = parseInt(puzzle[0], 10) || 3;
      this.currentN = n;
      if (this.cube3d) this.cube3d.destroy();
      this.cube3d = new RubiksCube3D(container, n);
    }

    // Reset playback
    this._resetPlayback();
  }

  _bindUI() {
    document.getElementById('scramble-depth-slider').addEventListener('input', (e) => {
      document.getElementById('scramble-depth-val').textContent = e.target.value;
    });

    document.getElementById('solve-btn').addEventListener('click', () => this._startSolve());

    document.getElementById('play-pause-btn').addEventListener('click', () => this._togglePlay());
    document.getElementById('step-back-btn').addEventListener('click', () => this._stepBack());
    document.getElementById('step-fwd-btn').addEventListener('click', () => this._stepForward());
    document.getElementById('step-start-btn').addEventListener('click', () => this._goToStart());
    document.getElementById('step-end-btn').addEventListener('click', () => this._goToEnd());

    document.getElementById('speed-slider').addEventListener('input', () => {
      if (this.playing) {
        clearInterval(this.playTimer);
        this._schedulePlay();
      }
    });
  }

  async _startSolve() {
    const puzzle       = document.getElementById('solve-puzzle').value;
    const solver       = document.getElementById('solve-solver').value;
    const scrambleDepth= parseInt(document.getElementById('scramble-depth-slider').value, 10);
    const seed         = parseInt(document.getElementById('solve-seed').value, 10) || 42;
    const maxGen       = parseInt(document.getElementById('solve-max-gen').value, 10) || 200;
    const popSize      = parseInt(document.getElementById('solve-pop-size').value, 10) || 100;

    this._stopPlay();
    setSpinner(true);
    document.getElementById('solve-btn').disabled = true;

    try {
      const resp = await fetch('/api/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          puzzle,
          solver,
          scramble_depth: scrambleDepth,
          seed,
          max_generations: maxGen,
          population_size: popSize,
        }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const { job_id } = await resp.json();

      const result = await pollJob(job_id, 'solve');

      if (result.status === 'error') {
        throw new Error(result.error || 'Unknown error');
      }

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
    // Combine scramble + solve sequences
    this.scrambleCount = (result.scramble_moves || []).length;
    this.allMoves  = [...(result.scramble_moves || []), ...(result.solve_moves || [])];
    this.allStates = [...(result.scramble_states || []), ...(result.solve_states || []).slice(1)];
    this.currentStep = 0;

    // Update stats
    document.getElementById('stat-iterations').textContent = result.iterations ?? '-';
    document.getElementById('stat-time').textContent =
      result.solve_time != null ? result.solve_time.toFixed(1) + 's' : '-';
    const solvedEl = document.getElementById('stat-solved');
    solvedEl.textContent = result.solved ? 'YES' : 'NO';
    solvedEl.className = 'stat-value ' + (result.solved ? 'success' : 'danger');
    document.getElementById('stat-moves').textContent =
      (result.solve_moves || []).length;

    // Build move list
    this._buildMoveList();

    // Show scrambled state immediately (skip scramble animation)
    this._renderStep(this.scrambleCount);
    this._updatePlaybackButtons();
  }

  _buildMoveList() {
    const list = document.getElementById('move-list');
    list.innerHTML = '';

    for (let i = 0; i < this.allMoves.length; i++) {
      const move = this.allMoves[i];
      const item = document.createElement('div');
      item.className = 'move-item' + (i < this.scrambleCount ? ' scramble' : '');
      item.dataset.step = i + 1;

      const numEl = document.createElement('span');
      numEl.className = 'move-num';
      numEl.textContent = i + 1;

      const nameEl = document.createElement('span');
      nameEl.className = 'move-name';
      nameEl.textContent = move.name;

      item.appendChild(numEl);
      item.appendChild(nameEl);
      item.addEventListener('click', () => this._goToStep(i + 1));
      list.appendChild(item);
    }
  }

  _renderStep(step) {
    this.currentStep = step;
    const state = this.allStates[step];
    if (!state) return;

    if (this.isMegaminx) {
      const container = document.getElementById('cube-canvas-container');
      renderMegaminx(container, state.colors);
    } else if (this.cube3d) {
      this.cube3d.setState(state.colors);
    }

    // Update step info
    const total = this.allMoves.length;
    const phase = step <= this.scrambleCount ? 'scramble' : 'solve';
    document.getElementById('step-info').innerHTML =
      `Step <b>${step}</b>/${total} <span class="phase-badge ${phase}">${phase}</span>`;

    // Highlight active move in list
    document.querySelectorAll('.move-item').forEach((el) => {
      el.classList.toggle('active', parseInt(el.dataset.step, 10) === step);
    });

    // Scroll active move into view
    const activeItem = document.querySelector('.move-item.active');
    if (activeItem) activeItem.scrollIntoView({ block: 'nearest' });
  }

  async _goToStep(step) {
    this._stopPlay();
    const clamped = Math.max(0, Math.min(step, this.allMoves.length));
    this._renderStep(clamped);
  }

  _goToStart() { this._goToStep(0); }
  _goToEnd()   { this._goToStep(this.allMoves.length); }

  _stepBack()    { this._goToStep(this.currentStep - 1); }
  _stepForward() { this._goToStep(this.currentStep + 1); }

  _togglePlay() {
    if (this.playing) {
      this._stopPlay();
    } else {
      this._startPlay();
    }
  }

  _startPlay() {
    // If at end or before the scramble section, reset to start of solve
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
      const step = this.currentStep + 1;
      const moveData = this.allMoves[step - 1];
      const newState = this.allStates[step];
      if (!moveData || !newState) break;

      // speed 1-10 → animation 500ms-10ms
      const speedVal = parseInt(document.getElementById('speed-slider').value, 10) || 2;
      const durationMs = Math.round(500 - (speedVal - 1) * (490 / 9));

      if (!this.isMegaminx && this.cube3d) {
        await this.cube3d.animateMove(moveData, newState.colors, durationMs);
      } else {
        this._renderStep(step);
        await delay(durationMs);
      }

      // Update step tracking and UI without re-setting cube state (already done by animateMove)
      this.currentStep = step;
      const total = this.allMoves.length;
      const phase = step <= this.scrambleCount ? 'scramble' : 'solve';
      document.getElementById('step-info').innerHTML =
        `Step <b>${step}</b>/${total} <span class="phase-badge ${phase}">${phase}</span>`;
      document.querySelectorAll('.move-item').forEach((el) => {
        el.classList.toggle('active', parseInt(el.dataset.step, 10) === step);
      });
      const activeItem = document.querySelector('.move-item.active');
      if (activeItem) activeItem.scrollIntoView({ block: 'nearest' });

      // Brief pause between moves
      if (this.playing) await delay(120);
    }
    if (this.playing) this._stopPlay();
  }

  _resetPlayback() {
    this._stopPlay();
    this.allMoves  = [];
    this.allStates = [];
    this.currentStep = 0;
    this.scrambleCount = 0;
    document.getElementById('move-list').innerHTML = '';
    document.getElementById('step-info').textContent = '';
    ['stat-iterations', 'stat-time', 'stat-moves'].forEach((id) => {
      document.getElementById(id).textContent = '-';
    });
    const solvedEl = document.getElementById('stat-solved');
    solvedEl.textContent = '-';
    solvedEl.className = 'stat-value';
  }

  _updatePlaybackButtons() {
    const hasData = this.allMoves.length > 0;
    ['play-pause-btn', 'step-back-btn', 'step-fwd-btn',
     'step-start-btn', 'step-end-btn'].forEach((id) => {
      document.getElementById(id).disabled = !hasData;
    });
  }
}

// ---------------------------------------------------------------------------
// TrainController
// ---------------------------------------------------------------------------

class TrainController {
  constructor() {
    this.chart = null;
    this._initDropdowns();
    this._bindUI();
  }

  async _initDropdowns() {
    const puzzles = await fetch('/api/puzzles').then((r) => r.json());
    const pd = document.getElementById('train-puzzle');
    puzzles.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p === 'megaminx' ? 'Megaminx' : p.toUpperCase();
      pd.appendChild(opt);
    });
    pd.value = '3x3';
  }

  _bindUI() {
    document.getElementById('train-btn').addEventListener('click', () => this._startTrain());
  }

  async _startTrain() {
    const puzzle       = document.getElementById('train-puzzle').value;
    const epochs       = parseInt(document.getElementById('train-epochs').value, 10) || 200;
    const scramble     = parseInt(document.getElementById('train-scramble').value, 10) || 5;
    const seed         = parseInt(document.getElementById('train-seed').value, 10) || 42;

    setSpinner(true);
    document.getElementById('train-btn').disabled = true;

    try {
      const resp = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ puzzle, solver: 'genetic', epochs, seed, scramble_depth: scramble }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const { job_id } = await resp.json();

      const result = await pollJob(job_id, 'train');

      if (result.status === 'error') throw new Error(result.error);

      this._renderChart(result);

      document.getElementById('train-stat-gens').textContent = result.iterations ?? '-';
      document.getElementById('train-stat-time').textContent =
        result.solve_time != null ? result.solve_time.toFixed(1) + 's' : '-';
      const solvedEl = document.getElementById('train-stat-solved');
      solvedEl.textContent = result.solved ? 'YES' : 'NO';
      solvedEl.className = 'stat-value ' + (result.solved ? 'success' : 'danger');

      showNotify('Training complete!', 'success');
    } catch (err) {
      showNotify('Error: ' + err.message, 'error');
      console.error(err);
    } finally {
      setSpinner(false);
      document.getElementById('train-btn').disabled = false;
    }
  }

  _renderChart(result) {
    const ctx = document.getElementById('train-chart').getContext('2d');
    const best    = result.fitness_best_history    || [];
    const mean    = result.fitness_mean_history    || [];
    const control = result.control_fitness_history || [];

    const labels = Array.from({ length: best.length }, (_, i) => i + 1);

    if (this.chart) this.chart.destroy();

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Best Fitness',
            data: best,
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88,166,255,0.08)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: true,
          },
          {
            label: 'Mean Fitness',
            data: mean,
            borderColor: '#3fb950',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            tension: 0.3,
            borderDash: [4, 3],
          },
          {
            label: 'Random Baseline',
            data: control,
            borderColor: '#d29922',
            backgroundColor: 'transparent',
            borderWidth: 1,
            pointRadius: 0,
            tension: 0.1,
            borderDash: [2, 4],
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: {
          legend: {
            labels: { color: '#8b949e', font: { size: 12 } },
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            backgroundColor: '#161b22',
            borderColor: '#30363d',
            borderWidth: 1,
            titleColor: '#e6edf3',
            bodyColor: '#8b949e',
          },
        },
        scales: {
          x: {
            title: { display: true, text: 'Generation', color: '#8b949e' },
            ticks: { color: '#8b949e', maxTicksLimit: 12 },
            grid: { color: '#21262d' },
          },
          y: {
            title: { display: true, text: 'Fitness', color: '#8b949e' },
            ticks: { color: '#8b949e' },
            grid: { color: '#21262d' },
          },
        },
      },
    });
  }
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  window._solveCtrl = new SolveController();
  window._trainCtrl = new TrainController();
});
