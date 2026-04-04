/**
 * train-controller.js — TrainController
 *
 * Manages the Train tab: puzzle/config inputs, job submission, and
 * fitness-over-generations chart rendering.
 *
 * Depends on globals: PUZZLE_DISPLAY_NAMES (colors.js),
 *   pollJob / showNotify / setSpinner (app.js), Chart (CDN)
 */

class TrainController {
  constructor() {
    this.chart = null;
    this._initDropdowns();
    this._bindUI();
  }

  // -------------------------------------------------------------------------
  // Initialisation
  // -------------------------------------------------------------------------

  async _initDropdowns() {
    const puzzles = await fetch('/api/puzzles').then(r => r.json());
    const pd = document.getElementById('train-puzzle');
    puzzles.forEach(p => {
      const opt = document.createElement('option');
      opt.value       = p;
      opt.textContent = PUZZLE_DISPLAY_NAMES[p] ?? p.toUpperCase();
      pd.appendChild(opt);
    });
    pd.value = '3x3';
  }

  _bindUI() {
    document.getElementById('train-btn').addEventListener('click', () => this._startTrain());
  }

  // -------------------------------------------------------------------------
  // Train job
  // -------------------------------------------------------------------------

  async _startTrain() {
    const puzzle   = document.getElementById('train-puzzle').value;
    const epochs   = parseInt(document.getElementById('train-epochs').value,   10) || 200;
    const scramble = parseInt(document.getElementById('train-scramble').value, 10) || 5;
    const seed     = parseInt(document.getElementById('train-seed').value,     10) || 42;

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
      solvedEl.className   = 'stat-value ' + (result.solved ? 'success' : 'danger');

      showNotify('Training complete!', 'success');
    } catch (err) {
      showNotify('Error: ' + err.message, 'error');
      console.error(err);
    } finally {
      setSpinner(false);
      document.getElementById('train-btn').disabled = false;
    }
  }

  // -------------------------------------------------------------------------
  // Chart
  // -------------------------------------------------------------------------

  _renderChart(result) {
    const ctx     = document.getElementById('train-chart').getContext('2d');
    const best    = result.fitness_best_history    || [];
    const mean    = result.fitness_mean_history    || [];
    const control = result.control_fitness_history || [];
    const labels  = Array.from({ length: best.length }, (_, i) => i + 1);

    if (this.chart) this.chart.destroy();

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label:           'Best Fitness',
            data:            best,
            borderColor:     '#58a6ff',
            backgroundColor: 'rgba(88,166,255,0.08)',
            borderWidth:     2,
            pointRadius:     0,
            tension:         0.3,
            fill:            true,
          },
          {
            label:        'Mean Fitness',
            data:         mean,
            borderColor:  '#3fb950',
            borderWidth:  1.5,
            pointRadius:  0,
            tension:      0.3,
            borderDash:   [4, 3],
          },
          {
            label:       'Random Baseline',
            data:        control,
            borderColor: '#d29922',
            borderWidth: 1,
            pointRadius: 0,
            tension:     0.1,
            borderDash:  [2, 4],
          },
        ],
      },
      options: {
        responsive:          true,
        maintainAspectRatio: false,
        animation:           { duration: 400 },
        plugins: {
          legend:  { labels: { color: '#8b949e', font: { size: 12 } } },
          tooltip: {
            mode:            'index',
            intersect:       false,
            backgroundColor: '#161b22',
            borderColor:     '#30363d',
            borderWidth:     1,
            titleColor:      '#e6edf3',
            bodyColor:       '#8b949e',
          },
        },
        scales: {
          x: {
            title: { display: true, text: 'Generation', color: '#8b949e' },
            ticks: { color: '#8b949e', maxTicksLimit: 12 },
            grid:  { color: '#21262d' },
          },
          y: {
            title: { display: true, text: 'Fitness', color: '#8b949e' },
            ticks: { color: '#8b949e' },
            grid:  { color: '#21262d' },
          },
        },
      },
    });
  }
}
