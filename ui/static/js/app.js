/**
 * app.js — Bootstrap and shared utilities
 *
 * Provides globals used by all other modules:
 *   delay, pollJob, showNotify, setSpinner, initTabs
 *
 * Script load order (index.html):
 *   1. colors.js          — color palettes and puzzle constants
 *   2. cube3d.js          — RubiksCube3D (Three.js)
 *   3. megaminx3d.js      — Megaminx3D (Three.js dodecahedron)
 *   4. skewb3d.js         — SkewbUltimate3D (Three.js cube)
 *   5. solve-controller.js — SolveController
 *   6. train-controller.js — TrainController
 *   7. app.js             — this file (bootstrap)
 */

// ---------------------------------------------------------------------------
// Async utilities
// ---------------------------------------------------------------------------

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
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

// ---------------------------------------------------------------------------
// Notification toast
// ---------------------------------------------------------------------------

function showNotify(msg, type = 'info') {
  const el = document.getElementById('notify');
  el.textContent = msg;
  el.className   = `notify show ${type}`;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove('show'), 3500);
}

// ---------------------------------------------------------------------------
// Global spinner
// ---------------------------------------------------------------------------

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

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      panels.forEach(p  => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(btn.dataset.tab).classList.add('active');
    });
  });
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  window._solveCtrl = new SolveController();
  window._trainCtrl = new TrainController();
});
