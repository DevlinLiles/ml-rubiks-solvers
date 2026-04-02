# ml-rubiks-solvers

A multi-algorithm machine learning suite for solving Rubik's cubes and similar twisty puzzles. Supports five puzzle types, seven solver algorithms, and composable multi-stage pipelines. All neural network training runs natively on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

---

## Puzzles

| Puzzle | Faces | Legal Moves | Move Limit (HTM) | Permutations |
|--------|-------|-------------|-------------------|--------------|
| 2×2 Pocket Cube | 6 | 9 | 11 | ~3.7M |
| 3×3 Standard Cube | 6 | 18 | 20 | ~4.3×10¹⁹ |
| 4×4 Revenge Cube | 6 | 45 | 40 | ~7.4×10⁴⁵ |
| 5×5 Professor Cube | 6 | 45 | 60 | ~2.8×10⁷⁴ |
| Megaminx | 12 | 24 | 70 | ~10⁶⁸ |

All puzzle states are **immutable** — every `apply_move()` call returns a new instance, making them safe to share across solvers and pipeline stages.

---

## Solvers

### Genetic Algorithm (`--solver genetic`)
Uses [DEAP](https://deap.readthedocs.io/) to evolve variable-length chromosomes of move sequences. Each individual is a list of move indices; fitness is the number of correctly-placed facelets.

Features:
- Three fitness strategies: `misplaced_facelets`, `weighted_layer`, `solved_bonus`
- **Stagnation detection** — after N generations without improvement, injects fresh random individuals to escape local optima
- **Control individual** — a fresh random chromosome is evaluated every generation as a population baseline
- Per-generation tracking of best fitness, mean fitness, and control fitness

### CNN Value Network (`--solver cnn`)
A fully-connected value network (MLX) trained to predict scramble depth. Used as a heuristic for greedy best-first search.

### Policy Network (`--solver policy`)
An imitation-learning policy network (MLX) trained on optimal inverse moves from randomly scrambled states (autodidactic iteration). Predicts the next best move directly.

### DQN (`--solver dqn`)
A Dueling Deep Q-Network (MLX) trained via self-play in a lightweight puzzle environment. Uses a replay buffer and a separate target network with periodic syncing.

### MCTS (`--solver mcts`)
Monte Carlo Tree Search with UCB1 exploration. Supports a configurable simulation budget and optional time limit.

### Transformer (stub)
Architecture scaffolded for sequence-to-sequence move prediction. Not yet trained.

---

## Pipelines

### `SolverChain`
Runs solvers sequentially, passing the (possibly partially-solved) puzzle state from one stage to the next. Useful for coarse-to-fine strategies, e.g. a fast genetic stage followed by a deeper MCTS stage.

```python
from rubiks_solve.pipeline.chain import SolverChain, StageConfig

chain = SolverChain(
    puzzle_type=Cube3x3,
    stages=[
        StageConfig(solver=genetic_solver, move_budget=20),
        StageConfig(solver=mcts_solver,    move_budget=20),
    ],
)
result = chain.solve(scrambled_puzzle)
```

### `EnsembleSolver`
Runs solvers concurrently (threaded) and returns the best result according to a voting strategy. Solved results are always preferred over unsolved ones.

```python
from rubiks_solve.pipeline.ensemble import EnsembleSolver, VotingStrategy

ensemble = EnsembleSolver(
    puzzle_type=Cube3x3,
    solvers=[genetic_solver, mcts_solver],
    strategy=VotingStrategy.FASTEST_SOLVE,
    timeout_seconds=30.0,
)
result = ensemble.solve(scrambled_puzzle)
```

Voting strategies: `FASTEST_SOLVE`, `SHORTEST_SOLUTION`, `CONFIDENCE_WEIGHTED`.

---

## CLI

```bash
# Train a solver
rubiks-train --solver genetic --puzzle 3x3 --epochs 10000
rubiks-train --solver cnn     --puzzle 3x3 --epochs 100 --n-train 10000
rubiks-train --solver dqn     --puzzle 2x2 --epochs 50

# Solve a puzzle
rubiks-solve --solver genetic --puzzle 3x3

# Benchmark solvers
rubiks-benchmark --puzzle 3x3

# Visualize a solution
rubiks-viz --puzzle 3x3
```

Supported `--puzzle` values: `2x2`, `3x3`, `4x4`, `5x5`, `megaminx`  
Supported `--solver` values: `genetic`, `cnn`, `policy`, `dqn`, `mcts`

Training outputs (CSV metrics + PNG plots) are saved to `models/metrics/`.

---

## Project Structure

```
rubiks_solve/
  core/           # Puzzle engines (immutable state, move definitions)
  encoding/       # State encoders: one-hot, cubie
  solvers/        # Solver implementations (genetic, cnn, policy, dqn, mcts, transformer)
  pipeline/       # SolverChain, EnsembleSolver, Router
  training/       # Data generation, curriculum, checkpointing, metrics
  utils/          # Logging, RNG, config, timer
  visualization/  # Training plots, 3D cube renderer, solution replay
scripts/
  train.py        # rubiks-train entrypoint
  solve.py        # rubiks-solve entrypoint
  benchmark.py    # rubiks-benchmark entrypoint
  visualize.py    # rubiks-viz entrypoint
tests/
  unit/           # Unit tests for all modules (~288 tests)
  e2e/            # End-to-end solve tests per puzzle type (~75 tests)
```

---

## Installation

Requires Python 3.11+ and Apple Silicon (MLX dependency).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

---

## Testing

```bash
pytest                     # full suite
pytest tests/unit/         # unit tests only
pytest tests/e2e/          # end-to-end tests only
```

334 tests pass; stochastic solver tests (MCTS, long genetic runs) are marked `xfail(strict=False)`.

---

## Training Outputs

Each training run produces:
- `models/metrics/<solver>_<puzzle>_metrics.csv` — per-epoch/generation metrics
- `models/metrics/<solver>_<puzzle>_plot.png` — training curve plot

Genetic solver plots show **best fitness** (crimson), **mean fitness** (dashed salmon, shaded), **control baseline** (dotted gray), and **stagnation injection events** (vertical blue dashes).
