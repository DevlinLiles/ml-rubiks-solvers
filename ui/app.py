"""FastAPI backend for the Rubik's cube ML solver web UI."""
from __future__ import annotations

import asyncio
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Puzzle registry
# ---------------------------------------------------------------------------

from rubiks_solve.core import (
    Cube2x2,
    Cube3x3,
    Cube4x4,
    Cube5x5,
    AbstractPuzzle,
    Move,
)

_PUZZLE_REGISTRY: dict[str, type[AbstractPuzzle]] = {
    "2x2": Cube2x2,
    "3x3": Cube3x3,
    "4x4": Cube4x4,
    "5x5": Cube5x5,
}

try:
    from rubiks_solve.core.megaminx import Megaminx
    _PUZZLE_REGISTRY["megaminx"] = Megaminx
except ImportError:
    pass

try:
    from rubiks_solve.core.skewb_ultimate import SkewbUltimate
    _PUZZLE_REGISTRY["skewb_ultimate"] = SkewbUltimate
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------

from rubiks_solve.solvers import GeneticSolver, GeneticConfig, MCTSSolver, MCTSConfig
from rubiks_solve.solvers.cnn import CNNSolver, CNNConfig
from rubiks_solve.solvers.policy import PolicyNetworkSolver, PolicyConfig
from rubiks_solve.solvers.dqn import DQNSolver, DQNConfig
from rubiks_solve.solvers.ida_star import IDAStarSolver, IDAStarConfig
from rubiks_solve.encoding import get_encoder
from rubiks_solve.solvers.base import AbstractSolver

_SOLVER_NAMES = ["genetic", "mcts", "cnn", "policy", "dqn", "ida_star"]

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Rubik's Cube ML Solver")

_STATIC_DIR = Path(__file__).parent / "static"
_MODELS_DIR = Path(__file__).parent.parent / "models"

# Job storage
_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=4)

# ML solver cache: (model_type, puzzle_name) -> solver instance
_ml_solver_cache: dict[tuple[str, str], AbstractSolver] = {}
_ml_solver_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_state(puzzle: AbstractPuzzle) -> dict:
    """Serialize a puzzle state to a JSON-compatible dict."""
    return {
        "colors": puzzle.state.tolist(),
        "is_solved": bool(puzzle.is_solved),
    }


def _serialize_move(move: Move) -> dict:
    """Serialize a Move to a JSON-compatible dict."""
    return {
        "name": move.name,
        "face": move.face,
        "layer": move.layer,
        "direction": move.direction,
        "double": move.double,
    }


# ---------------------------------------------------------------------------
# ML solver helpers
# ---------------------------------------------------------------------------


def _latest_ckpt(model_type: str, puzzle_name: str) -> Path | None:
    """Return the path to the latest checkpoint for *model_type* / *puzzle_name*.

    ``ida_star`` uses ADI-trained weights from ``models/cnn_adi/<puzzle>/``,
    falling back to the base CNN if no ADI checkpoint exists.
    All other model types look under ``models/<model_type>/<puzzle>/``.
    Returns ``None`` when no checkpoint is found.
    """
    if model_type == "ida_star":
        # Prefer ADI-trained weights; fall back to base CNN.
        for base in ("cnn_adi", "cnn"):
            model_dir = _MODELS_DIR / base / puzzle_name
            if model_dir.exists():
                ckpts = sorted(model_dir.glob("ckpt_*.npz"))
                if ckpts:
                    return ckpts[-1]
        return None

    model_dir = _MODELS_DIR / model_type
    if puzzle_name != "3x3":
        model_dir = model_dir / puzzle_name
    if not model_dir.exists():
        return None
    ckpts = sorted(model_dir.glob("ckpt_*.npz"))
    return ckpts[-1] if ckpts else None


def _get_ml_solver(
    model_type: str,
    puzzle_name: str,
    puzzle_cls: type[AbstractPuzzle],
) -> AbstractSolver | None:
    """Return a cached MLX solver for *model_type* / *puzzle_name*, building
    it on first access.  Returns ``None`` when no checkpoint is available.
    """
    key = (model_type, puzzle_name)
    with _ml_solver_lock:
        if key not in _ml_solver_cache:
            ckpt = _latest_ckpt(model_type, puzzle_name)
            if ckpt is None:
                return None
            encoder = get_encoder("one_hot", puzzle_cls)
            if model_type == "cnn":
                cfg = CNNConfig(
                    model_path=ckpt,
                    beam_width=1024,
                    max_depth=puzzle_cls.move_limit() * 10,
                )
                solver: AbstractSolver = CNNSolver(puzzle_cls, encoder, cfg)
            elif model_type == "policy":
                cfg = PolicyConfig(model_path=ckpt, deterministic=True)
                solver = PolicyNetworkSolver(puzzle_cls, encoder, cfg)
            elif model_type == "dqn":
                cfg = DQNConfig(
                    model_path=ckpt,
                    max_steps=puzzle_cls.move_limit() * 10,
                )
                solver = DQNSolver(puzzle_cls, encoder, cfg)
            elif model_type == "ida_star":
                cfg = IDAStarConfig(
                    model_path=ckpt,
                    max_depth=puzzle_cls.move_limit() + 10,
                    heuristic_weight=0.85,
                    time_limit_seconds=60.0,
                )
                solver = IDAStarSolver(puzzle_cls, encoder, cfg)
            else:
                return None
            _ml_solver_cache[key] = solver
        return _ml_solver_cache.get(key)


# ---------------------------------------------------------------------------
# Background job runners
# ---------------------------------------------------------------------------


def _run_solve_job(
    job_id: str,
    puzzle_name: str,
    solver_name: str,
    scramble_depth: int,
    seed: int,
    max_generations: int,
    population_size: int,
) -> None:
    """Background worker: run a solve job and store the result in _jobs."""
    try:
        puzzle_cls = _PUZZLE_REGISTRY[puzzle_name]
        rng = np.random.default_rng(seed)

        # Build scramble sequence from solved state
        solved = puzzle_cls.solved_state()
        scramble_moves: list[Move] = []
        current = solved
        legal = solved.legal_moves()
        last_move: Optional[Move] = None
        for _ in range(scramble_depth):
            candidates = [m for m in legal if (last_move is None or m.face != last_move.face)]
            if not candidates:
                candidates = legal
            move = candidates[int(rng.integers(len(candidates)))]
            scramble_moves.append(move)
            current = current.apply_move(move)
            last_move = move

        scrambled = current

        # Collect scramble states for animation
        scramble_states: list[dict] = [_serialize_state(solved)]
        replay = solved
        for m in scramble_moves:
            replay = replay.apply_move(m)
            scramble_states.append(_serialize_state(replay))

        # Build and run solver
        if solver_name == "genetic":
            # Use the policy network for a fast single-forward-pass-per-step
            # rollout; fall back to the GA when no checkpoint exists.
            solver = _get_ml_solver("policy", puzzle_name, puzzle_cls)
            if solver is None:
                cfg = GeneticConfig(
                    max_generations=max_generations,
                    population_size=population_size,
                    max_chromosome_length=puzzle_cls.move_limit(),
                    seed=seed,
                )
                solver = GeneticSolver(puzzle_cls, cfg)
        elif solver_name == "mcts":
            cfg = MCTSConfig(
                n_simulations=500_000,
                time_limit_seconds=float(max_generations),  # repurpose field as seconds
                seed=seed,
            )
            solver = MCTSSolver(puzzle_cls, cfg)
        elif solver_name == "cnn":
            solver = _get_ml_solver("cnn", puzzle_name, puzzle_cls)
            if solver is None:
                raise ValueError(f"No CNN checkpoint found for puzzle '{puzzle_name}'")
        elif solver_name == "policy":
            solver = _get_ml_solver("policy", puzzle_name, puzzle_cls)
            if solver is None:
                raise ValueError(f"No policy checkpoint found for puzzle '{puzzle_name}'")
        elif solver_name == "dqn":
            solver = _get_ml_solver("dqn", puzzle_name, puzzle_cls)
            if solver is None:
                raise ValueError(f"No DQN checkpoint found for puzzle '{puzzle_name}'")
        elif solver_name == "ida_star":
            solver = _get_ml_solver("ida_star", puzzle_name, puzzle_cls)
            if solver is None:
                raise ValueError(f"No IDA* checkpoint found for puzzle '{puzzle_name}'")
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

        result = solver.solve(scrambled)

        move_budget = puzzle_cls.move_limit() * 5
        print(
            f"[solve] puzzle={puzzle_name} solver={solver_name} scramble={scramble_depth} "
            f"solved={result.solved} moves={result.move_count} "
            f"iterations={result.iterations} budget={move_budget} "
            f"meta={result.metadata}",
            file=sys.stderr,
            flush=True,
        )

        # Collect solve states for animation — stop at solved or 5x move_limit
        solve_states: list[dict] = [_serialize_state(scrambled)]
        replay = scrambled
        for m in result.moves[:move_budget]:
            replay = replay.apply_move(m)
            solve_states.append(_serialize_state(replay))
            if replay.is_solved:
                break

        _jobs[job_id].update(
            {
                "status": "done",
                "solved": result.solved,
                "solve_time": result.solve_time_seconds,
                "iterations": result.iterations,
                "move_count": result.move_count,
                "scramble_moves": [_serialize_move(m) for m in scramble_moves],
                "scramble_states": scramble_states,
                "solve_moves": [_serialize_move(m) for m in result.moves],
                "solve_states": solve_states,
                "metadata": {
                    k: (v if not isinstance(v, np.ndarray) else v.tolist())
                    for k, v in result.metadata.items()
                },
            }
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _jobs[job_id].update({"status": "error", "error": str(exc)})


def _run_train_job(
    job_id: str,
    puzzle_name: str,
    _solver_name: str,
    epochs: int,
    seed: int,
    scramble_depth: int,
) -> None:
    """Background worker: run a training job and store the result in _jobs."""
    try:
        puzzle_cls = _PUZZLE_REGISTRY[puzzle_name]
        rng = np.random.default_rng(seed)
        solved = puzzle_cls.solved_state()
        scrambled = solved.scramble(scramble_depth, rng)

        cfg = GeneticConfig(
            max_generations=epochs,
            population_size=100,
            seed=seed,
        )
        solver = GeneticSolver(puzzle_cls, cfg)
        result = solver.solve(scrambled)

        meta = result.metadata
        _jobs[job_id].update(
            {
                "status": "done",
                "solved": result.solved,
                "solve_time": result.solve_time_seconds,
                "iterations": result.iterations,
                "fitness_best_history": meta.get("fitness_best_history", meta.get("fitness_history", [])),
                "fitness_mean_history": meta.get("fitness_mean_history", []),
                "control_fitness_history": meta.get("control_fitness_history", []),
            }
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _jobs[job_id].update({"status": "error", "error": str(exc)})


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/api/puzzles")
async def get_puzzles() -> list[str]:
    """Return the list of supported puzzle names."""
    return list(_PUZZLE_REGISTRY.keys())


@app.get("/api/move_limits")
async def get_move_limits() -> dict[str, int]:
    """Return the move limit for each puzzle type."""
    return {name: cls.move_limit() for name, cls in _PUZZLE_REGISTRY.items()}


@app.get("/api/solvers")
async def get_solvers() -> list[str]:
    """Return the list of available solver names."""
    return _SOLVER_NAMES


class SolveRequest(BaseModel):
    """Request body for the /api/solve endpoint."""

    puzzle: str
    solver: str
    scramble_depth: int = 5
    seed: int = 42
    max_generations: int = 200
    population_size: int = 100


@app.post("/api/solve")
async def start_solve(req: SolveRequest) -> dict:
    """Start a background solve job and return its job_id."""
    if req.puzzle not in _PUZZLE_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown puzzle: {req.puzzle}")
    if req.solver not in _SOLVER_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown solver: {req.solver}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "started_at": time.time()}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_solve_job,
        job_id,
        req.puzzle,
        req.solver,
        req.scramble_depth,
        req.seed,
        req.max_generations,
        req.population_size,
    )

    return {"job_id": job_id}


@app.get("/api/solve/{job_id}")
async def get_solve_result(job_id: str) -> dict:
    """Return the current status or result of a solve job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


class TrainRequest(BaseModel):
    """Request body for the /api/train endpoint."""

    puzzle: str
    solver: str = "genetic"
    epochs: int = 200
    seed: int = 42
    scramble_depth: int = 5


@app.post("/api/train")
async def start_train(req: TrainRequest) -> dict:
    """Start a background training job and return its job_id."""
    if req.puzzle not in _PUZZLE_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown puzzle: {req.puzzle}")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "running", "started_at": time.time()}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_train_job,
        job_id,
        req.puzzle,
        req.solver,
        req.epochs,
        req.seed,
        req.scramble_depth,
    )

    return {"job_id": job_id}


@app.get("/api/train/{job_id}")
async def get_train_result(job_id: str) -> dict:
    """Return the current status or result of a training job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


# ---------------------------------------------------------------------------
# Static file serving — must come AFTER API routes
# ---------------------------------------------------------------------------

# Serve index.html at root
@app.get("/")
async def serve_root():
    """Serve the main index.html page."""
    return FileResponse(_STATIC_DIR / "index.html")


# Mount static assets
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    uvicorn.run("ui.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
