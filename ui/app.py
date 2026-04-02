"""FastAPI backend for the Rubik's cube ML solver web UI."""
from __future__ import annotations

import asyncio
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

# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------

from rubiks_solve.solvers import GeneticSolver, GeneticConfig, MCTSSolver, MCTSConfig

_SOLVER_NAMES = ["genetic", "mcts"]

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Rubik's Cube ML Solver")

_STATIC_DIR = Path(__file__).parent / "static"

# Job storage
_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_state(puzzle: AbstractPuzzle) -> dict:
    return {
        "colors": puzzle.state.tolist(),
        "is_solved": bool(puzzle.is_solved),
    }


def _serialize_move(move: Move) -> dict:
    return {
        "name": move.name,
        "face": move.face,
        "layer": move.layer,
        "direction": move.direction,
        "double": move.double,
    }


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
        else:
            raise ValueError(f"Unknown solver: {solver_name}")

        result = solver.solve(scrambled)

        # Collect solve states for animation
        solve_states: list[dict] = [_serialize_state(scrambled)]
        replay = scrambled
        for m in result.moves:
            replay = replay.apply_move(m)
            solve_states.append(_serialize_state(replay))

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
    except Exception as exc:
        _jobs[job_id].update({"status": "error", "error": str(exc)})


def _run_train_job(
    job_id: str,
    puzzle_name: str,
    solver_name: str,
    epochs: int,
    seed: int,
    scramble_depth: int,
) -> None:
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
    except Exception as exc:
        _jobs[job_id].update({"status": "error", "error": str(exc)})


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@app.get("/api/puzzles")
async def get_puzzles() -> list[str]:
    return list(_PUZZLE_REGISTRY.keys())


@app.get("/api/solvers")
async def get_solvers() -> list[str]:
    return _SOLVER_NAMES


class SolveRequest(BaseModel):
    puzzle: str
    solver: str
    scramble_depth: int = 5
    seed: int = 42
    max_generations: int = 200
    population_size: int = 100


@app.post("/api/solve")
async def start_solve(req: SolveRequest) -> dict:
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
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


class TrainRequest(BaseModel):
    puzzle: str
    solver: str = "genetic"
    epochs: int = 200
    seed: int = 42
    scramble_depth: int = 5


@app.post("/api/train")
async def start_train(req: TrainRequest) -> dict:
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
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


# ---------------------------------------------------------------------------
# Static file serving — must come AFTER API routes
# ---------------------------------------------------------------------------

# Serve index.html at root
@app.get("/")
async def serve_root():
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
