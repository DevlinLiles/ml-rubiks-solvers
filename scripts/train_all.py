"""Train all solver/puzzle combinations on local (MLX) or DGX Spark (PyTorch).

For --backend local:
    Runs each combination in-process sequentially using the existing MLX trainers.

For --backend dgx:
    Syncs project once, ensures venv once, then runs every combination on the
    DGX Spark in a single SSH session. Results are pulled back at the end.
    Genetic solver is skipped on DGX (no GPU benefit) and noted in the summary.

Usage examples:
    # All solvers, all puzzles, local
    python scripts/train_all.py

    # CNN + DQN on 3x3 and 4x4, local
    python scripts/train_all.py --solvers cnn,dqn --puzzles 3x3,4x4

    # Everything on DGX Spark
    python scripts/train_all.py --backend dgx

    # Neural solvers only, DGX
    python scripts/train_all.py --backend dgx --solvers cnn,policy,dqn --epochs 200
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SOLVERS = ["cnn", "policy", "dqn", "genetic"]
ALL_PUZZLES = ["2x2", "3x3", "4x4", "5x5", "megaminx"]

# Solvers that run on the DGX (neural solvers via PyTorch; genetic is pure Python).
DGX_SOLVERS = {"cnn", "policy", "dqn", "genetic"}

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class JobResult:
    """Result record for a single solver/puzzle training job."""

    solver: str
    puzzle: str
    backend: str
    success: bool
    duration_seconds: float
    skipped: bool = False
    skip_reason: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the train-all script."""
    parser = argparse.ArgumentParser(
        prog="rubiks-train-all",
        description="Train all solver/puzzle combinations locally or on the DGX Spark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=["local", "dgx"],
        default="local",
        help=(
            "'local' uses MLX on Apple Silicon. "
            "'dgx' delegates to MSP-SPARK-01 via SSH (PyTorch + CUDA)."
        ),
    )
    parser.add_argument(
        "--solvers",
        default=",".join(ALL_SOLVERS),
        metavar="LIST",
        help="Comma-separated list of solvers to train.",
    )
    parser.add_argument(
        "--puzzles",
        default=",".join(ALL_PUZZLES),
        metavar="LIST",
        help="Comma-separated list of puzzle types to train on.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--max-scramble", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models"),
        help="Local directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _parse_list(value: str, valid: list[str]) -> list[str]:
    items = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [i for i in items if i not in valid]
    if unknown:
        raise ValueError(f"Unknown values: {unknown}. Valid: {valid}")
    return items


# ---------------------------------------------------------------------------
# Local training (MLX)
# ---------------------------------------------------------------------------


def _build_puzzle_registry() -> dict[str, type]:
    from rubiks_solve.core import Cube2x2, Cube3x3, Cube4x4, Cube5x5
    from rubiks_solve.core.megaminx import Megaminx

    return {
        "2x2": Cube2x2,
        "3x3": Cube3x3,
        "4x4": Cube4x4,
        "5x5": Cube5x5,
        "megaminx": Megaminx,
    }


def _make_local_args(solver: str, puzzle: str, args: argparse.Namespace) -> argparse.Namespace:
    """Build a train.py-compatible Namespace for a single solver/puzzle combo."""
    import copy
    local_args = copy.copy(args)
    local_args.solver = solver
    local_args.puzzle = puzzle
    local_args.backend = "local"
    local_args.config = None
    return local_args


def _run_local_job(solver: str, puzzle: str, args: argparse.Namespace) -> JobResult:
    """Run a single training job in-process using the MLX trainers."""
    # Import train helpers lazily to avoid pulling in MLX before we know we need it.
    sys.path.insert(0, str(Path(__file__).parent))
    from train import (  # type: ignore[import]
        _train_cnn,
        _train_dqn,
        _train_genetic,
        _train_policy,
        _build_puzzle_registry as _reg,
    )
    from rubiks_solve.utils.rng import set_global_seed

    set_global_seed(args.seed)
    puzzle_cls = _reg()[puzzle]
    job_args = _make_local_args(solver, puzzle, args)

    dispatch = {
        "cnn": _train_cnn,
        "policy": _train_policy,
        "dqn": _train_dqn,
        "genetic": _train_genetic,
    }

    t0 = time.perf_counter()
    try:
        metrics_df: pd.DataFrame = dispatch[solver](puzzle_cls, job_args, logger)

        # Save metrics CSV
        metrics_dir = args.output_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        csv_path = metrics_dir / f"{solver}_{puzzle}_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)

        duration = time.perf_counter() - t0
        return JobResult(
            solver=solver, puzzle=puzzle, backend="local",
            success=True, duration_seconds=duration,
        )
    except Exception as exc:  # noqa: BLE001
        duration = time.perf_counter() - t0
        logger.error("Job failed: %s/%s — %s", solver, puzzle, exc)
        return JobResult(
            solver=solver, puzzle=puzzle, backend="local",
            success=False, duration_seconds=duration, error=str(exc),
        )


def run_local(
    solvers: list[str],
    puzzles: list[str],
    args: argparse.Namespace,
) -> list[JobResult]:
    """Run all combinations locally using MLX trainers."""
    results: list[JobResult] = []
    total = len(solvers) * len(puzzles)
    count = 0

    for solver in solvers:
        for puzzle in puzzles:
            count += 1
            logger.info(
                "[%d/%d] Local training: %s on %s", count, total, solver, puzzle
            )
            result = _run_local_job(solver, puzzle, args)
            results.append(result)
            _log_job_result(result)

    return results


# ---------------------------------------------------------------------------
# DGX training (PyTorch, single SSH session)
# ---------------------------------------------------------------------------


def run_dgx(
    solvers: list[str],
    puzzles: list[str],
    args: argparse.Namespace,
) -> list[JobResult]:
    """Run all compatible combinations on the DGX Spark in one SSH session."""
    sys.path.insert(0, str(Path(__file__).parent))
    from remote_train import (  # type: ignore[import]
        _load_env,
        _require_env,
        _rsync_project,
        _ensure_venv,
        _build_train_cmd,
        _pull_results,
        _get_connection,
        _expand,
    )

    _load_env()
    _os = __import__("os")
    remote_dir = _expand(_os.environ.get("DGX_REMOTE_DIR", "~/rubiks-solve"))
    venv_path = _expand(_os.environ.get("DGX_VENV_PATH", "~/rubiks-venv"))
    torch_index_url = _os.environ.get(
        "DGX_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu130"
    )

    results: list[JobResult] = []

    # --- Classify jobs ---
    dgx_jobs: list[tuple[str, str]] = []
    for solver in solvers:
        for puzzle in puzzles:
            if solver not in DGX_SOLVERS:
                results.append(JobResult(
                    solver=solver, puzzle=puzzle, backend="dgx",
                    success=False, duration_seconds=0.0,
                    skipped=True,
                    skip_reason=f"'{solver}' is not supported on DGX (no GPU benefit). "
                                "Run with --backend local to train it.",
                ))
            else:
                dgx_jobs.append((solver, puzzle))

    if not dgx_jobs:
        logger.warning("No DGX-compatible jobs to run.")
        return results

    total = len(dgx_jobs)
    local_root = Path(__file__).parent.parent.resolve()

    # --- Step 1: rsync once ---
    logger.info("Syncing project to DGX Spark (%d jobs queued)...", total)
    _rsync_project(None, local_root, remote_dir)

    # --- Step 2: open SSH connection, prepare venv once, then close ---
    logger.info("Ensuring PyTorch venv on DGX Spark...")
    setup_conn = _get_connection()
    with setup_conn:
        _ensure_venv(setup_conn, venv_path, torch_index_url)

    # --- Step 3: run each job with a fresh SSH connection ---
    # A fresh connection per job ensures one dropped session doesn't cascade.
    import copy
    for count, (solver, puzzle) in enumerate(dgx_jobs, 1):
        logger.info("[%d/%d] DGX training: %s on %s", count, total, solver, puzzle)

        job_args = copy.copy(args)
        job_args.solver = solver
        job_args.puzzle = puzzle

        cmd = _build_train_cmd(venv_path, remote_dir, job_args)
        t0 = time.perf_counter()
        try:
            job_conn = _get_connection()
            with job_conn:
                job_conn.run(cmd, pty=True)
            duration = time.perf_counter() - t0
            result = JobResult(
                solver=solver, puzzle=puzzle, backend="dgx",
                success=True, duration_seconds=duration,
            )
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - t0
            logger.error("DGX job failed: %s/%s — %s", solver, puzzle, exc)
            result = JobResult(
                solver=solver, puzzle=puzzle, backend="dgx",
                success=False, duration_seconds=duration, error=str(exc),
            )

        results.append(result)
        _log_job_result(result)

    # --- Step 4: pull all results back once ---
    logger.info("Pulling all checkpoints and metrics from DGX...")
    _pull_results(None, remote_dir, args.output_dir)

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _log_job_result(result: JobResult) -> None:
    if result.skipped:
        logger.warning("SKIPPED  %s/%s — %s", result.solver, result.puzzle, result.skip_reason)
    elif result.success:
        logger.info(
            "DONE     %s/%s in %.1fs", result.solver, result.puzzle, result.duration_seconds
        )
    else:
        logger.error("FAILED   %s/%s — %s", result.solver, result.puzzle, result.error)


def _print_summary(results: list[JobResult]) -> None:
    print("\n" + "=" * 60)
    print(f"{'SOLVER':<10} {'PUZZLE':<12} {'STATUS':<10} {'DURATION':>10}")
    print("-" * 60)

    passed = failed = skipped = 0
    for r in results:
        if r.skipped:
            status = "SKIPPED"
            skipped += 1
            duration = "-"
        elif r.success:
            status = "DONE"
            passed += 1
            duration = f"{r.duration_seconds:.1f}s"
        else:
            status = "FAILED"
            failed += 1
            duration = f"{r.duration_seconds:.1f}s"

        print(f"{r.solver:<10} {r.puzzle:<12} {status:<10} {duration:>10}")

    print("=" * 60)
    print(f"  Done: {passed}  |  Failed: {failed}  |  Skipped: {skipped}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, run all training jobs, print summary, and exit with status."""
    args = parse_args()

    from rubiks_solve.utils.logging_config import configure_logging

    configure_logging(level=args.log_level)

    try:
        solvers = _parse_list(args.solvers, ALL_SOLVERS)
        puzzles = _parse_list(args.puzzles, ALL_PUZZLES)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    total_jobs = len(solvers) * len(puzzles)
    logger.info(
        "train_all starting | backend=%s | solvers=%s | puzzles=%s | total_jobs=%d",
        args.backend, solvers, puzzles, total_jobs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "local":
        results = run_local(solvers, puzzles, args)
    else:
        results = run_dgx(solvers, puzzles, args)

    _print_summary(results)

    failed = [r for r in results if not r.success and not r.skipped]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
