"""
Layered ADI + IDA* training pipeline.

Phase 1 — Autodidactic Iteration (ADI):
  Starting from the existing CNN checkpoint (trained on scramble-depth labels),
  iteratively improve the value network using exact distances extracted from
  beam-search solutions.  Each ADI round:
    a. Use the current CNN (beam search) to solve N puzzles.
    b. Extract (state, true_distance) pairs from every step of each solution.
    c. Mix with scramble-depth fallback data to prevent forgetting.
    d. Retrain the network for a few epochs.
    e. Increase the max_scramble ceiling so harder puzzles are attempted next.

Phase 2 — IDA* evaluation:
  Load the final ADI checkpoint and test the IDA* solver across scramble
  depths 1-25, comparing against pure beam-search CNN performance.

Usage::

    python -m scripts.train_adi_ida [--puzzle 3x3] [--iterations 5]
    python -m scripts.train_adi_ida --puzzle 3x3 --iterations 8 --start-scramble 11
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Puzzle registry
# ---------------------------------------------------------------------------

def _build_registry():
    from rubiks_solve.core import Cube2x2, Cube3x3, Cube4x4, Cube5x5
    return {"2x2": Cube2x2, "3x3": Cube3x3, "4x4": Cube4x4, "5x5": Cube5x5}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ADI + IDA* training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--puzzle", default="3x3",
                   choices=["2x2", "3x3", "4x4", "5x5"])
    p.add_argument("--iterations", type=int, default=5,
                   help="Number of ADI iterations.")
    p.add_argument("--start-scramble", type=int, default=11,
                   help="Max scramble depth for the first ADI iteration.")
    p.add_argument("--scramble-increment", type=int, default=2,
                   help="Increase max_scramble by this many moves each iteration.")
    p.add_argument("--n-solve", type=int, default=500,
                   help="Puzzles to attempt with beam search per iteration.")
    p.add_argument("--n-fallback", type=int, default=10000,
                   help="Scramble-depth fallback samples per iteration.")
    p.add_argument("--epochs-per-iter", type=int, default=150,
                   help="Training epochs per ADI round.")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--beam-width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("models"),
                   help="Root directory for checkpoints.")
    p.add_argument("--warmstart", type=Path, default=None,
                   help="Path to initial .npz checkpoint (defaults to latest CNN).")
    p.add_argument("--benchmark-n", type=int, default=30,
                   help="Puzzles per depth for the final IDA* benchmark.")
    p.add_argument("--ida-time-limit", type=float, default=30.0,
                   help="Time limit (s) per puzzle for the IDA* benchmark.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_ckpt(model_dir: Path) -> Path | None:
    ckpts = sorted(model_dir.glob("ckpt_*.npz"))
    return ckpts[-1] if ckpts else None


def _load_warmstart(model, ckpt_path: Path) -> None:
    import mlx.core as mx
    weights = mx.load(str(ckpt_path))
    weights.pop("__epoch__", None)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    logger = logging.getLogger(__name__)
    logger.info("Warm-started from %s", ckpt_path)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _benchmark(solver_name: str, solver, puzzle_cls, rng, n_per_depth: int,
               max_depth: int) -> dict[int, tuple[int, int]]:
    """Run solver on ``n_per_depth`` puzzles for each depth 1..max_depth.

    Returns dict mapping depth -> (n_solved, n_total).
    """
    results: dict[int, tuple[int, int]] = {}
    logger = logging.getLogger(__name__)

    for depth in range(1, max_depth + 1):
        n_solved = 0
        for _ in range(n_per_depth):
            puzzle = puzzle_cls.solved_state().scramble(depth, rng)
            r = solver.solve(puzzle)
            if r.solved:
                n_solved += 1
        results[depth] = (n_solved, n_per_depth)
        bar = "█" * (10 * n_solved // n_per_depth)
        logger.info(
            "  [%s] depth %2d: %d/%d  %s",
            solver_name, depth, n_solved, n_per_depth, bar,
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger(__name__)

    registry = _build_registry()
    puzzle_cls = registry[args.puzzle]

    from rubiks_solve.encoding.registry import get_encoder
    from rubiks_solve.solvers.cnn.model import CubeValueNet
    from rubiks_solve.solvers.cnn.adi_trainer import ADITrainer, ADIConfig

    encoder = get_encoder("one_hot", puzzle_cls)
    rng = np.random.default_rng(args.seed)

    # -----------------------------------------------------------------------
    # Build (or warm-start) the value network.
    # -----------------------------------------------------------------------
    model = CubeValueNet(input_size=encoder.output_size)

    adi_ckpt_dir = args.output_dir / "cnn_adi" / puzzle_cls.puzzle_name()
    adi_ckpt_dir.mkdir(parents=True, exist_ok=True)

    warmstart_path = args.warmstart
    if warmstart_path is None:
        # Try ADI checkpoint first, then fall back to base CNN.
        warmstart_path = (_latest_ckpt(adi_ckpt_dir)
                          or _latest_ckpt(args.output_dir / "cnn" / puzzle_cls.puzzle_name()))
    if warmstart_path is not None:
        _load_warmstart(model, warmstart_path)
    else:
        log.warning("No warmstart checkpoint found; training from random init.")

    # -----------------------------------------------------------------------
    # Phase 1: ADI iterations.
    # -----------------------------------------------------------------------
    log.info("=== Phase 1: Autodidactic Iteration (%d rounds) ===", args.iterations)

    adi_cfg = ADIConfig(
        n_iterations=args.iterations,
        n_solve_per_iter=args.n_solve,
        n_fallback_per_iter=args.n_fallback,
        start_max_scramble=args.start_scramble,
        scramble_depth_increment=args.scramble_increment,
        epochs_per_iter=args.epochs_per_iter,
        batch_size=args.batch_size,
        beam_width=args.beam_width,
        beam_max_depth=200,
        checkpoint_dir=adi_ckpt_dir,
        log_interval=50,
    )

    adi_trainer = ADITrainer(
        puzzle_cls=puzzle_cls,
        encoder=encoder,
        model=model,
        config=adi_cfg,
        rng=rng,
    )

    t0 = time.perf_counter()
    summaries = adi_trainer.run()
    adi_elapsed = time.perf_counter() - t0

    log.info("ADI complete in %.1fs", adi_elapsed)
    for s in summaries:
        log.info(
            "  iter %d | max_scramble=%d | solved=%d | adi_samples=%d "
            "| loss=%.4f | mae=%.4f",
            s["iteration"], s["max_scramble"], s["n_solved"],
            s["n_adi_samples"], s["final_loss"], s["final_mae"],
        )

    # -----------------------------------------------------------------------
    # Phase 2: IDA* benchmark.
    # -----------------------------------------------------------------------
    log.info("=== Phase 2: IDA* benchmark ===")

    final_ckpt = _latest_ckpt(adi_ckpt_dir)
    if final_ckpt is None:
        log.error("No ADI checkpoint found; cannot run IDA* benchmark.")
        sys.exit(1)

    from rubiks_solve.solvers.ida_star.solver import IDAStarSolver, IDAStarConfig
    from rubiks_solve.solvers.cnn.solver import CNNSolver, CNNConfig

    ida_cfg = IDAStarConfig(
        model_path=final_ckpt,
        max_depth=puzzle_cls.move_limit() + 10,
        heuristic_weight=0.85,
        time_limit_seconds=args.ida_time_limit,
    )
    ida_solver = IDAStarSolver(puzzle_cls, encoder, ida_cfg)

    cnn_cfg = CNNConfig(
        model_path=final_ckpt,
        beam_width=args.beam_width,
        max_depth=200,
    )
    cnn_solver = CNNSolver(puzzle_cls, encoder, cnn_cfg)

    bench_rng = np.random.default_rng(args.seed + 1)
    max_bench_depth = min(puzzle_cls.move_limit(), 20)

    log.info("--- CNN (beam=%d) ---", args.beam_width)
    cnn_results = _benchmark("cnn", cnn_solver, puzzle_cls, bench_rng,
                              args.benchmark_n, max_bench_depth)

    bench_rng2 = np.random.default_rng(args.seed + 1)
    log.info("--- IDA* (time_limit=%.0fs) ---", args.ida_time_limit)
    ida_results = _benchmark("ida*", ida_solver, puzzle_cls, bench_rng2,
                              args.benchmark_n, max_bench_depth)

    # Print comparison table.
    log.info("\n%s", "=" * 52)
    log.info("%-8s  %-16s  %-16s", "depth", "CNN beam=1024", "IDA*")
    log.info("%s", "-" * 52)
    for d in range(1, max_bench_depth + 1):
        cs, ct = cnn_results[d]
        is_, it = ida_results[d]
        log.info(
            "%-8d  %d/%d  (%3d%%)       %d/%d  (%3d%%)",
            d, cs, ct, 100 * cs // ct, is_, it, 100 * is_ // it,
        )
    log.info("%s", "=" * 52)

    cnn_total = sum(v[0] for v in cnn_results.values())
    ida_total = sum(v[0] for v in ida_results.values())
    n_total = args.benchmark_n * max_bench_depth
    log.info(
        "Overall — CNN: %d/%d (%d%%)  IDA*: %d/%d (%d%%)",
        cnn_total, n_total, 100 * cnn_total // n_total,
        ida_total, n_total, 100 * ida_total // n_total,
    )


if __name__ == "__main__":
    main()
