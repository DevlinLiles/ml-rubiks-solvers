"""PyTorch training entry point — runs on the DGX Spark.

Same CLI interface as scripts/train.py, but uses PyTorch models/trainers
instead of MLX. Only supports neural-network solvers (cnn, policy, dqn);
genetic/mcts training falls back to the local MLX path.

Usage (on DGX Spark):
    python scripts/train_torch.py --solver cnn --puzzle 3x3 --epochs 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script (package not installed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


# ---------------------------------------------------------------------------
# Puzzle registry (same as train.py)
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


# ---------------------------------------------------------------------------
# Argument parsing (mirrors train.py)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rubiks-train-torch",
        description="Train a solver on the DGX Spark using PyTorch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--solver",
        choices=["cnn", "policy", "dqn", "genetic"],
        required=True,
    )
    parser.add_argument(
        "--puzzle",
        choices=["2x2", "3x3", "4x4", "5x5", "megaminx"],
        default="3x3",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--max-scramble", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training helpers (PyTorch versions)
# ---------------------------------------------------------------------------


def _make_scramble_dataset(puzzle_cls: type, args: argparse.Namespace):
    from rubiks_solve.encoding.registry import get_encoder
    from rubiks_solve.training.data_gen import ScrambleDataset
    from rubiks_solve.utils.rng import make_rng

    encoder = get_encoder("one_hot", puzzle_cls)
    rng = make_rng(args.seed)
    return ScrambleDataset(
        puzzle_factory=puzzle_cls.solved_state,
        encoder=encoder,
        rng=rng,
    ), encoder


def _train_cnn(puzzle_cls: type, args: argparse.Namespace, logger: object):
    from rubiks_solve.solvers.cnn.model_torch import CubeValueNet
    from rubiks_solve.solvers.cnn.trainer_torch import CNNTrainer, CNNTrainerConfig
    from rubiks_solve.encoding.registry import get_encoder
    from rubiks_solve.utils.rng import make_rng

    rng = make_rng(args.seed)
    encoder = get_encoder("one_hot", puzzle_cls)
    input_size = encoder.output_size

    logger.info(  # type: ignore[union-attr]
        "Generating CNN training data",
        n_samples=args.n_train,
        max_scramble=args.max_scramble,
    )

    states_list: list[np.ndarray] = []
    labels_list: list[float] = []
    solved = puzzle_cls.solved_state()

    for _ in range(args.n_train):
        depth = int(rng.integers(1, args.max_scramble + 1))
        scrambled = solved.scramble(depth, rng)
        states_list.append(encoder.encode(scrambled))
        labels_list.append(float(depth))

    states = np.array(states_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)

    checkpoint_dir = args.output_dir / "cnn" / puzzle_cls.puzzle_name()
    model = CubeValueNet(input_size=input_size)
    trainer_cfg = CNNTrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir,
    )
    trainer = CNNTrainer(model, trainer_cfg)

    logger.info("Starting CNN training (PyTorch)", epochs=args.epochs)  # type: ignore[union-attr]
    history = trainer.train(states, labels)

    import pandas as pd  # noqa: PLC0415
    df = pd.DataFrame(history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def _train_policy(puzzle_cls: type, args: argparse.Namespace, logger: object):
    from rubiks_solve.solvers.policy.model_torch import CubePolicyNet
    from rubiks_solve.solvers.policy.trainer_torch import PolicyTrainer, PolicyTrainerConfig

    dataset, encoder = _make_scramble_dataset(puzzle_cls, args)
    n_actions = len(puzzle_cls.solved_state().legal_moves())
    input_size = encoder.output_size

    logger.info(  # type: ignore[union-attr]
        "Generating policy training data",
        n_samples=args.n_train,
        max_scramble=args.max_scramble,
    )
    states, actions = dataset.generate_policy_batch(
        batch_size=args.n_train,
        min_depth=1,
        max_depth=args.max_scramble,
        n_actions=n_actions,
    )
    valid = actions >= 0
    states, actions = states[valid], actions[valid]
    logger.info("Policy training data ready", n_valid=int(valid.sum()))  # type: ignore[union-attr]

    checkpoint_dir = args.output_dir / "policy" / puzzle_cls.puzzle_name()
    model = CubePolicyNet(input_size=input_size, n_actions=n_actions)
    trainer_cfg = PolicyTrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir,
    )
    trainer = PolicyTrainer(model, trainer_cfg)

    logger.info("Starting Policy training (PyTorch)", epochs=args.epochs)  # type: ignore[union-attr]
    history = trainer.train(states, actions)

    import pandas as pd  # noqa: PLC0415
    df = pd.DataFrame(history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def _train_dqn(puzzle_cls: type, args: argparse.Namespace, logger: object):
    from rubiks_solve.solvers.dqn.model_torch import DuelingDQN
    from rubiks_solve.solvers.dqn.trainer_torch import DQNTrainer, DQNTrainerConfig
    from rubiks_solve.encoding.registry import get_encoder
    from rubiks_solve.utils.rng import make_rng
    from rubiks_solve.env.reward import DenseReward

    encoder = get_encoder("one_hot", puzzle_cls)
    rng = make_rng(args.seed)
    reward_fn = DenseReward()

    class _SimpleEnv:
        def __init__(self) -> None:
            self._rng = rng
            self._scramble_depth = min(args.max_scramble, puzzle_cls.move_limit())

        def reset(self):
            depth = max(1, int(self._rng.integers(1, self._scramble_depth + 1)))
            return puzzle_cls.solved_state().scramble(depth, self._rng)

        def step(self, puzzle, move):
            next_puzzle = puzzle.apply_move(move)
            reward = reward_fn(puzzle, move, next_puzzle)
            done = next_puzzle.is_solved
            return next_puzzle, reward, done

        def legal_moves(self):
            return puzzle_cls.solved_state().legal_moves()

    env = _SimpleEnv()
    n_actions = len(env.legal_moves())
    input_size = encoder.output_size

    online_model = DuelingDQN(input_size=input_size, n_actions=n_actions)
    target_model = DuelingDQN(input_size=input_size, n_actions=n_actions)

    checkpoint_dir = args.output_dir / "dqn" / puzzle_cls.puzzle_name()
    steps_per_epoch = max(500, args.n_train // args.epochs)
    trainer_cfg = DQNTrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        steps_per_epoch=steps_per_epoch,
        min_replay_size=min(2000, args.n_train // 10),
        epsilon_decay_steps=args.epochs * steps_per_epoch,
        replay_buffer_size=min(100_000, max(50_000, args.n_train // 10)),
        target_update_freq=steps_per_epoch,
        checkpoint_dir=checkpoint_dir,
    )
    trainer = DQNTrainer(online_model, target_model, env, trainer_cfg, encoder)

    logger.info(  # type: ignore[union-attr]
        "Starting DQN training (PyTorch)",
        epochs=args.epochs,
        steps_per_epoch=trainer_cfg.steps_per_epoch,
    )
    history = trainer.train()

    import pandas as pd  # noqa: PLC0415
    df = pd.DataFrame(history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def _train_genetic(puzzle_cls: type, args: argparse.Namespace, logger: object):
    from rubiks_solve.solvers.genetic.solver import GeneticConfig, GeneticSolver
    from rubiks_solve.utils.rng import make_rng

    import pandas as pd  # noqa: PLC0415

    rng = make_rng(args.seed)
    cfg = GeneticConfig(
        max_generations=args.epochs,
        seed=args.seed,
        max_chromosome_length=puzzle_cls.move_limit(),
    )
    solver = GeneticSolver(puzzle_cls, cfg)
    puzzle = puzzle_cls.solved_state().scramble(10, rng)

    logger.info(  # type: ignore[union-attr]
        "Starting genetic training", puzzle=puzzle_cls.puzzle_name(), generations=args.epochs
    )
    result = solver.solve(puzzle)

    fitness_best = result.metadata.get("fitness_best_history", result.metadata.get("fitness_history", []))
    fitness_mean = result.metadata.get("fitness_mean_history", [])
    n = len(fitness_best)
    df = pd.DataFrame({
        "generation": list(range(1, n + 1)),
        "fitness_best": fitness_best,
        "fitness_mean": fitness_mean if len(fitness_mean) == n else [None] * n,
    })
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    from rubiks_solve.utils.logging_config import configure_logging, get_logger
    from rubiks_solve.utils.rng import set_global_seed

    configure_logging(level=args.log_level)
    logger = get_logger(__name__)

    set_global_seed(args.seed)

    # Log device info (genetic doesn't need torch)
    if args.solver != "genetic":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("PyTorch device", device=device)  # type: ignore[union-attr]
            if torch.cuda.is_available():
                logger.info("GPU", name=torch.cuda.get_device_name(0))  # type: ignore[union-attr]
        except ImportError:
            logger.error("PyTorch is not installed — cannot run DGX training")  # type: ignore[union-attr]
            sys.exit(1)

    puzzle_registry = _build_puzzle_registry()
    puzzle_cls = puzzle_registry[args.puzzle]

    logger.info(  # type: ignore[union-attr]
        "DGX training started",
        solver=args.solver,
        puzzle=args.puzzle,
        epochs=args.epochs,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if args.solver == "cnn":
        metrics_df = _train_cnn(puzzle_cls, args, logger)
        plot_col = "epoch"
    elif args.solver == "policy":
        metrics_df = _train_policy(puzzle_cls, args, logger)
        plot_col = "epoch"
    elif args.solver == "dqn":
        metrics_df = _train_dqn(puzzle_cls, args, logger)
        plot_col = "epoch"
    elif args.solver == "genetic":
        metrics_df = _train_genetic(puzzle_cls, args, logger)
        plot_col = "generation"
    else:
        logger.error("Solver not supported on DGX backend", solver=args.solver)  # type: ignore[union-attr]
        sys.exit(1)

    csv_path = metrics_dir / f"{args.solver}_{args.puzzle}_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info("Metrics saved", path=str(csv_path))  # type: ignore[union-attr]

    try:
        from rubiks_solve.visualization.training_plots import plot_loss_curve

        plot_path = metrics_dir / f"{args.solver}_{args.puzzle}_plot.png"
        solver_label = f"{args.solver.upper()} ({args.puzzle}) [DGX]"
        if "loss" in metrics_df.columns:
            plot_loss_curve(metrics_df, solver_name=solver_label, output_path=plot_path)
        logger.info("Training plot saved", path=str(plot_path))  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save training plot", error=str(exc))  # type: ignore[union-attr]

    logger.info("DGX training complete")  # type: ignore[union-attr]


if __name__ == "__main__":
    main()
