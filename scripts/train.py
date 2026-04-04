"""CLI: train a solver on a puzzle type.

Usage examples::

    rubiks-train --solver genetic --puzzle 3x3 --epochs 100
    rubiks-train --solver cnn --puzzle 3x3 --config config.json
    rubiks-train --solver genetic --puzzle 2x2 --seed 7 --output-dir runs/exp1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Puzzle registry
# ---------------------------------------------------------------------------

_PUZZLE_REGISTRY: dict[str, type] = {}


def _build_puzzle_registry() -> dict[str, type]:
    from rubiks_solve.core import Cube2x2, Cube3x3, Cube4x4, Cube5x5
    from rubiks_solve.core.megaminx import Megaminx
    from rubiks_solve.core.skewb_ultimate import SkewbUltimate

    return {
        "2x2": Cube2x2,
        "3x3": Cube3x3,
        "4x4": Cube4x4,
        "5x5": Cube5x5,
        "megaminx": Megaminx,
        "skewb_ultimate": SkewbUltimate,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script.

    Returns:
        Populated :class:`argparse.Namespace` with all training parameters.
    """
    parser = argparse.ArgumentParser(
        prog="rubiks-train",
        description="Train a solver on a Rubik's cube puzzle type.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--solver",
        choices=["genetic", "cnn", "policy", "dqn", "mcts"],
        required=True,
        help="Solver algorithm to train.",
    )
    parser.add_argument(
        "--puzzle",
        choices=["2x2", "3x3", "4x4", "5x5", "megaminx", "skewb_ultimate"],
        default="3x3",
        help="Puzzle type to train on.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (CNN) or generations (genetic).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to a JSON/YAML AppConfig file.  CLI flags override config values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        metavar="DIR",
        help="Directory for checkpoints and metrics output.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=10_000,
        help="Number of training samples to generate (CNN only).",
    )
    parser.add_argument(
        "--max-scramble",
        type=int,
        default=20,
        help="Maximum scramble depth when generating training data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size for neural network training.",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "dgx"],
        default="local",
        help=(
            "'local' runs training on this machine using MLX (Apple Silicon). "
            "'dgx' delegates training to the DGX Spark (MSP-SPARK-01) via SSH using PyTorch. "
            "Credentials are read from the .env file (see .env.example)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _train_genetic(puzzle_cls: type, args: argparse.Namespace, logger: object) -> pd.DataFrame:
    """Run genetic algorithm training and return a metrics DataFrame.

    Args:
        puzzle_cls: Concrete puzzle class (e.g. ``Cube3x3``).
        args:       Parsed CLI arguments.
        logger:     structlog logger instance.

    Returns:
        DataFrame with columns ``generation``, ``fitness_best``.
    """
    from rubiks_solve.solvers.genetic.solver import GeneticConfig, GeneticSolver
    from rubiks_solve.utils.rng import make_rng

    rng = make_rng(args.seed)
    cfg = GeneticConfig(
        max_generations=args.epochs,
        seed=args.seed,
        max_chromosome_length=puzzle_cls.move_limit(),
    )
    solver = GeneticSolver(puzzle_cls, cfg)

    puzzle = puzzle_cls.solved_state().scramble(10, rng)
    logger.info(  # type: ignore[union-attr]
        "Starting genetic training",
        puzzle=puzzle_cls.puzzle_name(),
        generations=args.epochs,
    )

    result = solver.solve(puzzle)
    fitness_best = result.metadata.get("fitness_best_history", result.metadata.get("fitness_history", []))
    fitness_mean = result.metadata.get("fitness_mean_history", [])
    control_fitness = result.metadata.get("control_fitness_history", [])
    stagnation_events = result.metadata.get("stagnation_events", [])
    n = len(fitness_best)

    df = pd.DataFrame(
        {
            "generation": list(range(1, n + 1)),
            "fitness_best": fitness_best,
            "fitness_mean": fitness_mean if len(fitness_mean) == n else [None] * n,
            "control_fitness": control_fitness if len(control_fitness) == n else [None] * n,
        }
    )
    # Stash stagnation events as metadata on the df for use in plotting
    df.attrs["stagnation_events"] = stagnation_events
    return df


def _make_scramble_dataset(puzzle_cls: type, args: argparse.Namespace):
    """Build a ScrambleDataset wired to puzzle_cls and a one-hot encoder."""
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


def _train_cnn(puzzle_cls: type, args: argparse.Namespace, logger: object) -> pd.DataFrame:
    """Generate training data and run CNN value-network training.

    Args:
        puzzle_cls: Concrete puzzle class.
        args:       Parsed CLI arguments.
        logger:     structlog logger instance.

    Returns:
        DataFrame with columns ``epoch``, ``loss``, ``mae``.
    """
    from rubiks_solve.solvers.cnn.model import CubeValueNet
    from rubiks_solve.solvers.cnn.trainer import CNNTrainer, CNNTrainerConfig
    from rubiks_solve.encoding.registry import get_encoder
    from rubiks_solve.utils.rng import make_rng

    rng = make_rng(args.seed)
    encoder = get_encoder("one_hot", puzzle_cls)
    input_size = encoder.output_size

    logger.info(  # type: ignore[union-attr]
        "Generating training data",
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

    logger.info("Starting CNN training", epochs=args.epochs)  # type: ignore[union-attr]
    history = trainer.train(states, labels)

    df = pd.DataFrame(history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def _train_policy(puzzle_cls: type, args: argparse.Namespace, logger: object) -> pd.DataFrame:
    """Generate imitation-learning data and train a CubePolicyNet.

    Args:
        puzzle_cls: Concrete puzzle class.
        args:       Parsed CLI arguments.
        logger:     structlog logger instance.

    Returns:
        DataFrame with columns ``epoch``, ``loss``, ``accuracy``.
    """
    from rubiks_solve.solvers.policy.model import CubePolicyNet
    from rubiks_solve.solvers.policy.trainer import PolicyTrainer, PolicyTrainerConfig

    dataset, encoder = _make_scramble_dataset(puzzle_cls, args)
    n_actions = len(puzzle_cls.solved_state().legal_moves())
    input_size = encoder.output_size

    logger.info(  # type: ignore[union-attr]
        "Generating policy training data",
        n_samples=args.n_train,
        max_scramble=args.max_scramble,
        n_actions=n_actions,
    )
    states, actions = dataset.generate_policy_batch(
        batch_size=args.n_train,
        min_depth=1,
        max_depth=args.max_scramble,
    )
    # Drop samples where optimal move could not be determined (label == -1)
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

    logger.info("Starting Policy training", epochs=args.epochs)  # type: ignore[union-attr]
    history = trainer.train(states, actions)

    df = pd.DataFrame(history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df


def _train_dqn(puzzle_cls: type, args: argparse.Namespace, logger: object) -> pd.DataFrame:
    """Train a DuelingDQN on a puzzle environment.

    Args:
        puzzle_cls: Concrete puzzle class.
        args:       Parsed CLI arguments.
        logger:     structlog logger instance.

    Returns:
        DataFrame with columns ``epoch``, ``loss``, ``mean_q``, ``epsilon``.
    """
    from rubiks_solve.solvers.dqn.model import DuelingDQN
    from rubiks_solve.solvers.dqn.trainer import DQNTrainer, DQNTrainerConfig
    from rubiks_solve.encoding.registry import get_encoder
    from rubiks_solve.utils.rng import make_rng
    encoder = get_encoder("one_hot", puzzle_cls)
    rng = make_rng(args.seed)
    from rubiks_solve.env.reward import DenseReward
    reward_fn = DenseReward()

    # Lightweight env adapter: DQNTrainer calls reset() -> puzzle,
    # step(puzzle, move) -> (next, reward, done), legal_moves() -> list[Move]
    class _SimpleEnv:
        def __init__(self) -> None:
            self._rng = rng
            self._scramble_depth = min(args.max_scramble, puzzle_cls.move_limit())

        def reset(self):
            """Return a freshly scrambled puzzle state."""
            depth = max(1, int(self._rng.integers(1, self._scramble_depth + 1)))
            return puzzle_cls.solved_state().scramble(depth, self._rng)

        def step(self, puzzle, move):
            """Apply move and return (next_state, reward, done)."""
            next_puzzle = puzzle.apply_move(move)
            reward = reward_fn(puzzle, move, next_puzzle)
            done = next_puzzle.is_solved
            return next_puzzle, reward, done

        def legal_moves(self):
            """Return the list of legal moves for this puzzle type."""
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
        "Starting DQN training",
        epochs=args.epochs,
        steps_per_epoch=trainer_cfg.steps_per_epoch,
        n_actions=n_actions,
    )
    history = trainer.train()

    df = pd.DataFrame(history)
    df.insert(0, "epoch", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, run training, save metrics, and render a loss plot."""
    args = parse_args()

    from rubiks_solve.utils.logging_config import configure_logging, get_logger

    configure_logging(level=args.log_level)
    logger = get_logger(__name__)

    # Optionally load AppConfig (CLI flags take priority)
    if args.config is not None:
        from rubiks_solve.utils.config import AppConfig

        app_cfg = AppConfig.from_file(args.config)
        if args.seed == 42:
            args.seed = app_cfg.training.seed
        if args.output_dir == Path("models"):
            args.output_dir = app_cfg.training.checkpoint_dir
        # Config file may set backend; CLI --backend flag overrides it.
        if args.backend == "local" and app_cfg.compute.backend == "dgx":
            args.backend = "dgx"

    # --- DGX delegation ---
    # When --backend dgx is requested, hand off to remote_train.delegate() and exit.
    if args.backend == "dgx":
        if args.solver in ("genetic", "mcts"):
            logger.warning(  # type: ignore[union-attr]
                "DGX backend does not support solver '%s' — falling back to local.",
                args.solver,
            )
            args.backend = "local"
        else:
            logger.info(  # type: ignore[union-attr]
                "Delegating training to DGX Spark",
                host="msp-spark-01.tail521f18.ts.net",
                solver=args.solver,
                puzzle=args.puzzle,
            )
            sys.path.insert(0, str(Path(__file__).parent))
            from remote_train import delegate  # type: ignore[import]

            delegate(args)
            return

    # Set global seed
    from rubiks_solve.utils.rng import set_global_seed

    set_global_seed(args.seed)

    puzzle_registry = _build_puzzle_registry()
    puzzle_cls = puzzle_registry[args.puzzle]

    logger.info(
        "Training started",
        solver=args.solver,
        puzzle=args.puzzle,
        epochs=args.epochs,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = args.output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if args.solver == "genetic":
        metrics_df = _train_genetic(puzzle_cls, args, logger)
        _plot_col = "generation"
    elif args.solver == "cnn":
        metrics_df = _train_cnn(puzzle_cls, args, logger)
        _plot_col = "epoch"
    elif args.solver == "policy":
        metrics_df = _train_policy(puzzle_cls, args, logger)
        _plot_col = "epoch"
    elif args.solver == "dqn":
        metrics_df = _train_dqn(puzzle_cls, args, logger)
        _plot_col = "epoch"
    else:
        logger.error("Solver training not yet implemented", solver=args.solver)  # type: ignore[union-attr]
        sys.exit(1)

    # Save metrics CSV
    csv_path = metrics_dir / f"{args.solver}_{args.puzzle}_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    logger.info("Metrics saved", path=str(csv_path))  # type: ignore[union-attr]

    # Render training plot
    try:
        from rubiks_solve.visualization.training_plots import (
            plot_fitness_curve,
            plot_loss_curve,
        )

        plot_path = metrics_dir / f"{args.solver}_{args.puzzle}_plot.png"
        solver_label = f"{args.solver.upper()} ({args.puzzle})"
        if args.solver == "genetic" and "fitness_best" in metrics_df.columns:
            plot_fitness_curve(
                metrics_df,
                output_path=plot_path,
                stagnation_events=metrics_df.attrs.get("stagnation_events"),
            )
        elif args.solver in ("cnn", "policy", "dqn") and "loss" in metrics_df.columns:
            plot_loss_curve(metrics_df, solver_name=solver_label, output_path=plot_path)
        logger.info("Training plot saved", path=str(plot_path))  # type: ignore[union-attr]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not save training plot", error=str(exc))  # type: ignore[union-attr]

    logger.info("Training complete")  # type: ignore[union-attr]


if __name__ == "__main__":
    main()
