"""Delegate training to the NVIDIA DGX Spark via SSH (Fabric).

Reads connection settings from .env:
    DGX_HOST          — Tailscale hostname or IP  (e.g. msp-spark-01.tail521f18.ts.net)
    DGX_USER          — SSH username
    DGX_PASSWORD      — SSH password
    DGX_REMOTE_DIR    — Absolute path on DGX for project files  (default ~/rubiks-solve)
    DGX_VENV_PATH     — Absolute path to Python venv on DGX     (default ~/rubiks-venv)
    DGX_TORCH_INDEX_URL — PyTorch wheel index for CUDA 13 aarch64

Workflow:
    1. rsync project source to DGX_REMOTE_DIR (excludes .git, __pycache__, models/).
    2. Create or refresh the Python venv and install requirements.
    3. Run scripts/train_torch.py on DGX with the forwarded CLI args.
    4. rsync checkpoints and metrics back to the local output dir.

Usage (called by train.py when --backend dgx):
    from scripts.remote_train import delegate
    delegate(args)

Or standalone:
    python scripts/remote_train.py --solver cnn --puzzle 3x3 --epochs 50
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

_ENV_LOADED = False


def _load_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        _ENV_LOADED = True
    except ImportError:
        logger.warning(
            "python-dotenv not installed — reading DGX credentials from environment only. "
            "Install with: pip install python-dotenv"
        )
        _ENV_LOADED = True


def _expand(value: str) -> str:
    """Expand $VAR / ${VAR} references using the current environment.

    python-dotenv loads values as literal strings — shell variable references
    like ${DGX_USER} in .env are NOT expanded automatically.  This helper
    applies os.path.expandvars() so that paths like /home/${DGX_USER}/rubiks-venv
    resolve correctly once DGX_USER is in the environment.
    """
    return os.path.expandvars(value)


def _require_env(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable {key!r} is not set. "
            "Add it to your .env file (see .env.example)."
        )
    return value


# ---------------------------------------------------------------------------
# Fabric connection helpers
# ---------------------------------------------------------------------------


def _get_connection():
    """Return an authenticated Fabric SSH connection to the DGX Spark."""
    try:
        from fabric import Connection
    except ImportError as exc:
        raise ImportError(
            "fabric is required for DGX training delegation. "
            "Install with: pip install fabric"
        ) from exc

    _load_env()
    host = _require_env("DGX_HOST")
    user = _require_env("DGX_USER")
    password = _require_env("DGX_PASSWORD")

    import paramiko

    conn = Connection(
        host=host,
        user=user,
        connect_kwargs={
            "password": password,
            "look_for_keys": False,
            "allow_agent": False,
        },
    )
    # Accept new host keys on first connection (adds to ~/.ssh/known_hosts).
    conn.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # Open the connection now so we can configure keepalive on the transport.
    # This prevents SSH sessions from being dropped during long training runs.
    conn.open()
    transport = conn.client.get_transport()
    if transport:
        transport.set_keepalive(30)  # send keepalive packet every 30 seconds
    return conn


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _rsync_project(conn, local_root: Path, remote_dir: str) -> None:
    """Push project source to the DGX Spark via rsync over SSH."""
    _load_env()
    host = _require_env("DGX_HOST")
    user = _require_env("DGX_USER")
    password = _require_env("DGX_PASSWORD")

    # Build rsync command using sshpass to pass the password non-interactively.
    # Excludes: git history, pycache, local models dir, venvs, and IDE files.
    exclude_flags = " ".join([
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude='*.pyc'",
        "--exclude='.mypy_cache'",
        "--exclude='.pytest_cache'",
        "--exclude='models/'",
        "--exclude='logs/'",
        "--exclude='.venv'",
        "--exclude='*.egg-info'",
    ])
    ssh_opts = "-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=~/.ssh/known_hosts"
    rsync_cmd = (
        f"sshpass -p '{password}' rsync -avz --delete {exclude_flags} "
        f"-e 'ssh {ssh_opts}' "
        f"{local_root}/ {user}@{host}:{remote_dir}/"
    )
    logger.info("Syncing project to DGX Spark at %s:%s", host, remote_dir)
    ret = os.system(rsync_cmd)
    if ret != 0:
        raise RuntimeError(
            f"rsync failed (exit code {ret}). "
            "Ensure sshpass is installed: brew install sshpass"
        )


def _ensure_venv(conn, venv_path: str, torch_index_url: str) -> None:
    """Create the Python venv on DGX and install required packages."""
    logger.info("Ensuring Python venv at %s on DGX Spark", venv_path)
    conn.run(f"python3 -m venv {venv_path}", warn=True)
    pip = f"{venv_path}/bin/pip"
    conn.run(f"{pip} install --upgrade pip --quiet")
    # Install base and torch requirements.
    conn.run(
        f"{pip} install --quiet "
        f"--extra-index-url {torch_index_url} "
        f"-r ~/rubiks-solve/requirements/torch.txt",
        warn=False,
    )
    logger.info("DGX venv ready")


# ---------------------------------------------------------------------------
# Training delegation
# ---------------------------------------------------------------------------


def _build_train_cmd(venv_path: str, remote_dir: str, args: argparse.Namespace) -> str:
    """Construct the remote python command that runs train_torch.py."""
    python = f"{venv_path}/bin/python"
    parts = [
        python,
        f"{remote_dir}/scripts/train_torch.py",
        f"--solver {args.solver}",
        f"--puzzle {args.puzzle}",
        f"--epochs {args.epochs}",
        f"--seed {args.seed}",
        f"--n-train {args.n_train}",
        f"--max-scramble {args.max_scramble}",
        f"--batch-size {args.batch_size}",
        f"--output-dir {remote_dir}/models",
        f"--log-level {args.log_level}",
    ]
    return " ".join(parts)


def _pull_results(conn, remote_dir: str, local_output_dir: Path) -> None:
    """rsync checkpoints and metrics from DGX back to local machine."""
    _load_env()
    host = _require_env("DGX_HOST")
    user = _require_env("DGX_USER")
    password = _require_env("DGX_PASSWORD")

    local_output_dir.mkdir(parents=True, exist_ok=True)
    ssh_opts = "-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=~/.ssh/known_hosts"
    rsync_cmd = (
        f"sshpass -p '{password}' rsync -avz "
        f"-e 'ssh {ssh_opts}' "
        f"{user}@{host}:{remote_dir}/models/ {local_output_dir}/"
    )
    logger.info("Pulling models from DGX to %s", local_output_dir)
    ret = os.system(rsync_cmd)
    if ret != 0:
        logger.warning("rsync pull returned exit code %d — check DGX output dir", ret)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def delegate(args: argparse.Namespace) -> None:
    """Delegate a training run to the DGX Spark.

    Args:
        args: Parsed CLI arguments (same shape as train.py's parse_args()).
              Must include: solver, puzzle, epochs, seed, n_train, max_scramble,
              batch_size, output_dir, log_level.
    """
    _load_env()

    remote_dir = _expand(os.environ.get("DGX_REMOTE_DIR", "~/rubiks-solve"))
    venv_path = _expand(os.environ.get("DGX_VENV_PATH", "~/rubiks-venv"))
    torch_index_url = os.environ.get(
        "DGX_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu130"
    )

    local_root = Path(__file__).parent.parent.resolve()

    # Step 1: push source
    _rsync_project(None, local_root, remote_dir)  # rsync uses sshpass directly

    # Step 2: open SSH connection and prepare venv
    conn = _get_connection()
    with conn:
        _ensure_venv(conn, venv_path, torch_index_url)

        # Step 3: run training
        cmd = _build_train_cmd(venv_path, remote_dir, args)
        logger.info("Launching training on DGX Spark: %s", cmd)
        # stream=True pipes remote stdout/stderr to our terminal in real-time
        conn.run(cmd, pty=True)

    # Step 4: pull results back (outside `with conn` is fine — uses sshpass)
    _pull_results(None, remote_dir, args.output_dir)
    logger.info("DGX training complete. Checkpoints saved to %s", args.output_dir)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="remote-train",
        description="Delegate rubiks-solve training to the DGX Spark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--solver", choices=["cnn", "policy", "dqn"], required=True)
    parser.add_argument(
        "--puzzle", choices=["2x2", "3x3", "4x4", "5x5", "megaminx"], default="3x3"
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    delegate(_parse_args())
