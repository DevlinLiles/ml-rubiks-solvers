"""Delegate training to the NVIDIA DGX Spark via SSH (Fabric).

Reads connection settings from .env:
    DGX_HOST          — Tailscale hostname or IP  (e.g. msp-spark-01.tail521f18.ts.net)
    DGX_USER          — SSH username
    DGX_PASSWORD      — SSH password
    DGX_REMOTE_DIR    — Absolute path on DGX for project files  (default ~/rubiks-solve)
    DGX_VENV_PATH     — Absolute path to Python venv on DGX     (default ~/rubiks-venv)
    DGX_TORCH_INDEX_URL — PyTorch wheel index for CUDA 13 aarch64

Workflow (connected):
    1. rsync project source to DGX_REMOTE_DIR (excludes .git, __pycache__, models/).
    2. Create or refresh the Python venv and install requirements.
    3. Run scripts/train_torch.py on DGX with the forwarded CLI args.
    4. rsync checkpoints and metrics back to the local output dir.

Workflow (detached):
    Steps 1-2 same, then launch with nohup and return immediately.
    A JSON run manifest is written to runs/<run_id>.json locally.
    Use --gather [run_id] later to check status and pull results.

Usage (called by train.py when --backend dgx):
    from scripts.remote_train import delegate
    delegate(args)

Or standalone:
    python scripts/remote_train.py --solver cnn --puzzle 3x3 --epochs 50
    python scripts/remote_train.py --solver cnn --puzzle 3x3 --epochs 50 --detach
    python scripts/remote_train.py --gather
    python scripts/remote_train.py --gather 20240403T172039_cnn_3x3
    python scripts/remote_train.py --gather --tail 50
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
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
    import socket
    try:
        conn.open()
    except socket.gaierror:
        raise RuntimeError(
            f"Cannot resolve DGX host {host!r}. "
            "Is Tailscale running and connected?\n"
            "  Check with: tailscale status\n"
            "  Connect with: tailscale up"
        ) from None
    transport = conn.client.get_transport()
    if transport:
        transport.set_keepalive(30)  # send keepalive packet every 30 seconds
    return conn


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _rsync_project(_conn, local_root: Path, remote_dir: str) -> None:
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


def _pull_results(conn, remote_dir: str, local_output_dir: Path, full: bool = False) -> None:
    """rsync checkpoints and metrics from DGX back to local machine.

    Args:
        conn:             Active Fabric SSH connection, used to locate the latest
                          checkpoint when ``full=False``.  May be ``None`` only
                          when ``full=True``.
        remote_dir:       Absolute path to the project root on DGX.
        local_output_dir: Local directory to write results into.
        full:             If ``True``, pull every checkpoint.  If ``False``
                          (default), pull only the most recent checkpoint per
                          solver/puzzle subdirectory plus all metrics files.
    """
    _load_env()
    host = _require_env("DGX_HOST")
    user = _require_env("DGX_USER")
    password = _require_env("DGX_PASSWORD")

    local_output_dir.mkdir(parents=True, exist_ok=True)
    ssh_opts = "-o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=~/.ssh/known_hosts"
    remote_models = f"{remote_dir}/models"

    if full:
        rsync_cmd = (
            f"sshpass -p '{password}' rsync -avz "
            f"-e 'ssh {ssh_opts}' "
            f"{user}@{host}:{remote_models}/ {local_output_dir}/"
        )
        logger.info("Pulling all checkpoints from DGX to %s", local_output_dir)
        ret = os.system(rsync_cmd)
        if ret != 0:
            logger.warning("rsync pull returned exit code %d — check DGX output dir", ret)
        return

    # --- Partial pull: metrics + latest checkpoint per solver/puzzle subdir ---

    # Use SSH to find the newest .npz per subdirectory (names are zero-padded,
    # so lexicographic sort gives chronological order).
    find_cmd = (
        f"find {remote_models} -name 'ckpt_*.npz' | sort | "
        f"awk '{{dir=$0; sub(/\\/ckpt_[^\\/]*$/, \"\", dir); latest[dir]=$0}} "
        f"END{{for (d in latest) print latest[d]}}'"
    )
    result = conn.run(find_cmd, hide=True, warn=True)
    latest_npz = [p.strip() for p in result.stdout.splitlines() if p.strip()]

    if not latest_npz:
        logger.info("No checkpoints found on DGX — pulling metrics only")

    # Build rsync filter rules: always include metrics/, include only the
    # specific latest checkpoint files, exclude everything else.
    filters: list[str] = ["--include='metrics/***'"]
    for npz_path in latest_npz:
        # Convert absolute DGX path to a path relative to remote_models/
        rel = npz_path.removeprefix(remote_models).lstrip("/")
        parts = rel.split("/")   # e.g. ['cnn', 'megaminx', 'ckpt_epoch0499_...npz']
        stem = parts[-1].replace(".npz", "")
        # Include each directory component leading to the file.
        for depth in range(1, len(parts)):
            filters.append(f"--include='{'/'.join(parts[:depth])}/'")
        filters.append(f"--include='{rel}'")
        filters.append(f"--include='{'/'.join(parts[:-1])}/{stem}.json'")
        filters.append(f"--exclude='{'/'.join(parts[:-1])}/ckpt_*'")

    filters.append("--exclude='ckpt_*'")  # catch any remaining checkpoint files
    filters.append("--include='*/'")      # traverse other directories
    filters.append("--exclude='*'")       # exclude everything else not matched

    filter_str = " ".join(filters)
    rsync_cmd = (
        f"sshpass -p '{password}' rsync -avz {filter_str} "
        f"-e 'ssh {ssh_opts}' "
        f"{user}@{host}:{remote_models}/ {local_output_dir}/"
    )
    logger.info(
        "Pulling latest checkpoint(s) + metrics from DGX to %s (%d checkpoint(s))",
        local_output_dir, len(latest_npz),
    )
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
# Run manifest helpers (detached mode)
# ---------------------------------------------------------------------------

_RUNS_DIR = Path(__file__).parent.parent / "runs"
_COMPLETION_MARKER = "DGX training complete"


def _runs_dir() -> Path:
    _RUNS_DIR.mkdir(exist_ok=True)
    return _RUNS_DIR


def _manifest_path(run_id: str) -> Path:
    return _runs_dir() / f"{run_id}.json"


def _save_manifest(manifest: dict) -> Path:
    path = _manifest_path(manifest["run_id"])
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def _load_manifest(run_id: str) -> dict:
    path = _manifest_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"No run manifest found for run_id={run_id!r}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _list_manifests(status: str | None = None) -> list[dict]:
    """Return all saved run manifests, optionally filtered by status."""
    results = []
    for path in sorted(_runs_dir().glob("*.json")):
        try:
            with path.open(encoding="utf-8") as fh:
                manifest = json.load(fh)
            if status is None or manifest.get("status") == status:
                results.append(manifest)
        except Exception:  # noqa: BLE001  # pylint: disable=broad-exception-caught
            continue
    return results


# ---------------------------------------------------------------------------
# Detached training
# ---------------------------------------------------------------------------


def delegate_detached(args: argparse.Namespace) -> dict:
    """Start training on DGX via nohup and return immediately.

    Syncs source, ensures the venv, launches ``train_torch.py`` in the
    background, and writes a run manifest to ``runs/<run_id>.json`` so that
    :func:`gather` can reconnect later.

    Args:
        args: Parsed CLI arguments (same shape as ``delegate()``).

    Returns:
        The run manifest dict that was saved locally.
    """
    _load_env()
    remote_dir = _expand(os.environ.get("DGX_REMOTE_DIR", "~/rubiks-solve"))
    venv_path = _expand(os.environ.get("DGX_VENV_PATH", "~/rubiks-venv"))
    torch_index_url = os.environ.get(
        "DGX_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu130"
    )
    local_root = Path(__file__).parent.parent.resolve()

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = f"{ts}_{args.solver}_{args.puzzle}"

    # Step 1: push source
    _rsync_project(None, local_root, remote_dir)

    # Step 2: ensure venv, then launch detached
    conn = _get_connection()
    with conn:
        _ensure_venv(conn, venv_path, torch_index_url)

        log_path = f"{remote_dir}/logs/{run_id}.log"
        conn.run(f"mkdir -p {remote_dir}/logs", hide=True)

        train_cmd = _build_train_cmd(venv_path, remote_dir, args)
        # nohup keeps the process alive after SSH disconnects; echo $! captures PID.
        nohup_cmd = f"nohup {train_cmd} > {log_path} 2>&1 & echo $!"
        result = conn.run(nohup_cmd, hide=True)
        pid = int(result.stdout.strip())

    manifest = {
        "run_id": run_id,
        "solver": args.solver,
        "puzzle": args.puzzle,
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dgx_pid": pid,
        "dgx_log": log_path,
        "remote_dir": remote_dir,
        "status": "running",
    }
    manifest_path = _save_manifest(manifest)
    logger.info(
        "Detached training started: run_id=%s pid=%d log=%s manifest=%s",
        run_id, pid, log_path, manifest_path,
    )
    return manifest


# ---------------------------------------------------------------------------
# Gather (reconnect and collect results)
# ---------------------------------------------------------------------------


def gather(
    run_id: str | None = None,
    output_dir: Path = Path("models"),
    tail: int = 0,
    full: bool = False,
) -> None:
    """Reconnect to the DGX, check run status, and pull finished results.

    For each matching run:
    * If the process is still running: log status and optionally tail the log.
    * If it has stopped: pull results locally and update the manifest.

    Args:
        run_id:     Specific run ID to check.  If ``None``, checks all runs
                    with ``status == "running"``.
        output_dir: Local directory to rsync results into.
        tail:       If > 0, print the last *tail* lines of the remote log.
        full:       If ``True``, pull every checkpoint.  If ``False`` (default),
                    pull only the latest checkpoint per solver/puzzle subdir.
    """
    _load_env()

    if run_id:
        manifests = [_load_manifest(run_id)]
    else:
        manifests = _list_manifests(status="running")

    if not manifests:
        logger.info("No running manifests found in %s", _runs_dir())
        return

    conn = _get_connection()
    with conn:
        for manifest in manifests:
            _gather_one(conn, manifest, output_dir, tail, full)


def _gather_one(conn, manifest: dict, output_dir: Path, tail: int, full: bool) -> None:
    """Check and update a single run manifest."""
    run_id = manifest["run_id"]
    pid = manifest["dgx_pid"]
    log_path = manifest["dgx_log"]
    remote_dir = manifest["remote_dir"]

    # Check whether the process is still alive.
    check = conn.run(
        f"kill -0 {pid} 2>/dev/null && echo running || echo stopped",
        hide=True,
    )
    is_running = check.stdout.strip() == "running"

    if tail > 0:
        tail_result = conn.run(
            f"tail -{tail} {log_path} 2>/dev/null || echo '(log not yet available)'",
            hide=True,
        )
        print(f"\n--- {run_id} (last {tail} lines) ---")
        print(tail_result.stdout.rstrip())
        print("---")

    if is_running:
        logger.info("[%s] status=running  pid=%d", run_id, pid)
        return

    # Process stopped — detect success vs failure.
    grep = conn.run(
        f"grep -c '{_COMPLETION_MARKER}' {log_path} 2>/dev/null || echo 0",
        hide=True,
    )
    success = int(grep.stdout.strip()) > 0
    new_status = "done" if success else "failed"
    logger.info(
        "[%s] status=%s — pulling %s to %s",
        run_id, new_status, "all checkpoints" if full else "latest checkpoint", output_dir,
    )

    _pull_results(conn, remote_dir, output_dir, full=full)

    manifest["status"] = new_status
    manifest["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _save_manifest(manifest)
    logger.info("[%s] Results pulled. Manifest updated.", run_id)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="remote-train",
        description="Delegate rubiks-solve training to the DGX Spark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- gather mode (mutually exclusive with training flags) ---
    parser.add_argument(
        "--gather",
        nargs="?",
        const="",       # --gather with no value → gather all running
        metavar="RUN_ID",
        help=(
            "Check status and pull results for detached runs. "
            "Pass a specific RUN_ID or omit to check all running runs."
        ),
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        metavar="N",
        help="With --gather: print the last N lines of the remote log.",
    )
    parser.add_argument(
        "--full-checkpoints",
        action="store_true",
        help="With --gather: pull every checkpoint instead of only the latest per run.",
    )

    # --- training flags ---
    parser.add_argument("--solver", choices=["cnn", "policy", "dqn"])
    parser.add_argument(
        "--puzzle", choices=["2x2", "3x3", "4x4", "5x5", "megaminx", "skewb_ultimate"], default="3x3"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--n-train", type=int, default=10_000)
    parser.add_argument("--max-scramble", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Launch training in the background on DGX and return immediately.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _args = _parse_args()

    if _args.gather is not None:
        # --gather [RUN_ID]
        _run_id = _args.gather if _args.gather else None
        gather(
            run_id=_run_id,
            output_dir=_args.output_dir,
            tail=_args.tail,
            full=_args.full_checkpoints,
        )
    else:
        if not _args.solver:
            import sys as _sys
            print("error: --solver is required when not using --gather", file=_sys.stderr)
            _sys.exit(1)
        if _args.detach:
            delegate_detached(_args)
        else:
            delegate(_args)
