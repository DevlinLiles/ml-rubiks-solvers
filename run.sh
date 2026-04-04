#!/usr/bin/env bash
# run.sh — run Rubik's cube ML training locally (MLX) or on the DGX Spark (PyTorch)
#
# Usage:
#   ./run.sh                                    # local, normal mode, all solvers + puzzles
#   ./run.sh --backend dgx                      # DGX, connected (blocks until done)
#   ./run.sh --backend dgx --detach             # DGX, fire-and-forget (returns immediately)
#   ./run.sh --backend dgx --gather             # reconnect and pull all finished runs
#   ./run.sh --backend dgx --gather RUN_ID      # check and pull one specific run
#   ./run.sh --backend dgx --gather --tail 50   # gather with last 50 log lines per run
#   ./run.sh --backend dgx --mode heavy         # DGX, heavy training
#   ./run.sh --backend local --mode heavy       # local, heavy training
#   ./run.sh --solvers cnn,dqn --puzzles 3x3    # subset
#   ./run.sh --backend dgx --solvers cnn --puzzles 3x3,4x4 --mode heavy
#
# Training modes:
#   normal  epochs=100   n-train=10,000  batch-size=512    (quick validation run)
#   heavy   epochs=500   n-train=100,000 batch-size=1024   (full production training)
#
# Options:
#   --backend  local|dgx              (default: local)
#   --mode     normal|heavy           (default: normal)
#   --solvers  comma-separated list   (default: cnn,policy,dqn,genetic)
#   --puzzles  comma-separated list   (default: 2x2,3x3,4x4,5x5,megaminx)
#   --log-level DEBUG|INFO|WARNING    (default: INFO)
#   --detach                          DGX only: launch jobs in background, return immediately
#   --gather [RUN_ID]                 DGX only: reconnect and pull finished detached runs
#   --tail N                          With --gather: print last N lines of each run's log
#   --full-checkpoints                With --gather: pull all checkpoints (default: latest only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BACKEND="local"
MODE="normal"
SOLVERS="cnn,policy,dqn,genetic"
PUZZLES="2x2,3x3,4x4,5x5,megaminx"
LOG_LEVEL="INFO"
DETACH=false
GATHER=false
GATHER_RUN_ID=""
TAIL=0
FULL_CHECKPOINTS=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)   BACKEND="$2";   shift 2 ;;
        --mode)      MODE="$2";      shift 2 ;;
        --solvers)   SOLVERS="$2";   shift 2 ;;
        --puzzles)   PUZZLES="$2";   shift 2 ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        --detach)    DETACH=true;    shift ;;
        --tail)             TAIL="$2";             shift 2 ;;
        --full-checkpoints) FULL_CHECKPOINTS=true; shift ;;
        --gather)
            GATHER=true
            # Optional RUN_ID argument: consume next arg if it doesn't start with --
            if [[ $# -gt 1 && "${2:-}" != --* ]]; then
                GATHER_RUN_ID="$2"
                shift
            fi
            shift
            ;;
        -h|--help)
            sed -n '2,35p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Run './run.sh --help' for usage." >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

if [[ "$BACKEND" != "local" && "$BACKEND" != "dgx" ]]; then
    echo "Error: --backend must be 'local' or 'dgx'" >&2; exit 1
fi

if [[ "$GATHER" == true && "$BACKEND" != "dgx" ]]; then
    echo "Error: --gather requires --backend dgx" >&2; exit 1
fi

if [[ "$DETACH" == true && "$BACKEND" != "dgx" ]]; then
    echo "Error: --detach requires --backend dgx" >&2; exit 1
fi

if [[ "$GATHER" == true && "$DETACH" == true ]]; then
    echo "Error: --gather and --detach are mutually exclusive" >&2; exit 1
fi

if [[ "$MODE" != "normal" && "$MODE" != "heavy" ]]; then
    echo "Error: --mode must be 'normal' or 'heavy'" >&2; exit 1
fi

# ---------------------------------------------------------------------------
# Activate venv
# ---------------------------------------------------------------------------

VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: .venv not found. Run ./setup.sh first." >&2
    exit 1
fi

export PATH="$VENV_DIR/bin:$PATH"

# ---------------------------------------------------------------------------
# DGX pre-flight checks (skip for gather — only needs SSH, not sshpass)
# ---------------------------------------------------------------------------

if [[ "$BACKEND" == "dgx" ]]; then
    if [ ! -f ".env" ]; then
        echo "Error: .env not found. Run ./setup.sh and fill in DGX_USER + DGX_PASSWORD." >&2
        exit 1
    fi

    set -a
    # shellcheck disable=SC2046
    export $(grep -E '^DGX_' .env | grep -v '^#' | xargs) 2>/dev/null || true
    set +a

    if [[ -z "${DGX_USER:-}" ]]; then
        echo "Error: DGX_USER is not set in .env" >&2; exit 1
    fi
    if [[ -z "${DGX_PASSWORD:-}" ]]; then
        echo "Error: DGX_PASSWORD is not set in .env" >&2; exit 1
    fi

    if [[ "$GATHER" == false ]] && ! command -v sshpass &>/dev/null; then
        echo "Error: sshpass not found. Run ./setup.sh to install it." >&2; exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Gather mode — reconnect and pull finished detached runs
# ---------------------------------------------------------------------------

if [[ "$GATHER" == true ]]; then
    echo ""
    echo "=========================================="
    echo "  rubiks-solve gather"
    echo "=========================================="
    if [[ -n "$GATHER_RUN_ID" ]]; then
        echo "  Run ID   : $GATHER_RUN_ID"
    else
        echo "  Run ID   : (all running)"
    fi
    if [[ "$TAIL" -gt 0 ]]; then
        echo "  Tail     : $TAIL lines"
    fi
    echo "=========================================="
    echo ""

    GATHER_ARGS=(--backend dgx --gather)
    [[ -n "$GATHER_RUN_ID" ]]       && GATHER_ARGS+=("$GATHER_RUN_ID")
    [[ "$TAIL" -gt 0 ]]             && GATHER_ARGS+=(--tail "$TAIL")
    [[ "$FULL_CHECKPOINTS" == true ]] && GATHER_ARGS+=(--full-checkpoints)
    GATHER_ARGS+=(--log-level "$LOG_LEVEL")

    rubiks-train-all "${GATHER_ARGS[@]}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Training hyperparameters by mode
# ---------------------------------------------------------------------------

if [[ "$MODE" == "normal" ]]; then
    EPOCHS=100
    N_TRAIN=10000
    BATCH_SIZE=512
else
    EPOCHS=500
    N_TRAIN=100000
    BATCH_SIZE=1024
fi

# ---------------------------------------------------------------------------
# Print run summary
# ---------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "  rubiks-solve training run"
echo "=========================================="
echo "  Backend  : $BACKEND"
echo "  Mode     : $MODE"
echo "  Detach   : $DETACH"
echo "  Solvers  : $SOLVERS"
echo "  Puzzles  : $PUZZLES"
echo "  Epochs   : $EPOCHS"
echo "  N-train  : $N_TRAIN"
echo "  Batch    : $BATCH_SIZE"
echo "  Log level: $LOG_LEVEL"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

TRAIN_ARGS=(
    --backend "$BACKEND"
    --solvers "$SOLVERS"
    --puzzles "$PUZZLES"
    --epochs "$EPOCHS"
    --n-train "$N_TRAIN"
    --batch-size "$BATCH_SIZE"
    --log-level "$LOG_LEVEL"
)

[[ "$DETACH" == true ]] && TRAIN_ARGS+=(--detach)

rubiks-train-all "${TRAIN_ARGS[@]}"
