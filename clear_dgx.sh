#!/usr/bin/env bash
# clear_dgx.sh — clean up project files and/or models on the DGX Spark
#
# Usage:
#   ./clear_dgx.sh                  # remove models and metrics only (keep venv + source)
#   ./clear_dgx.sh --all            # remove everything (source, models, venv)
#   ./clear_dgx.sh --models         # remove models only
#   ./clear_dgx.sh --venv           # remove venv only (forces full reinstall on next run)
#
# Options:
#   --models    Delete ~/rubiks-solve/models/ on DGX
#   --venv      Delete the Python venv on DGX
#   --source    Delete the synced project source on DGX
#   --all       Delete everything (models + venv + source)
#   -y          Skip confirmation prompt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[clear_dgx]${NC} $*"; }
success() { echo -e "${GREEN}[clear_dgx]${NC} $*"; }
warn()    { echo -e "${YELLOW}[clear_dgx]${NC} $*"; }
die()     { echo -e "${RED}[clear_dgx] ERROR:${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

CLEAR_MODELS=false
CLEAR_VENV=false
CLEAR_SOURCE=false
SKIP_CONFIRM=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    # Default: models only
    CLEAR_MODELS=true
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)    CLEAR_MODELS=true; CLEAR_VENV=true; CLEAR_SOURCE=true; shift ;;
        --models) CLEAR_MODELS=true; shift ;;
        --venv)   CLEAR_VENV=true;   shift ;;
        --source) CLEAR_SOURCE=true; shift ;;
        -y)       SKIP_CONFIRM=true; shift ;;
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            die "Unknown option: $1. Run './clear_dgx.sh --help' for usage."
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------

if [ ! -f ".env" ]; then
    die ".env not found. Run ./setup.sh first."
fi

set -a
# shellcheck disable=SC2046
export $(grep -E '^DGX_' .env | grep -v '^#' | xargs) 2>/dev/null || true
set +a

DGX_HOST="${DGX_HOST:-msp-spark-01.tail521f18.ts.net}"
DGX_USER="${DGX_USER:-}"
DGX_PASSWORD="${DGX_PASSWORD:-}"

# Expand ${DGX_USER} references
DGX_REMOTE_DIR="$(echo "${DGX_REMOTE_DIR:-/home/\${DGX_USER}/rubiks-solve}" | envsubst 2>/dev/null || eval echo "${DGX_REMOTE_DIR:-/home/${DGX_USER}/rubiks-solve}")"
DGX_VENV_PATH="$(echo "${DGX_VENV_PATH:-/home/\${DGX_USER}/rubiks-venv}" | envsubst 2>/dev/null || eval echo "${DGX_VENV_PATH:-/home/${DGX_USER}/rubiks-venv}")"

[[ -z "$DGX_USER" ]]     && die "DGX_USER is not set in .env"
[[ -z "$DGX_PASSWORD" ]] && die "DGX_PASSWORD is not set in .env"

# ---------------------------------------------------------------------------
# Build list of what will be deleted
# ---------------------------------------------------------------------------

TARGETS=()
$CLEAR_MODELS && TARGETS+=("$DGX_REMOTE_DIR/models  (checkpoints + metrics)")
$CLEAR_VENV   && TARGETS+=("$DGX_VENV_PATH          (Python virtual environment)")
$CLEAR_SOURCE && TARGETS+=("$DGX_REMOTE_DIR          (full project source)")

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    die "Nothing selected to clear. Use --models, --venv, --source, or --all."
fi

# ---------------------------------------------------------------------------
# Confirm
# ---------------------------------------------------------------------------

echo ""
warn "The following will be DELETED on $DGX_USER@$DGX_HOST:"
for t in "${TARGETS[@]}"; do
    echo "    $t"
done
echo ""

if [[ "$SKIP_CONFIRM" == false ]]; then
    read -r -p "Continue? [y/N] " REPLY
    if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# Build remote cleanup command
# ---------------------------------------------------------------------------

REMOTE_CMD=""

if $CLEAR_MODELS; then
    REMOTE_CMD+="echo 'Removing models...' && rm -rf '$DGX_REMOTE_DIR/models' '$DGX_REMOTE_DIR/logs' && echo 'Done: models'; "
fi

if $CLEAR_VENV && ! $CLEAR_SOURCE; then
    REMOTE_CMD+="echo 'Removing venv...' && rm -rf '$DGX_VENV_PATH' && echo 'Done: venv'; "
fi

if $CLEAR_SOURCE; then
    # If removing source, also remove venv regardless of --venv flag
    REMOTE_CMD+="echo 'Removing project source and venv...' && rm -rf '$DGX_REMOTE_DIR' '$DGX_VENV_PATH' && echo 'Done: source + venv'; "
elif $CLEAR_VENV; then
    : # already handled above
fi

# ---------------------------------------------------------------------------
# Execute via SSH
# ---------------------------------------------------------------------------

info "Connecting to $DGX_USER@$DGX_HOST..."

sshpass -p "$DGX_PASSWORD" ssh \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile=~/.ssh/known_hosts \
    "$DGX_USER@$DGX_HOST" \
    "bash -c \"$REMOTE_CMD\""

echo ""
success "DGX cleanup complete."

if $CLEAR_VENV || $CLEAR_SOURCE; then
    info "The venv was removed — it will be recreated automatically on the next ./run.sh --backend dgx"
fi
