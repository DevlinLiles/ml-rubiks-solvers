#!/usr/bin/env bash
# setup.sh — one-time environment setup for rubiks-solve
#
# What this does:
#   1. Checks for required system tools (brew, sshpass)
#   2. Creates/updates the Python virtual environment
#   3. Installs all Python dependencies (MLX + PyTorch/Fabric for DGX)
#   4. Installs the project in editable mode
#   5. Creates .env from .env.example if it doesn't exist
#   6. Adds the DGX Spark host key to ~/.ssh/known_hosts (eliminates the
#      "Host key verification failed" error on first rsync/SSH)
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# Re-running is safe — all steps are idempotent.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[setup]${NC} $*"; }
success() { echo -e "${GREEN}[setup]${NC} $*"; }
warn()    { echo -e "${YELLOW}[setup]${NC} $*"; }
die()     { echo -e "${RED}[setup] ERROR:${NC} $*" >&2; exit 1; }

# Use venv binaries directly — never rely on PATH from `source activate` in scripts.
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PIP="$VENV_DIR/bin/pip"
VENV_PYTHON="$VENV_DIR/bin/python"

# ---------------------------------------------------------------------------
# 1. System tools
# ---------------------------------------------------------------------------

info "Checking system tools..."

if ! command -v brew &>/dev/null; then
    die "Homebrew is not installed. Install it from https://brew.sh then re-run."
fi

if ! command -v sshpass &>/dev/null; then
    info "Installing sshpass (needed for rsync to DGX)..."
    brew install sshpass
else
    success "sshpass already installed"
fi

if ! command -v rsync &>/dev/null; then
    info "Installing rsync..."
    brew install rsync
else
    success "rsync already installed"
fi

# ---------------------------------------------------------------------------
# 2. Python virtual environment
# ---------------------------------------------------------------------------

info "Setting up Python virtual environment..."

# Prefer a modern Python (3.11+). Check brew-installed python first, then fall back.
if command -v python3.12 &>/dev/null; then
    PYTHON="$(command -v python3.12)"
elif command -v python3.11 &>/dev/null; then
    PYTHON="$(command -v python3.11)"
elif command -v "$(brew --prefix 2>/dev/null)/bin/python3" &>/dev/null; then
    PYTHON="$(brew --prefix)/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
else
    die "Python 3 not found. Install via: brew install python"
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Using Python $PY_VERSION at $PYTHON"

# Warn if Python is too old (project requires 3.11+)
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")
if [[ "$PY_MINOR" -lt 11 ]]; then
    warn "Python $PY_VERSION is older than the required 3.11. Install a newer version:"
    warn "  brew install python@3.12"
fi

# Recreate the venv if it doesn't exist OR if its pip is missing/broken
# (happens when the venv was created with a different Python version).
NEEDS_VENV=false
if [ ! -d "$VENV_DIR" ]; then
    NEEDS_VENV=true
elif [ ! -x "$VENV_PIP" ]; then
    warn ".venv exists but pip is missing (likely created with a different Python). Recreating..."
    rm -rf "$VENV_DIR"
    NEEDS_VENV=true
elif ! "$VENV_PYTHON" -c "import sys; assert sys.version_info >= (3,11)" 2>/dev/null; then
    warn ".venv Python is older than 3.11. Recreating with $PYTHON..."
    rm -rf "$VENV_DIR"
    NEEDS_VENV=true
fi

if [ "$NEEDS_VENV" = true ]; then
    info "Creating .venv with $PYTHON..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    info ".venv is valid — updating packages"
fi

success "Virtual environment ready at $VENV_DIR"

# ---------------------------------------------------------------------------
# 3. Install Python dependencies
# ---------------------------------------------------------------------------

info "Upgrading pip..."
"$VENV_PIP" install --upgrade pip --quiet

info "Installing base + ML requirements (MLX, PyTorch tools)..."
"$VENV_PIP" install --quiet -r requirements/base.txt
"$VENV_PIP" install --quiet -r requirements/ml.txt

info "Installing DGX delegation tools (fabric, python-dotenv)..."
# torch itself is only needed on the DGX — skip it here, install the rest.
"$VENV_PIP" install --quiet fabric python-dotenv \
    || warn "Some DGX delegation tools failed to install — check output above"

info "Installing visualization dependencies..."
"$VENV_PIP" install --quiet -r requirements/viz.txt 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Install project in editable mode
# ---------------------------------------------------------------------------

info "Installing rubiks-solve in editable mode..."
"$VENV_PIP" install --quiet -e .
success "Project installed — CLI commands available: rubiks-train, rubiks-train-all, etc."

# ---------------------------------------------------------------------------
# 5. Environment file
# ---------------------------------------------------------------------------

if [ ! -f ".env" ]; then
    info "Creating .env from .env.example — fill in DGX_USER and DGX_PASSWORD"
    cp .env.example .env
    warn ".env created. Edit it and set DGX_USER and DGX_PASSWORD before running with --backend dgx"
else
    success ".env already exists"
fi

# ---------------------------------------------------------------------------
# 6. Trust the DGX Spark host key
# ---------------------------------------------------------------------------

# Load .env so we can read DGX_HOST
if [ -f ".env" ]; then
    # Export only the DGX_ variables (avoids polluting the shell with everything)
    set -a
    # shellcheck disable=SC2046
    export $(grep -E '^DGX_' .env | grep -v '^#' | xargs) 2>/dev/null || true
    set +a
fi

DGX_HOST="${DGX_HOST:-msp-spark-01.tail521f18.ts.net}"

info "Adding DGX Spark host key to ~/.ssh/known_hosts ($DGX_HOST)..."
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Remove any stale entry first, then re-scan
ssh-keygen -R "$DGX_HOST" 2>/dev/null || true
if ssh-keyscan -T 10 "$DGX_HOST" >> ~/.ssh/known_hosts 2>/dev/null; then
    success "Host key for $DGX_HOST added to known_hosts"
else
    warn "Could not reach $DGX_HOST — host key NOT added."
    warn "Make sure you're connected to Tailscale, then re-run setup.sh or run:"
    warn "  ssh-keyscan $DGX_HOST >> ~/.ssh/known_hosts"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
success "Setup complete!"
echo ""
echo "  Activate venv:          source .venv/bin/activate"
echo "  Local training:         ./run.sh"
echo "  DGX training (normal):  ./run.sh --backend dgx"
echo "  DGX training (heavy):   ./run.sh --backend dgx --mode heavy"
echo ""
