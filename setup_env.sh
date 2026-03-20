#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Ubuntu/Linux cloud instance bootstrapper
#
# Sets up the complete software environment for the radiation damage pipeline:
#   1. System packages (compilers, MPI, HDF5, FFTW)
#   2. LAMMPS (compiled with ADP + Python bindings)
#   3. Python virtual environment + pip dependencies
#   4. Downloads MACE universal potential (MLIP fallback)
#   5. Creates directory structure
#   6. Optionally runs the full pipeline
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh              # setup only
#   ./setup_env.sh --run        # setup + execute pipeline
#   ./setup_env.sh --run --gpu  # setup + execute with CUDA support
#
# Environment variables (set before running):
#   MP_API_KEY     — Materials Project API key (required)
#   LAMMPS_BRANCH  — LAMMPS git tag, default: stable_29Aug2024
#   CUDA_VERSION   — e.g. "12.1", set to "none" for CPU-only
#   N_WORKERS      — Parallel jobs for compilation, default: nproc
# =============================================================================

set -euo pipefail

# ---- Colour output -----------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---- Parse arguments ---------------------------------------------------------
RUN_PIPELINE=false
USE_GPU=false
for arg in "$@"; do
  case $arg in
    --run) RUN_PIPELINE=true ;;
    --gpu) USE_GPU=true ;;
    *) warn "Unknown argument: $arg" ;;
  esac
done

# ---- Defaults ----------------------------------------------------------------
LAMMPS_BRANCH="${LAMMPS_BRANCH:-stable_29Aug2024}"
CUDA_VERSION="${CUDA_VERSION:-none}"
N_WORKERS="${N_WORKERS:-$(nproc)}"
VENV_DIR="${HOME}/venv/radiation_pipeline"
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POTENTIALS_DIR="${PIPELINE_DIR}/potentials"
DATA_DIR="${PIPELINE_DIR}/data"
LAMMPS_SRC="${HOME}/lammps_src"

info "Pipeline directory : ${PIPELINE_DIR}"
info "Python venv        : ${VENV_DIR}"
info "LAMMPS branch      : ${LAMMPS_BRANCH}"
info "CUDA version       : ${CUDA_VERSION}"
info "Workers            : ${N_WORKERS}"

# ============================================================================
# 1. System packages
# ============================================================================
info "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    curl \
    wget \
    unzip \
    libopenmpi-dev \
    openmpi-bin \
    libhdf5-dev \
    libfftw3-dev \
    libgsl-dev \
    liblapack-dev \
    libblas-dev \
    zlib1g-dev \
    liblzma-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    swig \
    pkg-config \
    htop \
    nvtop || true   # nvtop may not exist on CPU instances

success "System packages installed."

# ============================================================================
# 2. LAMMPS compilation (with ADP pair style + Python bindings)
# ============================================================================
if command -v lmp &>/dev/null; then
    warn "LAMMPS already installed at $(which lmp) — skipping build."
else
    info "Cloning LAMMPS ${LAMMPS_BRANCH}..."
    if [ ! -d "${LAMMPS_SRC}" ]; then
        git clone --depth 1 --branch "${LAMMPS_BRANCH}" \
            https://github.com/lammps/lammps.git "${LAMMPS_SRC}"
    fi

    BUILD_DIR="${LAMMPS_SRC}/build"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX="${HOME}/.local"
        -DBUILD_SHARED_LIBS=ON
        -DPKG_MANYBODY=yes          # EAM, ADP, Finnis-Sinclair
        -DPKG_MISC=yes
        -DPKG_EXTRA-PAIR=yes
        -DPKG_ML-IAP=yes            # Machine Learning IAP package
        -DPKG_PYTHON=yes            # Python bindings
        -DWITH_GZIP=yes
        -DWITH_JPEG=no
        -DWITH_PNG=no
        -DLAMMPS_EXCEPTIONS=yes
    )

    if [ "${CUDA_VERSION}" != "none" ] && [ "$USE_GPU" = true ]; then
        info "Enabling CUDA GPU packages..."
        CMAKE_ARGS+=(
            -DPKG_GPU=yes
            -DGPU_API=cuda
            -DCUDA_MPS_SUPPORT=yes
        )
    fi

    cmake "${LAMMPS_SRC}/cmake" "${CMAKE_ARGS[@]}"
    make -j"${N_WORKERS}"
    make install

    # Add to PATH
    export PATH="${HOME}/.local/bin:${PATH}"
    echo 'export PATH="${HOME}/.local/bin:${PATH}"' >> "${HOME}/.bashrc"
    echo 'export LD_LIBRARY_PATH="${HOME}/.local/lib:${LD_LIBRARY_PATH:-}"' >> "${HOME}/.bashrc"

    success "LAMMPS built and installed."
    cd "${PIPELINE_DIR}"
fi

# ============================================================================
# 3. Python virtual environment
# ============================================================================
if [ ! -d "${VENV_DIR}" ]; then
    info "Creating Python virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip wheel setuptools -q

# ---- PyTorch (GPU or CPU) ---------------------------------------------------
if [ "${CUDA_VERSION}" != "none" ] && [ "$USE_GPU" = true ]; then
    info "Installing PyTorch with CUDA ${CUDA_VERSION}..."
    CUDA_TAG="cu$(echo "${CUDA_VERSION}" | tr -d '.')"
    pip install torch torchvision \
        --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" -q
else
    info "Installing PyTorch (CPU only)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
fi

# ---- PyTorch Geometric ------------------------------------------------------
info "Installing PyTorch Geometric..."
# Use the pre-built wheels from the PyG CDN (matches torch version)
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
if [ "${CUDA_VERSION}" != "none" ] && [ "$USE_GPU" = true ]; then
    CUDA_TAG="cu$(echo "${CUDA_VERSION}" | tr -d '.')"
else
    CUDA_TAG="cpu"
fi
PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f "${PYG_URL}" -q

# ---- Scientific stack -------------------------------------------------------
info "Installing scientific Python packages..."
pip install -r "${PIPELINE_DIR}/requirements.txt" -q

success "Python environment ready. Torch: ${TORCH_VERSION}"

# ============================================================================
# 4. Download MACE universal potential (MLIP fallback for GRACE)
# ============================================================================
mkdir -p "${POTENTIALS_DIR}"

MACE_MODEL="${POTENTIALS_DIR}/mace_mp_medium.model"
if [ ! -f "${MACE_MODEL}" ]; then
    info "Downloading MACE-MP-0 medium checkpoint..."
    python - <<'PYEOF'
from mace.calculators import mace_mp
# This call downloads and caches the checkpoint to potentials/ directory
import os, shutil
calc = mace_mp(model="medium", default_dtype="float64", device="cpu")
src = calc.model_path if hasattr(calc, 'model_path') else None
if src:
    dst = os.environ.get('MACE_MODEL', 'potentials/mace_mp_medium.model')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)
    print(f"MACE-MP model at: {dst}")
else:
    print("MACE model cached in ~/.cache/mace")
PYEOF
    success "MACE-MP-0 model downloaded."
else
    info "MACE model already exists at ${MACE_MODEL}"
fi

# ---- Placeholder ADP potential (replace with real file in production) -------
ADP_POT="${POTENTIALS_DIR}/Fe.adp.alloy"
if [ ! -f "${ADP_POT}" ]; then
    warn "ADP potential not found at ${ADP_POT}."
    warn "Download from NIST IPRP: https://www.ctcms.nist.gov/potentials/"
    warn "e.g. Bonny et al. Fe-Ni-Cr ADP, or Marinica et al. W ADP"
    warn "Set ADP_POTENTIAL env var to your .alloy file path."
    # Create a placeholder so the pipeline doesn't crash on import
    touch "${ADP_POT}.PLACEHOLDER"
fi

# ============================================================================
# 5. Directory structure
# ============================================================================
info "Creating data directory structure..."
mkdir -p \
    "${DATA_DIR}/raw_structures" \
    "${DATA_DIR}/defect_structures" \
    "${DATA_DIR}/evaluated/lammps_tmp" \
    "${DATA_DIR}/graphs" \
    "${DATA_DIR}/shards" \
    "${PIPELINE_DIR}/logs"
success "Directories created."

# ============================================================================
# 6. Validate MP API key
# ============================================================================
if [ -z "${MP_API_KEY:-}" ]; then
    warn "MP_API_KEY environment variable not set."
    warn "Stage 1 (data mining) will fail without a valid key."
    warn "Get one at: https://materialsproject.org/api"
else
    success "MP_API_KEY is set."
fi

# ============================================================================
# 7. Write .env for python-dotenv
# ============================================================================
ENV_FILE="${PIPELINE_DIR}/.env"
cat > "${ENV_FILE}" <<EOF
MP_API_KEY=${MP_API_KEY:-YOUR_MP_API_KEY}
LAMMPS_CMD=${HOME}/.local/bin/lmp
ADP_POTENTIAL=${ADP_POT}
GRACE_MODEL=${MACE_MODEL}
STORAGE_BACKEND=lmdb
EOF
success ".env written to ${ENV_FILE}"

# ============================================================================
# 8. Quick smoke-test
# ============================================================================
info "Running import smoke-test..."
python - <<'PYEOF'
import sys
failed = []
for pkg in ["ase", "pymatgen", "mp_api", "torch", "torch_geometric",
            "lmdb", "h5py", "ray", "numpy", "scipy"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError as e:
        failed.append(f"{pkg}: {e}")
if failed:
    print("FAILED imports:", *failed, sep="\n  ")
    sys.exit(1)
else:
    print("All core imports succeeded.")
PYEOF
success "Smoke-test passed."

# ============================================================================
# 9. Optionally run the pipeline
# ============================================================================
if [ "$RUN_PIPELINE" = true ]; then
    info "Launching radiation damage ML pipeline..."
    cd "${PIPELINE_DIR}"

    # Capture start time
    PIPELINE_START=$(date +%s)

    python run_pipeline.py 2>&1 | tee logs/pipeline_$(date +%Y%m%d_%H%M%S).log

    PIPELINE_END=$(date +%s)
    ELAPSED=$(( PIPELINE_END - PIPELINE_START ))
    success "Pipeline finished in ${ELAPSED}s ($(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s)."
else
    info "Setup complete. To run the pipeline:"
    echo ""
    echo "  source ${VENV_DIR}/bin/activate"
    echo "  cd ${PIPELINE_DIR}"
    echo "  export MP_API_KEY=your_key_here"
    echo "  python run_pipeline.py"
    echo ""
    echo "  # Or start from a specific stage:"
    echo "  python run_pipeline.py --start-stage 3"
    echo ""
    echo "  # For Ray cluster execution, set RAY_ADDRESS first:"
    echo "  export RAY_ADDRESS=ray://<head-node-ip>:10001"
    echo "  python run_pipeline.py"
fi

success "setup_env.sh complete."
