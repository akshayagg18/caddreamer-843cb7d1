#!/bin/bash
# Minimal end-to-end reproduction of CADDreamer's core mechanism:
#   Module 2 -- primitive-aware geometric extraction via Efficient-RANSAC.
#
# This builds the paper's `fitpoints` C++/pybind11 RANSAC extension and runs
# primitive detection (plane / cylinder / cone / sphere / torus) on the repo's
# bundled point+normal sample (pyransac/test_data_for_pyransac.pth).
#
# CPU-only: no multi-view diffusion, no NeuS, no GPU, no gated checkpoints.
set -e

echo "=============================================="
echo " CADDreamer minimal repro (RANSAC core, CPU)"
echo "=============================================="

mkdir -p .openresearch/artifacts

# ---- 1. System build deps -------------------------------------------------
echo "[1/4] Installing build dependencies (cmake, g++, eigen, python-dev)..."
export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y -qq build-essential cmake libeigen3-dev pkg-config python3-dev >/dev/null 2>&1 || true
fi

# Python deps for the driver
python3 -m pip install --quiet numpy dill 2>/dev/null || pip install --quiet numpy dill 2>/dev/null || true

# ---- 2. Build the fitpoints extension ------------------------------------
echo "[2/4] Building fitpoints (pybind11 RANSAC primitive fitter)..."
cd pyransac
PYTAG=$(python3 -c "import sys;print(f'{sys.version_info.major}{sys.version_info.minor}')")
BUILT_SO=""
if cmake -S . -B build-orx -DCMAKE_BUILD_TYPE=Release >build_cmake.log 2>&1 \
   && cmake --build build-orx --target fitpoints -j "$(nproc)" >>build_cmake.log 2>&1; then
    BUILT_SO=$(ls build-orx/fitpoints*.so 2>/dev/null | head -1 || true)
    echo "    built: $BUILT_SO"
else
    echo "    fresh build failed; will try prebuilt .so (see build_cmake.log tail):"
    tail -20 build_cmake.log || true
fi

# Locate an importable fitpoints module: freshly built, else prebuilt 3.10 .so
if [ -n "$BUILT_SO" ]; then
    export FITPOINTS_DIR="$(pwd)/build-orx"
elif [ "$PYTAG" = "310" ] && ls cmake-build-release/fitpoints*.so >/dev/null 2>&1; then
    echo "    using prebuilt cmake-build-release/*.so (py310 match)"
    export FITPOINTS_DIR="$(pwd)/cmake-build-release"
else
    PRE=$(ls cmake-build-release/fitpoints*.so 2>/dev/null | head -1 || true)
    echo "    no fresh build and prebuilt is py310 (box is py${PYTAG}); attempting prebuilt anyway"
    export FITPOINTS_DIR="$(pwd)/cmake-build-release"
fi
cd ..

# ---- 3. Run primitive fitting --------------------------------------------
echo "[3/4] Running RANSAC primitive detection on bundled sample..."
python3 caddreamer_minimal_repro.py

# ---- 4. Done --------------------------------------------------------------
echo "[4/4] Done. Results in EVAL.md and .openresearch/artifacts/"
cp -f EVAL.md .openresearch/artifacts/EVAL.md 2>/dev/null || true
