#!/bin/bash
# Full CADDreamer reproduction — Module 2 end to end on the authors' cached
# stage-1 output (single example 0_0deepcad).
#
#   cached normal/semantic maps + reconstructed mesh  (authors' stage-1 output)
#        -> Stage 2: graph-cut segmentation into primitive patches
#        -> Stage 3: RANSAC primitive fit + geometric optimization
#                    + OpenCascade primitive intersection -> watertight B-rep STEP
#
# We then compare the generated STEP against the authors' bundled result.step.
# Stage 1 (the diffusion model) is skipped because its checkpoints are offline
# (UT Dallas Box links 404); we start from the cached maps the authors shipped.
set -e
set -o pipefail

EXAMPLE="cached_output/cropsize-256-cfg1.0-syne/0_0deepcad"
ART=".openresearch/artifacts"
mkdir -p "$ART"
LOG() { echo "=== $* ==="; }

LOG "CADDreamer full pipeline (stage2->stage3) on $EXAMPLE"

# --------------------------------------------------------------------------
# 1. Conda (Miniforge) — binary deps that are painful to build: FreeCAD,
#    pythonOCC, pytorch3d, pymeshlab. Far more reliable than the README's
#    from-source OCC build.
# --------------------------------------------------------------------------
LOG "1. Bootstrap conda environment"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq wget git build-essential cmake libeigen3-dev pkg-config \
    libgl1 libglu1-mesa libxrender1 libxi6 libxkbcommon0 libsm6 >/dev/null 2>&1 || true

CONDA_DIR="$HOME/miniforge3"
if [ ! -d "$CONDA_DIR" ]; then
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
    bash /tmp/mf.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

ENV=cad
if ! conda env list | grep -q "/$ENV$"; then
    LOG "1a. Create env (python 3.10) + conda-forge binaries"
    conda create -y -n "$ENV" python=3.10 >/dev/null
    conda activate "$ENV"
    # FreeCAD brings OpenCascade + Part/Mesh bindings; pythonocc-core gives OCC.*;
    # pymeshlab/potpourri3d/eigen via conda-forge as prebuilt wheels/pkgs.
    conda install -y -c conda-forge \
        freecad=0.21 pythonocc-core=7.7.2 eigen \
        numpy=1.24 scipy networkx trimesh shapely rtree pyembree 2>&1 | tail -3
else
    conda activate "$ENV"
fi
PY="$CONDA_DIR/envs/$ENV/bin/python"
LOG "python: $($PY --version 2>&1)  at $PY"

# --------------------------------------------------------------------------
# 2. pip deps (torch CPU is enough for stage2/3 tensor ops; pytorch3d + the
#    rendering/segmentation stack).
# --------------------------------------------------------------------------
LOG "2. pip dependencies"
$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet \
    "torch==2.0.1" "torchvision==0.15.2" --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2 || \
    $PY -m pip install --quiet "torch" "torchvision" 2>&1 | tail -2
$PY -m pip install --quiet \
    opencv-python-headless pillow scikit-image scikit-learn matplotlib \
    dill tqdm einops omegaconf pyhocon icecream loguru \
    potpourri3d pymeshlab python-louvain trimesh open3d \
    torch-scatter 2>&1 | tail -3 || true
# pytorch3d (CPU) from source-light wheel; fall back to git if needed
$PY -m pip install --quiet "git+https://github.com/facebookresearch/pytorch3d.git@stable" 2>&1 | tail -3 || \
    $PY -m pip install --quiet pytorch3d 2>&1 | tail -2 || true
# blenderproc/bpy are imported by stage2; install bpy (headless blender python)
$PY -m pip install --quiet "bpy==3.6.0" --extra-index-url https://download.blender.org/pypi/ 2>&1 | tail -2 || true
$PY -m pip install --quiet blenderproc 2>&1 | tail -2 || true

# --------------------------------------------------------------------------
# 3. Build the fitpoints RANSAC extension against THIS python.
# --------------------------------------------------------------------------
LOG "3. Build fitpoints"
( cd pyransac
  rm -rf build-orx
  cmake -S . -B build-orx -DCMAKE_BUILD_TYPE=Release \
        -DPython3_EXECUTABLE="$PY" >cmake.log 2>&1
  cmake --build build-orx --target fitpoints -j "$(nproc)" >>cmake.log 2>&1
  # the code expects it under pyransac/cmake-build-release/
  mkdir -p cmake-build-release
  cp build-orx/fitpoints*.so cmake-build-release/ 2>/dev/null || true
) || { echo "fitpoints build failed:"; tail -25 pyransac/cmake.log; exit 1; }
ls -la pyransac/cmake-build-release/fitpoints*.so

# FreeCAD python path so `import FreeCAD/Part/Mesh` works under our python
export PYTHONPATH="$CONDA_DIR/envs/$ENV/lib:$CONDA_DIR/envs/$ENV/Mod:$PYTHONPATH"

# --------------------------------------------------------------------------
# 4. Stage 2 — segmentation (writes the temp cache stage 3 consumes)
# --------------------------------------------------------------------------
LOG "4. Stage 2: segmentation"
$PY test_syne_images_stage_2_segmentation.py --config_dir "./$EXAMPLE" --review True 2>&1 | tee "$ART/stage2.log" | tail -40

# --------------------------------------------------------------------------
# 5. Stage 3 — primitive fit + OCC intersection -> STEP
# --------------------------------------------------------------------------
LOG "5. Stage 3: primitive fitting + B-rep STEP generation"
$PY test_syne_images_stage_3_generate_step.py --config_dir "./$EXAMPLE" --review True 2>&1 | tee "$ART/stage3.log" | tail -40

# --------------------------------------------------------------------------
# 6. Collect outputs + compare to the authors' reference STEP
# --------------------------------------------------------------------------
LOG "6. Verify outputs"
$PY caddreamer_eval.py --example "./$EXAMPLE" 2>&1 | tee -a "$ART/eval.log"
cp -f EVAL.md "$ART/EVAL.md" 2>/dev/null || true
LOG "done"
