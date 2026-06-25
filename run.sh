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
# Use mamba (bundled in Miniforge) — far faster solver than classic conda; the
# combined native solve otherwise grinds for many minutes. Split into a few
# smaller transactions instead of one giant cross-channel solve.
if ! conda env list | grep -q "/$ENV$"; then
    LOG "1a. Create env (python 3.10)"
    mamba create -y -n "$ENV" python=3.10 2>&1 | tail -2
    conda activate "$ENV"

    LOG "1b. torch (CUDA) via conda; pytorch3d + torch-scatter via pip wheels"
    # The segmentation has hardcoded CUDA paths (pytorch3d rasterization) so we
    # need a real CUDA torch. Install torch from pytorch/nvidia channels (this
    # transaction is fast/fine), but get pytorch3d + torch-scatter from PIP
    # WHEELS — the cross-channel conda solve for pytorch3d on CUDA does NOT
    # converge (it hung for ~3h). PyG hosts cu118 wheels; pytorch3d hosts a
    # prebuilt wheel index keyed by torch+cuda+python.
    mamba install -y -c pytorch -c nvidia -c conda-forge \
        "pytorch=2.0.1" "torchvision=0.15.2" "pytorch-cuda=11.8" \
        2>&1 | tail -4
    PIPI="$CONDA_DIR/envs/$ENV/bin/pip"
    $PIPI install --quiet torch-scatter \
        -f https://data.pyg.org/whl/torch-2.0.1+cu118.html 2>&1 | tail -2 || true
    # pytorch3d prebuilt wheel for torch2.0.1 / cu118 / py310
    $PIPI install --quiet --no-index pytorch3d \
        -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html 2>&1 | tail -2 || \
      $PIPI install --quiet "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4" 2>&1 | tail -3 || true

    LOG "1c. FreeCAD + pythonOCC (conda-forge)"
    mamba install -y -c conda-forge freecad=0.21 pythonocc-core=7.7.2 eigen \
        2>&1 | tail -4

    LOG "1d. scientific / mesh libs (conda-forge)"
    mamba install -y -c conda-forge \
        "numpy=1.24" scipy networkx trimesh shapely rtree pandas \
        scikit-image scikit-learn matplotlib pillow opencv \
        2>&1 | tail -4
else
    conda activate "$ENV"
fi
PY="$CONDA_DIR/envs/$ENV/bin/python"
LOG "python: $($PY --version 2>&1)  at $PY"

# --------------------------------------------------------------------------
# 2. pip deps — install each INDIVIDUALLY so one failure can't abort the rest.
#    These are the pure-python / small libs not on the conda channels above.
# --------------------------------------------------------------------------
LOG "2. pip dependencies (individual, best-effort)"
$PY -m pip install --quiet --upgrade pip
for pkg in opencv-python-headless dill tqdm einops omegaconf pyhocon \
           icecream loguru potpourri3d pymeshlab python-louvain open3d \
           gitpython rich pyyaml requests setuptools transforms3d \
           pyquaternion coloredlogs pypng \
           meshio geomdl plyfile openmesh numba sympy svgwrite \
           rtree lapsolver vedo pyvista polyscope optimparallel; do
    $PY -m pip install --quiet "$pkg" 2>&1 | tail -1 || echo "  (pip $pkg failed, continuing)"
done
# bpy / blenderproc are imported by stage2; best-effort headless blender python
$PY -m pip install --quiet "bpy==3.6.0" --extra-index-url https://download.blender.org/pypi/ 2>&1 | tail -1 || echo "  (bpy failed)"
$PY -m pip install --quiet blenderproc 2>&1 | tail -1 || echo "  (blenderproc failed)"

# Patch blenderproc's __init__.py to remove its "only runnable via blenderproc
# run" guard, exactly as the repo's setup.sh does — the stages import it under
# plain `python`.
LOG "2b. Patch blenderproc guard"
# Locate __init__.py on disk (NOT via import — importing triggers the very
# guard we're removing, so an import-based lookup fails before patching).
SITEPKG=$($PY -c "import site;print(site.getsitepackages()[0])")
BP_INIT="$SITEPKG/blenderproc/__init__.py"
if [ -f "$BP_INIT" ]; then
# The CADDreamer stages only use bproc.camera and bproc.math. Import just
# those (plus utility/types they depend on) to avoid pulling blenderproc's
# entire writer/loader/renderer chain and its long optional-dependency tail
# (pypng, etc.). Guard removed so it imports under plain `python`.
cat > "$BP_INIT" <<'BPEOF'
"""A procedural Blender pipeline for photorealistic rendering."""
import os
import sys
from .version import __version__
if sys.version_info.major < 3:
    raise Exception("BlenderProc requires at least python 3.X to run.")
from .api import utility
from .api import math
from .api import camera
from .api import types
if "INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT" in os.environ:
    sys.path.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    if "PYTHONPATH" in os.environ:
        del os.environ["PYTHONPATH"]
    from .python.utility.SetupUtility import SetupUtility
    SetupUtility.setup([])
BPEOF
    echo "  patched $BP_INIT"
else
    echo "  blenderproc not importable; skipping patch"
fi

# FreeCAD python path so `import FreeCAD/Part/Mesh` works under our python.
export PYTHONPATH="$CONDA_DIR/envs/$ENV/lib:$CONDA_DIR/envs/$ENV/Mod:$PYTHONPATH"

# PyMesh: try the real package; if it won't build, install a trimesh-backed
# shim so the import resolves and the OCC STEP path (which doesn't depend on
# PyMesh's CGAL self-intersection ops) can run. The shim covers the few calls
# in neus/fit_surfaces/io_utils.py.
LOG "2c. PyMesh (real, else trimesh-backed shim)"
if ! $PY -c "import pymesh" 2>/dev/null; then
    $PY -m pip install --quiet pymesh2 2>&1 | tail -2 || true
fi
if ! $PY -c "import pymesh" 2>/dev/null; then
    echo "  real PyMesh unavailable -> installing trimesh-backed shim"
    SHIM_DIR="$SITEPKG/pymesh"
    mkdir -p "$SHIM_DIR"
    cp "$(pwd)/pymesh_shim.py" "$SHIM_DIR/__init__.py"
    $PY -c "import pymesh; print('  pymesh shim OK')"
else
    echo "  real PyMesh present"
fi

# Sanity: confirm the hard imports resolve before running the stages.
LOG "2a. Import sanity check"
$PY - <<'PYCHK'
mods = ["torch","torchvision","cv2","numpy","scipy","trimesh","pytorch3d",
        "torch_scatter","pymeshlab","potpourri3d","networkx","FreeCAD","Part","OCC",
        "community","pandas","pymesh"]
ok, bad = [], []
for m in mods:
    try:
        __import__(m); ok.append(m)
    except Exception as e:
        bad.append(f"{m}: {type(e).__name__}: {e}")
print("OK:", ok)
print("MISSING:", *bad, sep="\n  ")
# Surface the REAL error behind a failing `from utils.graphcut import ...`
# (Python masks a sub-import failure as "No module named 'utils.graphcut'").
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.getcwd(), "neus"))
# Reproduce the REAL condition: bpy/blenderproc imported first (they mutate
# sys.path), THEN utils.graphcut — exactly stage 2's order.
try:
    import bpy            # noqa
    import blenderproc    # noqa
except Exception as e:
    print("bpy/blenderproc import note:", e)
print("sys.path[0:6] after bpy:", sys.path[:6])
print("cwd on path?", os.getcwd() in sys.path)
try:
    import importlib
    importlib.import_module("utils.util")
    print("utils.util: OK")
    importlib.import_module("utils.graphcut")
    print("utils.graphcut: import OK")
except Exception:
    import traceback
    print("utils.graphcut REAL error:")
    traceback.print_exc()
PYCHK

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

# Path setup. The repo root must come BEFORE neus/ so that top-level `import
# utils.graphcut` resolves to repo-root utils/ (there is also a neus/utils/ —
# if neus/ were first, `utils` would bind to neus/utils, which lacks graphcut).
# neus/ is also on the path because neus/newton/process.py hardcodes
# `sys.path.append("/mnt/disk/CADDreamer/neus")` so it can `import newton.*`.
export PYTHONPATH="$(pwd):$(pwd)/neus:$PYTHONPATH"

# The stages write intermediate .obj/cache files into these dirs but don't
# create them. Make them upfront.
mkdir -p neus/temp_mid_outputs neus/temp_mid_results

# --------------------------------------------------------------------------
# 4. Stage 2 — segmentation (writes the temp cache stage 3 consumes)
# --------------------------------------------------------------------------
LOG "4. Stage 2: segmentation"
set +e
$PY test_syne_images_stage_2_segmentation.py --config_dir "./$EXAMPLE" --review True 2>&1 | tee "$ART/stage2.log" | tail -40
echo "stage2 exit: ${PIPESTATUS[0]}"

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
