#!/usr/bin/env bash
# Builds detectron2 (wrapper 'detectron2') from source at a pinned commit and installs it into the main venv (.venv in
# the repository root). detectron2 has no wheels for current torch, so it is not a project extra.
#   envs/detectron2/setup.sh            build (once per torch version, wheel kept in envs/detectron2/dist/) + install
# Rerun after a plain `uv sync`, which removes packages that are not in the lock file (the cached wheel is reused, so
# that takes seconds); or sync with `uv sync --inexact ...`, which keeps them.
# The build compiles detectron2's C++ ops only (no CUDA compiler needed): the standard Faster / Mask R-CNN models use
# torchvision's CUDA ops. It needs the Python headers; if the venv's Python has none (e.g. a system Python without
# python3.x-dev), the headers of a uv-managed Python of the same version are used (same ABI).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
venv="${VENV:-$root/.venv}"
py="$venv/bin/python"
D2_COMMIT=247691a1c292654867deb35cabb27f18d75a1363      # detectron2 main, 2026-10-27 (tested with torch 2.10 + cu128)

[ -x "$py" ] || { echo "no venv at $venv: run 'uv sync' first" >&2; exit 1; }
torch_ver="$("$py" -c 'import torch; print(torch.__version__)')"
pyver="$("$py" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
dist="$here/dist/${D2_COMMIT:0:7}-torch${torch_ver}-py${pyver}"

# runtime dependencies of detectron2 (installed with --no-deps below: its hydra-core pin would downgrade omegaconf)
uv pip install -q --python "$py" fvcore iopath yacs termcolor cloudpickle tabulate

if ! ls "$dist"/detectron2-*.whl >/dev/null 2>&1; then
    if [ ! -d "$here/src/.git" ]; then
        git clone -q https://github.com/facebookresearch/detectron2 "$here/src"
    fi
    git -C "$here/src" fetch -q origin "$D2_COMMIT" || true
    git -C "$here/src" -c advice.detachedHead=false checkout -q -f "$D2_COMMIT"
    # pkg_resources was removed in setuptools >= 81: use importlib.resources for the model zoo configs
    "$py" - "$here/src/detectron2/model_zoo/model_zoo.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
s = s.replace("import pkg_resources\n", "from importlib import resources\n")
s = s.replace('cfg_file = pkg_resources.resource_filename(\n        "detectron2.model_zoo", os.path.join("configs", config_path)\n    )',
              'cfg_file = str(resources.files("detectron2.model_zoo").joinpath("configs", config_path))')
assert "pkg_resources" not in s, "model_zoo.py patch failed (detectron2 source changed?)"
open(p, "w").write(s)
EOF
    uv pip install -q --python "$py" setuptools wheel ninja
    inc="$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
    if [ ! -f "$inc/Python.h" ]; then
        uv python install -q "$pyver"
        mpy="$(uv python find --managed-python "$pyver")"
        inc="$("$mpy" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
        echo "Python headers from the uv-managed Python $pyver: $inc"
        export CPATH="$inc${CPATH:+:$CPATH}"
    fi
    mkdir -p "$dist"
    echo "building detectron2 ${D2_COMMIT:0:7} for torch $torch_ver (a few minutes) ..."
    (cd "$here/src" && VIRTUAL_ENV="$venv" uv build -q --wheel --no-build-isolation --python "$py" -o "$dist" .)
fi

uv pip install -q --python "$py" --no-deps --reinstall "$dist"/detectron2-*.whl
"$py" -c 'import detectron2; from detectron2 import model_zoo; model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"); print("detectron2", detectron2.__version__, "OK")'
