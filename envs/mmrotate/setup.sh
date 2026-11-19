#!/usr/bin/env bash
# Creates .venv-mmrotate (in the repository root) for the 'mmrotate' wrapper (see pyproject.toml) and the pinned ai4rs checkout.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
AI4RS_COMMIT=9aedabc228d86c15b73429e932251192c34c9742      # ai4rs main, 2026-10-06
if [ ! -d "$here/ai4rs/.git" ]; then
    git clone https://github.com/wokaikaixinxin/ai4rs.git "$here/ai4rs"
fi
git -C "$here/ai4rs" fetch -q origin "$AI4RS_COMMIT" || true
git -C "$here/ai4rs" -c advice.detachedHead=false checkout -q "$AI4RS_COMMIT"
UV_PROJECT_ENVIRONMENT="$root/.venv-mmrotate" uv sync --project "$here"
echo "ai4rs $(git -C "$here/ai4rs" rev-parse --short HEAD); venv $root/.venv-mmrotate"
