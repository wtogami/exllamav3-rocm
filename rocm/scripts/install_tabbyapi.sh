#!/bin/bash
# Install TabbyAPI next to this repository, patched to accept RDNA3 GPUs, using the active environment.
# Re-runnable: the clone, the patch and the model links are all left as they are if already in place.
# usage: rocm/scripts/install_tabbyapi.sh [tabby_dir] [models_dir]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
TABBY=${1:-$REPO/../tabbyAPI}
MODELS_SRC=${2:-$REPO/models}
if [ ! -d "$MODELS_SRC" ]; then
    echo "no model directory at $MODELS_SRC - run rocm/scripts/download_models.sh first" >&2
    exit 1
fi
MODELS=$(cd "$MODELS_SRC" && pwd)
TABBY_COMMIT=f07131cd8fe34e449fe87cdd3a066b52b96d3cac   # tested revision
PATCH="$REPO/rocm/tabbyapi/0001-exllamav3-allow-rdna3.patch"
[ -d "$TABBY" ] || git clone https://github.com/theroyallab/tabbyAPI "$TABBY"
cd "$TABBY"
git checkout -q "$TABBY_COMMIT"
# reverse --check succeeds only when the patch is already in the tree
git apply --reverse --check "$PATCH" >/dev/null 2>&1 || git apply "$PATCH"
# TabbyAPI's own dependencies only: torch and exllamav3 come from this repository's environment
pip install "fastapi-slim>=0.115" "pydantic>=2.11,<3" ruamel.yaml rich "uvicorn>=0.28.1" "jinja2>=3.0.0" loguru \
            "sse-starlette>=2.2.0" packaging aiofiles aiohttp async_lru psutil "httptools>=0.5.0" requests uvloop setuptools
# TabbyAPI ships its own models/ directory (it tracks models/place_your_models_here.txt), so
# `ln -s "$MODELS" models` lands the link *inside* it as models/models and every model_name then
# resolves to models/<name>, which does not exist. Link the model directories individually into
# TabbyAPI's models/ instead, and undo the nested link if an older run of this script left one.
if [ -L models ]; then rm models; fi
if [ -L models/models ]; then rm models/models; fi   # cruft left by older versions of this script
mkdir -p models
linked=0
for model in "$MODELS"/*/; do
    name=$(basename "$model")
    case "$name" in .*) continue ;; esac
    ln -sfn "${model%/}" "models/$name"
    linked=$((linked+1))
done
if [ "$linked" = 0 ]; then
    echo "warning: $MODELS holds no model directories yet (rocm/scripts/download_models.sh)" >&2
fi
cp -n "$REPO/rocm/tabbyapi/config.dflash2-192k.yml" config.yml
echo "TabbyAPI ready in $TABBY (config.yml = DFlash2 / 192K profile, $linked model dirs linked from $MODELS)"
