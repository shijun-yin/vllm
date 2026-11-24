#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/qwen35-9b-deploy
VALIDATION=$ROOT/h20_validation
VLLM_SRC=$ROOT/vllm-h20
PY=$ROOT/.venv-h20/bin/python
export PATH=/root/.local/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

mkdir -p "$ROOT/results"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=/root/.local/bin:$PATH

if [[ ! -d "$VLLM_SRC/.git" ]]; then
  git init "$VLLM_SRC"
  git -C "$VLLM_SRC" remote add origin https://github.com/vllm-project/vllm.git
  git -C "$VLLM_SRC" fetch --depth=1 origin bf151b7eb1403d3671200e0cd19ad9821fe4320f
fi
git -C "$VLLM_SRC" checkout bf151b7eb1
if git -C "$VLLM_SRC" apply --check "$VALIDATION/suffix_splice.patch"; then
  git -C "$VLLM_SRC" apply "$VALIDATION/suffix_splice.patch"
else
  git -C "$VLLM_SRC" apply --reverse --check "$VALIDATION/suffix_splice.patch"
fi

if [[ ! -x "$PY" ]]; then
  uv venv --python 3.12 "$ROOT/.venv-h20"
fi
cd "$VLLM_SRC"
VLLM_USE_PRECOMPILED=1 uv pip install \
  --python "$PY" -e . --torch-backend=auto
uv pip install --python "$PY" -r requirements/test/cuda.in

"$PY" "$VALIDATION/prepare_data.py" --endpoint https://hf-mirror.com
"$PY" "$VALIDATION/prepare_data.py" --check-only

"$PY" - <<'PY'
from huggingface_hub import snapshot_download

for repo, revision in [
    ("Qwen/Qwen3.8-27B-FP8", "c3a462cc3d38cf351f3b26cbd763469eeb74ef81"),
    ("incoai/Qwen3.8-27B-DFlash2", "015e795645c74b1a0eeef3b570031fb62e769bc5"),
]:
    path = snapshot_download(repo, revision=revision)
    print(f"MODEL_READY {repo} {path}", flush=True)
PY

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
"$PY" -c "import torch, vllm; print('TORCH', torch.__version__, 'CUDA', torch.cuda.is_available()); print('VLLM', vllm.__version__)"
printf 'SETUP_OK\n'
