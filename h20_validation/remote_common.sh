#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/qwen35-9b-deploy
VALIDATION=$ROOT/h20_validation
PY=$ROOT/.venv-h20/bin/python
TARGET=Qwen/Qwen3.8-27B-FP8
DRAFTER=incoai/Qwen3.8-27B-DFlash2
TARGET_REV=c3a462cc3d38cf351f3b26cbd763469eeb74ef81
DRAFTER_REV=015e795645c74b1a0eeef3b570031fb62e769bc5

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

COMMON=(
  --family dflash
  --target-model "$TARGET" --target-revision "$TARGET_REV"
  --draft-model "$DRAFTER" --draft-revision "$DRAFTER_REV"
  --num-speculative-tokens 7 --prefix-len 1
  --temperature 1.0 --top-p 0.95 --top-k 20
  --gpu-memory-utilization 0.90
)
