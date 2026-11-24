#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/qwen35-9b-deploy}
PY=${PY:-$ROOT/.venv-local/bin/python}
TARGET=${TARGET:-$ROOT/models/Qwen3.5-4B-AWQ-4bit}
DRAFTER=${DRAFTER:-$ROOT/models/Qwen3.5-4B-DFlash2}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATASETS=${DATASETS:-blazedit,speed_coding,speed_rag,speed_writing,spec_rag,spec_summarization}
RUNS=${RUNS:-1}

mkdir -p "$ROOT/results"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for ((run = 1; run <= RUNS; run++)); do
  "$PY" "$SCRIPT_DIR/benchmark_splice.py" \
    --family dflash --mode pure \
    --target-model "$TARGET" --draft-model "$DRAFTER" \
    --num-speculative-tokens 7 \
    --temperature 1.0 --top-p 0.95 --top-k 20 \
    --corpus-tokens 65536 --datasets "$DATASETS" \
    --num-prompts 16 --max-tokens 384 --hot-rounds 2 --batch-size 4 \
    --gpu-memory-utilization 0.98 \
    --output "$ROOT/results/local_dflash2_k7_pure_r${run}.json" \
    > "$ROOT/results/local_dflash2_k7_pure_r${run}.log" 2>&1

  for j in 1 2 4; do
    "$PY" "$SCRIPT_DIR/benchmark_splice.py" \
      --family dflash --mode splice \
      --target-model "$TARGET" --draft-model "$DRAFTER" \
      --num-speculative-tokens 7 --prefix-len "$j" \
      --temperature 1.0 --top-p 0.95 --top-k 20 \
      --corpus-tokens 65536 --datasets "$DATASETS" \
      --num-prompts 16 --max-tokens 384 --hot-rounds 2 --batch-size 4 \
      --gpu-memory-utilization 0.98 \
      --output "$ROOT/results/local_dflash2_k7_j${j}_r${run}.json" \
      > "$ROOT/results/local_dflash2_k7_j${j}_r${run}.log" 2>&1
  done
done
