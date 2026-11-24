#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/remote_common.sh"

for mode in no_spec pure splice; do
  "$PY" "$VALIDATION/benchmark_splice.py" "${COMMON[@]}" \
    --mode "$mode" --datasets blazedit,speed_coding \
    --num-prompts 4 --max-tokens 128 --hot-rounds 1 --batch-size 4 \
    --output "$ROOT/results/h20_smoke_${mode}.json" \
    > "$ROOT/results/h20_smoke_${mode}.log" 2>&1
done

"$PY" "$VALIDATION/summarize_results.py" \
  "$ROOT/results/h20_smoke_pure.json" \
  "$ROOT/results/h20_smoke_splice.json" \
  | tee "$ROOT/results/h20_smoke_summary.md"
printf 'SMOKE_OK\n'
