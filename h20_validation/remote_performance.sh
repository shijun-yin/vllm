#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/remote_common.sh"
DATASETS=blazedit,speed_coding,speed_rag

"$PY" "$VALIDATION/benchmark_splice.py" "${COMMON[@]}" \
  --mode no_spec --datasets "$DATASETS" \
  --num-prompts 16 --max-tokens 384 --hot-rounds 2 --batch-size 4 \
  --output "$ROOT/results/h20_no_spec_b4.json" \
  > "$ROOT/results/h20_no_spec_b4.log" 2>&1

for run in 1 2; do
  for mode in pure splice; do
    "$PY" "$VALIDATION/benchmark_splice.py" "${COMMON[@]}" \
      --mode "$mode" --datasets "$DATASETS" \
      --num-prompts 16 --max-tokens 384 --hot-rounds 2 --batch-size 4 \
      --output "$ROOT/results/h20_${mode}_b4_r${run}.json" \
      > "$ROOT/results/h20_${mode}_b4_r${run}.log" 2>&1
  done
  "$PY" "$VALIDATION/summarize_results.py" \
    "$ROOT/results/h20_pure_b4_r${run}.json" \
    "$ROOT/results/h20_splice_b4_r${run}.json" \
    | tee "$ROOT/results/h20_summary_b4_r${run}.md"
done

for batch in 1 16 32; do
  for mode in pure splice; do
    "$PY" "$VALIDATION/benchmark_splice.py" "${COMMON[@]}" \
      --mode "$mode" --datasets "$DATASETS" \
      --num-prompts 16 --max-tokens 384 --hot-rounds 2 --batch-size "$batch" \
      --output "$ROOT/results/h20_${mode}_b${batch}.json" \
      > "$ROOT/results/h20_${mode}_b${batch}.log" 2>&1
  done
  "$PY" "$VALIDATION/summarize_results.py" \
    "$ROOT/results/h20_pure_b${batch}.json" \
    "$ROOT/results/h20_splice_b${batch}.json" \
    | tee "$ROOT/results/h20_summary_b${batch}.md"
done

for corpus in 0 131072; do
  for mode in pure splice; do
    "$PY" "$VALIDATION/benchmark_splice.py" "${COMMON[@]}" \
      --mode "$mode" --datasets blazedit,speed_rag \
      --corpus-tokens "$corpus" --num-prompts 16 --max-tokens 384 \
      --hot-rounds 2 --batch-size 4 \
      --output "$ROOT/results/h20_${mode}_corpus${corpus}.json" \
      > "$ROOT/results/h20_${mode}_corpus${corpus}.log" 2>&1
  done
done

printf 'PERFORMANCE_SUITE_OK\n'
