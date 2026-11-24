#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/qwen35-9b-deploy}
PY=${PY:-$ROOT/.venv-local/bin/python}
cd "$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for mode in no_spec splice; do
  "$PY" validate_semantics.py "$mode" --structured \
    > "local_${mode}_structured.log" 2>&1
done

for mode in no_spec no_spec_alt splice; do
  "$PY" validate_semantics.py "$mode" \
    > "local_${mode}_distribution.log" 2>&1
done

"$PY" validate_semantics.py compare > local_distribution_summary.log 2>&1
"$PY" validate_preemption.py > local_splice_preemption.log 2>&1
