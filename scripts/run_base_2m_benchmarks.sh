#!/usr/bin/env bash
# Run base Qwen3.5-4B (YaRN factor=8 → 2M) benchmarks against vllm at MAX_LEN=2097152.
# Emits [bench-done] per combo for progress monitoring.
set -uo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8001}"
PYBIN="${PYBIN:-.venv/bin/python}"

RULER_LENGTHS=(1000000 1500000 2000000)
NIAH_LENGTHS=(1000000 1500000 2000000)

run() {
    local bench="$1" ctx="$2" out="$3"
    local t0=$(date +%s)
    echo "[bench-start] $bench L=$ctx out=$out @ $(date +%H:%M:%S)"
    if [[ "$bench" == "ruler" ]]; then
        "$PYBIN" eval/runners/run_via_vllm.py --bench ruler --base-url "http://localhost:$PORT" \
            --lengths "$ctx" --n-per-task 10 --out-dir "$out" 2>&1 | tail -3
    else
        "$PYBIN" eval/runners/run_via_vllm.py --bench niah --base-url "http://localhost:$PORT" \
            --lengths "$ctx" --n-per-cell 2 --depths 10 30 50 70 90 --out-dir "$out" 2>&1 | tail -3
    fi
    local rc=$?
    local t1=$(date +%s)
    if [[ $rc -eq 0 && -f "$out/metrics.json" ]]; then
        local acc=$("$PYBIN" -c "import json; m=json.load(open('$out/metrics.json')); print(f\"{m['accuracy']:.3f}\")" 2>/dev/null || echo "?")
        echo "[bench-done]  $bench L=$ctx accuracy=$acc elapsed=$((t1-t0))s"
    else
        echo "[bench-fail]  $bench L=$ctx rc=$rc elapsed=$((t1-t0))s"
    fi
}

echo "[batch-start] base@2M benchmarks @ $(date)"

for L in "${NIAH_LENGTHS[@]}"; do
    run niah "$L" "eval/results/base_yarn8_niah_$L"
done

for L in "${RULER_LENGTHS[@]}"; do
    run ruler "$L" "eval/results/base_yarn8_ruler_$L"
done

echo "[batch-done] @ $(date)"
