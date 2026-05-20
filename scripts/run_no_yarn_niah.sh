#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8001}"
PYBIN=".venv/bin/python"
run() {
    local ctx="$1" suffix="$2"
    local out="eval/results/base_no_yarn_niah_${suffix}"
    local t0=$(date +%s)
    echo "[no-yarn-niah] L=$ctx -> $out @ $(date +%H:%M:%S)"
    "$PYBIN" eval/runners/run_via_vllm.py --bench niah --base-url "http://localhost:$PORT" \
        --lengths "$ctx" --n-per-cell 2 --depths 10 30 50 70 90 --out-dir "$out" 2>&1 | tail -3
    local t1=$(date +%s)
    if [[ -f "$out/metrics.json" ]]; then
        local acc=$("$PYBIN" -c "import json; print(json.load(open('$out/metrics.json'))['accuracy'])" 2>/dev/null)
        echo "[no-yarn-niah] DONE L=$ctx accuracy=$acc elapsed=$((t1-t0))s"
    else
        echo "[no-yarn-niah] FAIL L=$ctx elapsed=$((t1-t0))s"
    fi
}
run 1000000 1m
run 1500000 1500k
run 2000000 2m
echo "[no-yarn-niah] all done @ $(date)"
