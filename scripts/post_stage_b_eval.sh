#!/usr/bin/env bash
# Post Stage B eval on checkpoint-200 (training killed at step 249/250, ckpt-200 is latest intact).
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT="checkpoints/stage_b_v3_128k/checkpoint-200"
echo "[post_b_eval] ckpt=$CKPT @ $(date)"

# Patch ckpt config with YaRN factor=8 (model config from base path)
.venv-lf/bin/python - <<PY
import json
base = json.load(open("/home/user01/Minko/models/Qwen3.5-4B/config.json"))
dst = json.load(open("$CKPT/config.json"))
b_tc = base.get("text_config", base)
d_tc = dst.get("text_config", dst)
b_rope = b_tc.get("rope_parameters") or b_tc.get("rope_scaling")
d_rope = d_tc.get("rope_parameters") or d_tc.get("rope_scaling")
if b_rope and (not d_rope or d_rope.get("factor") != b_rope.get("factor")):
    d_tc["rope_parameters"] = b_rope
    d_tc["max_position_embeddings"] = b_tc["max_position_embeddings"]
    json.dump(dst, open("$CKPT/config.json", "w"), indent=2)
    print("[post_b_eval] patched YaRN: factor=", b_rope["factor"])
else:
    print("[post_b_eval] ckpt config already has YaRN factor=", (d_rope or {}).get("factor"))
PY

# Stop any GPU procs
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1 | grep -v '^$' | xargs)
[[ -n "$PIDS" ]] && kill -9 $PIDS 2>/dev/null
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 0,1,2,3

# Launch vLLM at 2M
echo "[post_b_eval] launching vLLM"
MODEL="$CKPT" PORT=8001 MAX_LEN=2097152 GPU_MEM_UTIL=0.95 \
    bash eval/runners/vllm_server.sh > /tmp/vllm_stage_b.log 2>&1 &

# Wait for ready
for i in {1..120}; do
    if curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "[post_b_eval] vllm ready after $((i*15))s"
        break
    fi
    sleep 15
done

OUTPREFIX="eval/results/stage_b_v3_128k"
PYBIN=".venv/bin/python"

run() {
    local bench="$1" ctx="$2" suffix="$3"
    local out="${OUTPREFIX}_${suffix}"
    echo "[post_b_eval] $bench L=$ctx -> $out"
    if [[ "$bench" == "ruler" ]]; then
        "$PYBIN" eval/runners/run_via_vllm.py --bench ruler --base-url http://localhost:8001 \
            --lengths "$ctx" --n-per-task 10 --out-dir "$out" 2>&1 | tail -3
    elif [[ "$bench" == "niah" ]]; then
        "$PYBIN" eval/runners/run_via_vllm.py --bench niah --base-url http://localhost:8001 \
            --lengths "$ctx" --n-per-cell 2 --depths 10 30 50 70 90 --out-dir "$out" 2>&1 | tail -3
    elif [[ "$bench" == "lbv2" ]]; then
        "$PYBIN" eval/runners/run_longbench_v2.py --base-url http://localhost:8001 \
            --length-bucket long --max-context-chars 2800000 --out-dir "$out" 2>&1 | tail -3
    fi
    if [[ -f "$out/metrics.json" ]]; then
        local acc=$("$PYBIN" -c "import json; m=json.load(open('$out/metrics.json')); print(f\"{m['accuracy']:.3f}\")" 2>/dev/null || echo "?")
        echo "[post_b_eval] DONE $bench L=$ctx accuracy=$acc"
    fi
}

run niah 1000000 niah_1m
run niah 1500000 niah_1500k
run niah 2000000 niah_2m
run ruler 1000000 ruler_1m
run ruler 1500000 ruler_1500k
run ruler 2000000 ruler_2m
run lbv2 long longbench_v2_long

PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1 | grep -v '^$' | xargs)
[[ -n "$PIDS" ]] && kill -9 $PIDS 2>/dev/null
echo "[post_b_eval] complete @ $(date)"
