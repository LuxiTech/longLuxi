#!/usr/bin/env bash
# After Stage A 128K training completes:
#  1. Locate the final checkpoint
#  2. Stop any GPU procs
#  3. Launch vLLM at MAX_LEN=2097152 against the ckpt (factor=8 = 2M)
#  4. Run NIAH + RULER at 1M, 1.5M, 2M
#  5. Run LongBench-v2 long subset
#  6. Stop vLLM
# Idempotent; safe to re-run.
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT_DIR="checkpoints/stage_a_v2_128k"
# pick latest checkpoint-N
LATEST=$(ls -d $CKPT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -1)
# fall back to parent dir if save_only_model put files at top
if [[ -z "$LATEST" || ! -f "$LATEST/model.safetensors" ]] && [[ -f "$CKPT_DIR/model.safetensors" ]]; then
    LATEST="$CKPT_DIR"
fi

if [[ -z "$LATEST" ]]; then
    echo "[post_eval] no checkpoint found under $CKPT_DIR"; exit 1
fi
echo "[post_eval] using ckpt: $LATEST"

# 1. Kill any GPU procs from training
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1 | grep -v '^$' | xargs)
if [[ -n "$PIDS" ]]; then
    echo "[post_eval] killing GPU pids: $PIDS"
    kill -9 $PIDS 2>/dev/null || true
fi
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 0,1,2,3

# 2. Patch ckpt config if YaRN not present (LF may save without it; just copy YaRN section from base)
BASE_CFG="/home/user01/Minko/models/Qwen3.5-4B/config.json"
CKPT_CFG="$LATEST/config.json"
.venv-lf/bin/python - <<PY
import json
base = json.load(open("$BASE_CFG"))
dst  = json.load(open("$CKPT_CFG"))
# Use the SAME YaRN factor=8 config as base (factor=8 = 2M serving)
b_tc = base.get("text_config", base)
d_tc = dst.get("text_config", dst)
b_rope = b_tc.get("rope_parameters") or b_tc.get("rope_scaling")
d_rope = d_tc.get("rope_parameters") or d_tc.get("rope_scaling")
if b_rope and (not d_rope or d_rope.get("factor") != b_rope.get("factor")):
    d_tc["rope_parameters"] = b_rope
    d_tc["max_position_embeddings"] = b_tc["max_position_embeddings"]
    json.dump(dst, open("$CKPT_CFG", "w"), indent=2)
    print("[post_eval] patched YaRN into ckpt config: factor=", b_rope["factor"])
else:
    print("[post_eval] ckpt config already has YaRN factor=", (d_rope or {}).get("factor"))
PY

# 3. Launch vLLM at 2M max_len
echo "[post_eval] launching vLLM (MAX_LEN=2097152, model=$LATEST)"
MODEL="$LATEST" PORT=8001 MAX_LEN=2097152 GPU_MEM_UTIL=0.95 \
    bash eval/runners/vllm_server.sh > /tmp/vllm_post_a.log 2>&1 &
VLLM_PID=$!

# 4. Wait for ready
for i in {1..120}; do
    if curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "[post_eval] vllm ready after ${i}*15 = $((i*15))s"
        break
    fi
    if ! pgrep -fa vllm | grep -q "serve"; then
        echo "[post_eval] vllm died"
        tail -30 /tmp/vllm_post_a.log; exit 1
    fi
    sleep 15
done

# 5. Benchmarks (re-using existing batch script, write to stage_a_eval_* dirs)
OUTPREFIX="eval/results/stage_a_v2_128k"
PYBIN=".venv/bin/python"

run() {
    local bench="$1" ctx="$2" suffix="$3"
    local out="${OUTPREFIX}_${suffix}"
    echo "[post_eval] $bench L=$ctx -> $out"
    if [[ "$bench" == "ruler" ]]; then
        "$PYBIN" eval/runners/run_via_vllm.py --bench ruler --base-url http://localhost:8001 \
            --lengths "$ctx" --n-per-task 10 --out-dir "$out" 2>&1 | tail -5
    elif [[ "$bench" == "niah" ]]; then
        "$PYBIN" eval/runners/run_via_vllm.py --bench niah --base-url http://localhost:8001 \
            --lengths "$ctx" --n-per-cell 2 --depths 10 30 50 70 90 --out-dir "$out" 2>&1 | tail -5
    elif [[ "$bench" == "lbv2" ]]; then
        "$PYBIN" eval/runners/run_longbench_v2.py --base-url http://localhost:8001 \
            --length-bucket long --max-context-chars 2800000 --out-dir "$out" 2>&1 | tail -5
    fi
    if [[ -f "$out/metrics.json" ]]; then
        local acc=$("$PYBIN" -c "import json; m=json.load(open('$out/metrics.json')); print(f\"{m['accuracy']:.3f}\")" 2>/dev/null || echo "?")
        echo "[post_eval] DONE $bench L=$ctx accuracy=$acc"
    fi
}

# Order: cheap → expensive, so we see signal fast.
run niah 1000000 niah_1m
run niah 1500000 niah_1500k
run niah 2000000 niah_2m
run ruler 1000000 ruler_1m
run ruler 1500000 ruler_1500k
run ruler 2000000 ruler_2m
run lbv2 long longbench_v2_long

# 6. Cleanup
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1 | grep -v '^$' | xargs)
[[ -n "$PIDS" ]] && kill -9 $PIDS 2>/dev/null

echo "[post_eval] full suite complete @ $(date)"
echo "[post_eval] results in eval/results/stage_a_v2_128k_*"
