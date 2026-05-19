"""Quick leakage control: same InfiniteBench longbook_choice questions, NO context.

If accuracy >> 25%, the benchmark is testing memorized world knowledge,
not long-context reasoning over the provided book.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval.runners._vllm_client import VllmClient  # noqa: E402

from huggingface_hub import hf_hub_download

PROMPT = (
    "Answer the multiple-choice question. Output only a single letter "
    "(A, B, C, or D) — nothing else.\n\n"
    "Question: {question}\n\n"
    "(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n\n"
    "Answer:"
)

_LETTER_RE = re.compile(r"\b([ABCD])\b")


def main():
    base_url = "http://localhost:8001"
    out_dir = Path("eval/results/base_infbench_longbook_choice_NOCTX")
    out_dir.mkdir(parents=True, exist_ok=True)

    import httpx
    served = httpx.get(f"{base_url}/v1/models").json()["data"][0]["id"]
    client = VllmClient(base_url=base_url, model=served, timeout=120.0)

    path = hf_hub_download("xinrongzhang2022/InfiniteBench", "longbook_choice_eng.jsonl",
                           repo_type="dataset")
    records = [json.loads(l) for l in open(path)]

    n_ok = 0
    preds: list[dict] = []
    for i, r in enumerate(records):
        prompt = PROMPT.format(question=r["input"],
                               A=r["options"][0], B=r["options"][1],
                               C=r["options"][2], D=r["options"][3])
        try:
            resp = client.complete_chat(
                [{"role": "user", "content": prompt}],
                max_new_tokens=8, enable_thinking=False,
            )
        except Exception as exc:
            print(f"err idx={i}: {exc}", flush=True)
            continue
        m = _LETTER_RE.search((resp or "").upper())
        letter = m.group(1) if m else ""
        gold_raw = r["answer"][0] if isinstance(r["answer"], list) else r["answer"]
        gold_letter = ""
        for idx, opt in enumerate(r["options"][:4]):
            if str(opt).strip() == str(gold_raw).strip():
                gold_letter = "ABCD"[idx]; break
        ok = (letter == gold_letter) if gold_letter else False
        n_ok += int(ok)
        preds.append({"id": r.get("id", i), "answer": gold_raw, "gold_letter": gold_letter,
                      "pred_raw": resp[:60], "pred_letter": letter, "ok": ok})
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(records)} acc_so_far={n_ok/(i+1):.3f}", flush=True)

    metrics = {
        "model_id": served,
        "task": "longbook_choice_eng_NO_CONTEXT",
        "n_attempted": len(preds),
        "accuracy": n_ok / max(len(preds), 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    with (out_dir / "predictions.jsonl").open("w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
