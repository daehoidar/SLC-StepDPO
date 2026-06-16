"""6개 페르소나에 같은 문제를 던져서 풀이 스타일이 다르게 나오는지 비교."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from utils import load_personas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3-1.7B")
    p.add_argument("--adapter", default=str(REPO_ROOT / "checkpoints" / "sft_qwen3_1.7b"))
    p.add_argument("--problem", default=None,
                   help="평가할 문제. 미지정 시 SFT 데이터에서 랜덤 한 개.")
    p.add_argument("--data", default=str(REPO_ROOT / "data_pipeline" / "output" / "sft_data.jsonl"))
    p.add_argument("--personas", nargs="+",
                   default=["elem_low", "elem_high", "mid_low", "mid_high", "high_low", "high_high"])
    p.add_argument("--max-new", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0, help="0이면 greedy")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.problem is None:
        import json, random
        random.seed(args.seed)
        with open(args.data) as f:
            rows = [json.loads(l) for l in f]
        chosen = random.choice(rows)
        problem = chosen["problem"]
        gt = chosen.get("ground_truth", "?")
        print(f"[sampled problem id={chosen.get('problem_id')}  gt={gt}]")
    else:
        problem = args.problem

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[device] {device}  [dtype] {dtype}")
    print(f"[base] {args.base}")
    print(f"[adapter] {args.adapter}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, args.adapter).to(device).eval()

    personas = {p["id"]: p for p in load_personas(REPO_ROOT / "personas.json")}

    print("\n" + "#" * 70)
    print(f"PROBLEM: {problem}")
    print("#" * 70)

    do_sample = args.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=args.max_new,
        do_sample=do_sample,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=6,
    )
    if do_sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = 0.95

    for pid in args.personas:
        if pid not in personas:
            print(f"[skip] unknown persona: {pid}")
            continue
        persona = personas[pid]
        prompt = f"{persona['tag']}\nProblem: {problem}\nSolution:\n"
        enc = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        text = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        print("\n" + "=" * 70)
        print(f"[{pid}]  grade_band={persona.get('grade_band')}  level={persona.get('level')}")
        print("=" * 70)
        print(text.strip())


if __name__ == "__main__":
    main()
