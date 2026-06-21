"""정성 비교 그림용 실제 생성 — 같은 문제를 초등학생(elem_low)·고등학생(high_high) 페르소나로.

eval(5_evaluate.generate)과 동일한 프롬프트/그리디 세팅 사용.
사용: python scripts/gen_qual_examples.py --model checkpoints/bc_3term_merged --output eval/qual_bc.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vllm import LLM, SamplingParams

from utils import load_personas

# held-out 인덱스에서 레벨대비 좋은 문제 (sft_test_heldout200.jsonl 기준)
PICK_IDX = [8, 22, 13, 5, 0, 11]
PERSONA_IDS = ["elem_low", "high_high"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--personas-path", default="personas.json")
    ap.add_argument("--test-set", default="data_pipeline/output/sft_test_heldout200.jsonl")
    ap.add_argument("--problems-file", default=None,
                    help="주어지면 이 jsonl의 모든 문제를 사용(PICK_IDX 무시)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.problems_file:
        problems = [json.loads(l) for l in open(args.problems_file)]
    else:
        rows = [json.loads(l) for l in open(args.test_set)]
        problems = [rows[i] for i in PICK_IDX]
    personas = {p["id"]: p for p in load_personas(args.personas_path)}
    persL = [personas[i] for i in PERSONA_IDS]

    llm = LLM(model=args.model, dtype="bfloat16")
    sp = SamplingParams(temperature=0.0, max_tokens=800)

    prompts, meta = [], []
    for p in problems:
        q = p.get("problem") or p.get("question")
        for pers in persL:
            prompts.append(f"{pers['tag']}\nProblem: {q}\nSolution:\n")
            meta.append({"problem": q, "persona_id": pers["id"], "tag": pers["tag"]})
    outs = llm.generate(prompts, sp)

    res = []
    for m, o in zip(meta, outs):
        res.append({**m, "output": o.outputs[0].text.strip()})
    json.dump(res, open(args.output, "w"), ensure_ascii=False, indent=2)
    print(f"[saved] {args.output} ({len(res)} gens)")


if __name__ == "__main__":
    main()
