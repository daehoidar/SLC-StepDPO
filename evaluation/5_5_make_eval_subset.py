"""페르소나별 동일 개수의 균형 평가 서브셋 생성.

태스크 1(형식 준수 체크)을 빠르게 돌리기 위한 held-out 평가용 서브셋을
만든다. 입력(기본 sft_test.jsonl)에서 페르소나별로 정확히 --per-persona개씩
뽑아 저장한다.

설계:
  - 페르소나별 *동일 개수* 보장 (균형). 모자라면 에러.
  - problem_id 단위로 뽑아 같은 base 문제의 다른 페르소나 변형이 함께 들어가도
    무방하다(이미 train과는 problem_id가 분리되어 있다고 가정 — sft_test 입력).
  - --train 을 주면 평가 서브셋의 problem_id가 train과 겹치지 않는지 검증한다.

Usage:
    python data_pipeline/make_eval_subset.py \
        --input data_pipeline/output/sft_test.jsonl \
        --output data_pipeline/output/sft_test_eval60.jsonl \
        --per-persona 10 --seed 0 \
        --train data_pipeline/output/sft_train.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data_pipeline/output/sft_test.jsonl")
    p.add_argument("--output", default="data_pipeline/output/sft_test_eval60.jsonl")
    p.add_argument("--per-persona", type=int, default=10,
                   help="페르소나별로 뽑을 행 수 (균형 보장)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", default=None,
                   help="주어지면 서브셋 problem_id가 train과 겹치지 않는지 검증")
    args = p.parse_args()

    rows = load_jsonl(args.input)
    by_persona: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_persona[r["persona_id"]].append(r)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    for pid in sorted(by_persona):
        pool = by_persona[pid]
        if len(pool) < args.per_persona:
            raise RuntimeError(
                f"persona '{pid}' has only {len(pool)} rows < per-persona "
                f"{args.per_persona}. 더 작은 --per-persona 값을 쓰세요."
            )
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        picked.extend(pool[i] for i in idx[:args.per_persona])

    rng.shuffle(picked)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = Counter(r["persona_id"] for r in picked)
    print(f"[input]   {args.input}  rows={len(rows)}")
    print(f"[output]  {args.output}  rows={len(picked)}")
    print(f"[balance] per-persona={args.per_persona}  -> {dict(sorted(dist.items()))}")
    assert len(set(dist.values())) == 1, "페르소나별 개수가 균형이 아닙니다!"

    if args.train:
        train_pids = {r["problem_id"] for r in load_jsonl(args.train)}
        eval_pids = {r["problem_id"] for r in picked}
        overlap = train_pids & eval_pids
        if overlap:
            raise RuntimeError(
                f"[leak] eval 서브셋의 problem_id {len(overlap)}개가 train과 겹칩니다: "
                f"{sorted(overlap)[:5]}..."
            )
        print(f"[ok] train과 problem_id 겹침 없음 (eval unique problems={len(eval_pids)})")


if __name__ == "__main__":
    main()
