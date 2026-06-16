"""SFT 데이터를 train/test로 problem_id 단위 분할.

같은 base 문제의 다른 증강·페르소나 변형이 train과 test에 동시에 들어가지 않도록
problem_id 기준으로 분리한다.

Usage:
    python data_pipeline/split_train_test.py \
        --input data_pipeline/output/sft_data.jsonl \
        --train-out data_pipeline/output/sft_train.jsonl \
        --test-out data_pipeline/output/sft_test.jsonl \
        --n-test-problems 15 --seed 0
"""
from __future__ import annotations
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data_pipeline/output/sft_data.jsonl")
    p.add_argument("--train-out", default="data_pipeline/output/sft_train.jsonl")
    p.add_argument("--test-out", default="data_pipeline/output/sft_test.jsonl")
    p.add_argument("--n-test-problems", type=int, default=15,
                   help="test로 뺄 base 문제 갯수")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    by_pid = defaultdict(list)
    for r in rows:
        by_pid[r["problem_id"]].append(r)

    pids = sorted(by_pid.keys())
    random.seed(args.seed)
    random.shuffle(pids)
    test_pids = set(pids[:args.n_test_problems])
    train_pids = set(pids[args.n_test_problems:])

    train_rows = [r for pid in train_pids for r in by_pid[pid]]
    test_rows  = [r for pid in test_pids  for r in by_pid[pid]]
    random.shuffle(train_rows)
    random.shuffle(test_rows)

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_out, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.test_out, "w", encoding="utf-8") as f:
        for r in test_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[input]  total rows={len(rows)}  unique problems={len(by_pid)}")
    print(f"[split]  test_problems={len(test_pids)}  train_problems={len(train_pids)}")
    print(f"[train]  {args.train_out}  rows={len(train_rows)}")
    print(f"[test]   {args.test_out}   rows={len(test_rows)}")

    def persona_count(rows):
        return Counter(r.get("persona_id") for r in rows)

    print(f"\n[persona dist in train] {dict(persona_count(train_rows))}")
    print(f"[persona dist in test]  {dict(persona_count(test_rows))}")

    overlap = set(r["problem_id"] for r in train_rows) & set(r["problem_id"] for r in test_rows)
    if overlap:
        raise RuntimeError(f"problem_id overlap! {overlap}")
    print("\n[ok] no problem_id overlap between splits.")


if __name__ == "__main__":
    main()
