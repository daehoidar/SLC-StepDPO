"""GSM8K train에서 문제를 샘플링하고 페르소나별 난이도 버킷에 배정한다.

GSM8K answer 필드의 `<<...>>` 계산기 마커 개수를 추론 깊이의 프록시로 사용해
easy(1-2 ops) / medium(3-4 ops) / hard(5+ ops) 세 버킷으로 나눈 뒤,
페르소나마다 적합한 버킷에서 무작위 추출한다.

출력: data_pipeline/output/seed_problems.jsonl
한 행 = (문제, 페르소나) 한 쌍. 같은 문제가 여러 페르소나에 배정될 수 있다.

사용 예:
    python data_pipeline/0_seed_problems.py --per-persona 1500 --seed 42
"""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from personas import PERSONA_GRID  # noqa: E402

PERSONA_BUCKETS = {
    "초등-하위권": ["easy"],
    "초등-상위권": ["easy", "medium"],
    "중등-하위권": ["medium"],
    "중등-상위권": ["medium", "hard"],
    "고등-하위권": ["hard"],
    "고등-상위권": ["hard"],
}

OPS_RE = re.compile(r"<<[^>]+>>")
ANS_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)


def difficulty_bucket(item: dict) -> str:
    n_ops = len(OPS_RE.findall(item["answer"]))
    q_words = len(item["question"].split())
    if n_ops <= 2 and q_words <= 30:
        return "easy"
    if n_ops <= 4 and q_words <= 60:
        return "medium"
    return "hard"


def extract_gt_answer(answer_field: str) -> str:
    m = ANS_RE.search(answer_field)
    return m.group(1).strip() if m else ""


def load_gsm8k_train():
    """HuggingFace datasets로 GSM8K train 로드. 캐시 사용."""
    from datasets import load_dataset
    return load_dataset("gsm8k", "main", split="train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-persona", type=int, default=1500,
                    help="페르소나당 샘플 개수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default=str(REPO_ROOT / "data_pipeline" / "output" / "seed_problems.jsonl"))
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[load] GSM8K train")
    ds = load_gsm8k_train()
    print(f"[load] {len(ds)} problems")

    # 버킷별 분류
    buckets = {"easy": [], "medium": [], "hard": []}
    for idx, it in enumerate(ds):
        b = difficulty_bucket(it)
        buckets[b].append({
            "problem_id": f"gsm8k_train_{idx}",
            "question": it["question"],
            "gt_answer_raw": it["answer"],
            "gt_answer": extract_gt_answer(it["answer"]),
            "difficulty": b,
            "n_ops": len(OPS_RE.findall(it["answer"])),
        })
    for b, lst in buckets.items():
        print(f"[bucket] {b}: {len(lst)}개")

    n_total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for age, level in PERSONA_GRID:
            pid = f"{age}-{level}"
            allowed = PERSONA_BUCKETS[pid]
            pool = []
            for b in allowed:
                pool.extend(buckets[b])
            random.shuffle(pool)
            picked = pool[: args.per_persona]
            for item in picked:
                row = {
                    "problem_id": item["problem_id"],
                    "persona": pid,
                    "question": item["question"],
                    "gt_answer": item["gt_answer"],
                    "gt_answer_raw": item["gt_answer_raw"],
                    "difficulty": item["difficulty"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_total += len(picked)
            print(f"[persona] {pid}: {len(picked)}개 (buckets={allowed})")
    print(f"\n[done] 총 {n_total}행 -> {out_path}")


if __name__ == "__main__":
    main()
