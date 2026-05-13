"""GSM8K train에서 문제를 샘플링하여 페르소나 6종 공통 풀로 배정한다.

설계:
- GSM8K answer 필드의 `<<...>>` 계산기 마커 개수로 추론 깊이를 추정해
  easy(1-2 ops) / medium(3-4 ops) / hard(5+ ops) 세 버킷으로 분류.
- hard는 제외하고 easy + medium만 사용하여 6종 페르소나 모두가 풀 만한 수준 확보.
- N개의 문제를 한 번 샘플링한 뒤, 같은 문제를 모든 페르소나에 복제 배정.
  => belief_pair 단계에서 동일 문제에 대한 6개 페르소나 풀이가 자연스럽게 모임.

출력: data_pipeline/output/seed_problems.jsonl
한 행 = (문제, 페르소나) 한 쌍. 총 행 수 = N x 6.

사용 예:
    python data_pipeline/0_seed_problems.py --n-problems 1500 --seed 42
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

# 6종 페르소나가 공통으로 사용하는 난이도 풀. hard는 제외.
COMMON_BUCKETS = ["easy", "medium"]

OPS_RE = re.compile(r"<<[^>]+>>")
ANS_RE = re.compile(r"####\s*(.+)$", re.MULTILINE)


# 너무 복잡한 문제는 제외하고 실험하기 위해 hard만 분류
def difficulty_bucket(item: dict) -> str:
    n_ops = len(OPS_RE.findall(item["answer"]))
    q_words = len(item["question"].split())
    
    if n_ops <= 2 and q_words <= 30:
        return "easy"
    if n_ops <= 4 and q_words <= 60:
        return "medium"    
    return "hard"

# GSM8K에서 정답만 추출할 수 있는 함수(#### 뒤의 텍스트를 답으로 출력)
def extract_gt_answer(answer_field: str) -> str:
    m = ANS_RE.search(answer_field)
    return m.group(1).strip() if m else ""

# HuggingFace datasets로 GSM8K train 로드. 캐시 사용.
def load_gsm8k_train():
    from datasets import load_dataset
    return load_dataset("gsm8k", "main", split="train")


def main():
    ap = argparse.ArgumentParser()
    '''문제 개수, 시드값, 출력 경로 입력받을 수 있음 '''
    ap.add_argument("--n-problems", type=int, default=1500, # 기본 1500개 추출
                    help="공통 풀에서 뽑을 문제 개수 (각 문제는 6종 페르소나 모두에 배정)")
    ap.add_argument("--seed", type=int, default=42) # 시드번호 디폴트 42
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
    

    # 공통 풀 구성 (easy + medium) = 1500개
    # 걍 따로 두는건 어떨까욤.. reasoning 난이도에 따라 얼마나 다르게 하는지 확인하기 위해?
    common_pool = []
    for b in COMMON_BUCKETS:
        common_pool.extend(buckets[b])
    print(f"[common pool] {len(common_pool)}개 (buckets={COMMON_BUCKETS})")

    random.shuffle(common_pool)
    picked = common_pool[: args.n_problems]
    if len(picked) < args.n_problems:
        print(f"[warn] 요청 {args.n_problems}개 대비 풀 크기 {len(picked)}개. 가능한 만큼만 사용.")
    print(f"[pick] {len(picked)}개 문제 선정")

    # 같은 문제를 6종 페르소나 모두에 복제 배정 = 9000개
    n_total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in picked:
            for age, level in PERSONA_GRID:
                pid = f"{age}-{level}"
                row = {
                    "problem_id": item["problem_id"],
                    "persona": pid,
                    "question": item["question"],
                    "gt_answer": item["gt_answer"],
                    "gt_answer_raw": item["gt_answer_raw"],
                    "difficulty": item["difficulty"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_total += 1
    print(f"\n[done] 문제 {len(picked)}개 x 페르소나 {len(PERSONA_GRID)}종 = {n_total}행")
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
