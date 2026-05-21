"""MetaMathQA-40K에서 문제를 샘플링하여 페르소나 6종 공통 풀로 배정한다.

설계:
- MetaMathQA-40K (arXiv 2309.12284, meta-math/MetaMathQA-40K)는
  GSM8K + MATH 두 원본을 4가지 방식으로 augment한 데이터셋:
    AnsAug    (같은 query → 다른 response): *질문 중복*
    Rephrased (같은 의미, 다른 query 표현)
    FOBAR     (forward → backward 변환, query 구조 변경)
    SV        (self-verification 형식 변환)
- 본 파이프라인은 **GSM 계열만** 사용 (type 컬럼이 'GSM_'으로 시작).
- AnsAug가 같은 query 위에 다른 response를 붙이므로 학습 데이터 중복을 유발.
  → **query 컬럼 기준 dedupe**로 unique 문제만 채택.
  (Rephrased/FOBAR/SV는 query가 달라 자연스럽게 별 행으로 살아남음.)
- 난이도 버킷팅은 response 내 GSM8K 형식 <<X+Y=Z>> 마커 개수 + query 길이로 추정.
  마커가 없는 변형(FOBAR/SV 일부)은 query 길이만으로 보수적 분류.
- 같은 문제를 6 페르소나 모두에 복제 배정 (belief_pair 단계의 cross-persona
  비교 자연 발생을 위함).

출력: data_pipeline/output/seed_problems.jsonl
한 행 = (문제, 페르소나) 한 쌍. 총 행 수 = N × 6.

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

from utils import load_personas  # noqa: E402

# 6 페르소나가 공통으로 사용할 난이도 버킷 (hard 제외).
COMMON_BUCKETS = ["easy", "medium"]

OPS_RE = re.compile(r"<<[^>]+>>")
# MetaMathQA response 형식: "...calculations...\nThe answer is: X" 또는 "The answer is X"
ANS_RE = re.compile(r"The answer is:?\s*([^\.\n]+?)\.?\s*$", re.MULTILINE)


def difficulty_bucket(query: str, response: str) -> str:
    """response의 <<...>> 마커 개수 + query 길이로 난이도 추정.

    GSM8K augmentation은 보통 원본의 <<X+Y=Z>> 형식을 보존한다. Rephrased/FOBAR/SV
    중 일부는 마커가 사라질 수 있어 fallback으로 query 길이만 본다.
    """
    n_ops = len(OPS_RE.findall(response))
    q_words = len(query.split())
    if n_ops > 0:
        if n_ops <= 2 and q_words <= 30:
            return "easy"
        if n_ops <= 4 and q_words <= 60:
            return "medium"
        return "hard"
    # 마커 없음 → query 길이로 보수적 분류
    if q_words <= 30:
        return "medium"
    return "hard"


def extract_gt_answer(response: str) -> str:
    """response 끝의 'The answer is: X' 또는 'The answer is X'에서 X 추출."""
    m = ANS_RE.search(response)
    if m:
        return m.group(1).strip().rstrip(".")
    # fallback: 마지막 비공백 줄
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def load_metamath():
    from datasets import load_dataset
    return load_dataset("meta-math/MetaMathQA-40K", split="train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=1500,
                    help="공통 풀에서 뽑을 문제 개수 (각 문제는 6 페르소나 모두에 배정)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                    default=str(REPO_ROOT / "data_pipeline" / "output" / "seed_problems.jsonl"))
    ap.add_argument("--include-math", action="store_true",
                    help="기본은 GSM_*만. 본 플래그 시 MATH_*도 포함.")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[load] MetaMathQA-40K")
    ds = load_metamath()
    print(f"[load] {len(ds)} rows (raw)")

    # 소스 데이터셋 필터 (기본: GSM 계열만)
    prefix_allow = ("GSM_",) if not args.include_math else ("GSM_", "MATH_")
    filtered = [r for r in ds if r.get("type", "").startswith(prefix_allow)]
    by_type: dict[str, int] = {}
    for r in filtered:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print(f"[filter] allowed={prefix_allow} → {len(filtered)} rows")
    for t, n in sorted(by_type.items()):
        print(f"  - {t}: {n}")

    # query 기준 dedupe (AnsAug의 같은 query 중복 제거).
    # Rephrased/FOBAR/SV는 query가 달라 별 행으로 살아남음.
    seen_first: dict[str, dict] = {}
    for r in filtered:
        q = r["query"]
        if q not in seen_first:
            seen_first[q] = r
    unique_rows = list(seen_first.values())
    print(f"[dedupe] unique query: {len(unique_rows)} "
          f"(removed {len(filtered) - len(unique_rows)} duplicate-query rows)")

    # 버킷 분류
    buckets = {"easy": [], "medium": [], "hard": []}
    for idx, it in enumerate(unique_rows):
        b = difficulty_bucket(it["query"], it.get("response", ""))
        buckets[b].append({
            "problem_id": f"metamath_{idx}",
            "question": it["query"],
            "gt_answer_raw": it["response"],
            "gt_answer": extract_gt_answer(it["response"]),
            "augmentation_type": it.get("type", ""),
            "difficulty": b,
            "n_ops": len(OPS_RE.findall(it.get("response", ""))),
        })
    for b, lst in buckets.items():
        print(f"[bucket] {b}: {len(lst)}")

    # 공통 풀 구성 (easy + medium)
    common_pool = []
    for b in COMMON_BUCKETS:
        common_pool.extend(buckets[b])
    print(f"[common pool] {len(common_pool)} (buckets={COMMON_BUCKETS})")

    random.shuffle(common_pool)
    picked = common_pool[: args.n_problems]
    if len(picked) < args.n_problems:
        print(f"[warn] 요청 {args.n_problems}개 대비 풀 크기 {len(picked)}개. 가능한 만큼만 사용.")
    print(f"[pick] {len(picked)}개 문제 선정")

    # personas.json에서 페르소나 id 목록 로드
    personas = load_personas(REPO_ROOT / "personas.json")
    persona_ids = [p["id"] for p in personas]
    print(f"[personas] {len(persona_ids)}종: {persona_ids}")

    # 6 페르소나 복제 배정
    n_total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in picked:
            for pid in persona_ids:
                row = {
                    "problem_id": item["problem_id"],
                    "persona": pid,
                    "question": item["question"],
                    "gt_answer": item["gt_answer"],
                    "gt_answer_raw": item["gt_answer_raw"],
                    "difficulty": item["difficulty"],
                    "augmentation_type": item["augmentation_type"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_total += 1
    print(f"\n[done] 문제 {len(picked)} × 페르소나 {len(persona_ids)} = {n_total}행")
    print(f"[done] → {out_path}")


if __name__ == "__main__":
    main()
