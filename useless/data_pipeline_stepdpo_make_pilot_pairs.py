"""data_pipeline/make_pilot_pairs.py

태스크 2의 *파일럿* — GPT-4o 없이 선호쌍(win/lose)을 결정론적으로 생성한다.

목적: (B) 로그확률 차이 분석 파이프라인을 GPT-4o/API 없이 먼저 검증하기 위한
"정답이 분명한" 쌍을 만든다. 두 종류:

  Type-1 (type1_math)   : 수학 정오 신호.
      win  = 정답이 든 마지막 step
      lose = 같은 step에서 정답 숫자만 틀리게 바꾼 것
      (같은 persona_tag, 같은 prefix) → 모델이 맞는 쪽에 더 높은 logp를 주는가?

  Type-2 (type2_belief) : belief-flip(페르소나) 신호.
      win  = 풀이 전체를 *작성된(적합한) persona* 태그 아래에서
      lose = *같은 텍스트*를 초등 저학년(elem_low) 태그 아래에서
      (초등 금지어를 포함한 고학년 풀이만 선별 → lose belief가 진짜 부적합)
      → 모델이 적합한 belief에서 더 높은 logp를 주는가? (Type-2 belief-flip 축소판)

출력 스키마(일반형 — win/lose 각각 자기 컨텍스트를 들고 있어 두 타입을 통일):
  {
    "pair_id", "pair_type", "persona_id",
    "win":  {"persona_tag", "problem", "prefix_steps":[...], "step"},
    "lose": {"persona_tag", "problem", "prefix_steps":[...], "step"},
    "meta": {...}
  }

Usage:
    python data_pipeline/make_pilot_pairs.py \
        --input data_pipeline/output/sft_test.jsonl \
        --personas-path personas.json \
        --output data_pipeline/output/pilot_pairs.jsonl \
        --n-per-type 80 --seed 0
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

from utils import load_personas, parse_steps  # noqa: E402


def _corrupt_number(text: str, gt: str, rng: random.Random) -> str | None:
    """마지막 step 안의 정답 숫자 gt를 *다른* 숫자로 바꿔 틀린 step을 만든다."""
    if not re.search(r"\d", gt):
        return None
    if gt not in text:
        return None
    try:
        base = int(re.sub(r"[^\d-]", "", gt))
    except ValueError:
        return None
    # gt와 확실히 다른 값으로 교란 (±1~9, 0 제외)
    delta = rng.choice([d for d in range(-9, 10) if d != 0])
    wrong = str(base + delta)
    if wrong == gt:
        wrong = str(base + 10)
    return text.replace(gt, wrong)


def _forbidden_regex(terms: list[str]) -> re.Pattern | None:
    terms = [t for t in terms if t]
    if not terms:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data_pipeline/output/sft_test.jsonl")
    ap.add_argument("--personas-path", default="personas.json")
    ap.add_argument("--output", default="data_pipeline/output/pilot_pairs.jsonl")
    ap.add_argument("--n-per-type", type=int, default=80,
                    help="타입별 최대 쌍 수")
    ap.add_argument("--lose-belief", default="elem_high",
                    help="Type-2에서 lose(부적합) belief로 쓸 persona id. "
                         "elem_high 금지어(equation/variable/function 등)가 고학년 "
                         "풀이에 자주 나와 belief-flip 쌍이 풍부하다.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    personas = {p["id"]: p for p in load_personas(args.personas_path)}
    if args.lose_belief not in personas:
        ap.error(f"--lose-belief '{args.lose_belief}' 가 personas에 없음")
    lose_p = personas[args.lose_belief]
    forbid_re = _forbidden_regex(lose_p.get("forbidden_terms", []))

    pairs: list[dict] = []

    # ── Type-1: 수학 정오 ────────────────────────────────────────────
    t1 = []
    for r in rows:
        steps = parse_steps(r["solution_text"])
        if len(steps) < 2:
            continue
        gt = str(r.get("ground_truth", "")).strip()
        last = steps[-1]
        lose_step = _corrupt_number(last, gt, rng)
        if lose_step is None or lose_step == last:
            continue
        tag = r["persona_tag"]
        ctx = {"persona_tag": tag, "problem": r["problem"],
               "prefix_steps": steps[:-1]}
        t1.append({
            "pair_type": "type1_math",
            "persona_id": r["persona_id"],
            "win":  {**ctx, "step": last},
            "lose": {**ctx, "step": lose_step},
            "meta": {"problem_id": r["problem_id"], "ground_truth": gt},
        })
    rng.shuffle(t1)
    pairs += t1[:args.n_per_type]

    # ── Type-2: belief-flip (페르소나 부적합) ─────────────────────────
    t2 = []
    if forbid_re is not None:
        for r in rows:
            if r["persona_id"] == args.lose_belief:
                continue  # 자기 자신은 제외
            sol = r["solution_text"]
            if not forbid_re.search(sol):
                continue  # lose belief의 금지어가 없으면 부적합 보장 불가 → skip
            problem = r["problem"]
            win_ctx = {"persona_tag": r["persona_tag"], "problem": problem,
                       "prefix_steps": []}
            lose_ctx = {"persona_tag": lose_p["tag"], "problem": problem,
                        "prefix_steps": []}
            term = forbid_re.search(sol).group(1)
            t2.append({
                "pair_type": "type2_belief",
                "persona_id": r["persona_id"],
                "win":  {**win_ctx, "step": sol},
                "lose": {**lose_ctx, "step": sol},
                "meta": {"problem_id": r["problem_id"],
                         "lose_belief": args.lose_belief,
                         "trigger_forbidden_term": term},
            })
    rng.shuffle(t2)
    pairs += t2[:args.n_per_type]

    for i, p in enumerate(pairs):
        p["pair_id"] = f"{p['pair_type']}_{i:05d}"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n1 = sum(1 for p in pairs if p["pair_type"] == "type1_math")
    n2 = sum(1 for p in pairs if p["pair_type"] == "type2_belief")
    print(f"[input]  {args.input}  rows={len(rows)}")
    print(f"[type1_math]   candidates={len(t1)}  -> used={n1}")
    print(f"[type2_belief] candidates={len(t2)}  -> used={n2}  (lose_belief={args.lose_belief})")
    print(f"[output] {args.output}  total_pairs={len(pairs)}")
    if n2 == 0:
        print("[WARN] Type-2 쌍 0개 — lose belief 금지어가 풀이에 안 나타남. "
              "--lose-belief 변경 또는 personas.json forbidden_terms 보강 필요.")


if __name__ == "__main__":
    main()
