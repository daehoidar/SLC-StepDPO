"""
data/analyze_flip_rate.py

Stage 3.5: 데이터셋의 label flip rate 통계 측정.

Proposition 3에 의해, label flip이 존재하면 (A7) belief-dependent reward 가정이
empirical하게 정당화된다. 본 스크립트는 다음 통계를 계산한다:

1. Type-1 pair 중 reject_persona 비율 (단순 conditioning을 넘어선 신호의 양)
2. Type-2 belief-flip pair의 수 및 비율
3. Persona pair별 flip 빈도 매트릭스 (어떤 페르소나 조합에서 flip이 많은지)
4. Step text level의 unique flip cases

Usage:
    python data/analyze_flip_rate.py --pairs data/preference_pairs.jsonl \\
                                      --output data/flip_stats.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", default="data/flip_stats.json")
    args = parser.parse_args()

    pairs = []
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    n_total = len(pairs)
    n_type1 = sum(1 for p in pairs if p["pair_type"] == "step_pair")
    n_type2 = sum(1 for p in pairs if p["pair_type"] == "belief_flip_pair")

    # Type-1 내 reject 유형 분포
    type1_reject_counts = Counter(
        p["reject_type"] for p in pairs if p["pair_type"] == "step_pair"
    )

    # Type-2 페르소나 매트릭스: (cur_persona, flip_persona) → count
    flip_matrix = defaultdict(int)
    for p in pairs:
        if p["pair_type"] == "belief_flip_pair":
            flip_matrix[(p["persona_id"], p["flip_persona_id"])] += 1

    # 고유 flip step text 개수
    unique_flip_steps = set()
    for p in pairs:
        if p["pair_type"] == "belief_flip_pair":
            unique_flip_steps.add(p["step_lose"])

    # 핵심 지표
    label_flip_rate_type2 = n_type2 / max(1, n_total)
    reject_persona_share = type1_reject_counts.get("reject_persona", 0) / max(1, n_type1)

    stats = {
        "n_total_pairs": n_total,
        "n_type1": n_type1,
        "n_type2_belief_flip": n_type2,
        "label_flip_rate_type2": round(label_flip_rate_type2, 4),
        "type1_reject_distribution": {
            "reject_math": type1_reject_counts.get("reject_math", 0),
            "reject_persona": type1_reject_counts.get("reject_persona", 0),
        },
        "reject_persona_share_in_type1": round(reject_persona_share, 4),
        "n_unique_flip_step_texts": len(unique_flip_steps),
        "flip_matrix": {
            f"{cur}__{flip}": cnt for (cur, flip), cnt in flip_matrix.items()
        },
    }

    # Proposition 3 검증 명시
    stats["proposition_3_verification"] = {
        "label_flip_observed": n_type2 > 0,
        "interpretation": (
            "Label flip rate > 0 implies (A7) belief-dependent reward is "
            "empirically supported by this dataset."
        ),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 콘솔 출력
    print("=" * 60)
    print("Label Flip Rate Statistics")
    print("=" * 60)
    print(f"Total pairs:               {n_total}")
    print(f"  - Type-1 (step_pair):    {n_type1}")
    print(f"  - Type-2 (belief_flip):  {n_type2}")
    print(f"")
    print(f"Label flip rate (Type-2 / Total): {label_flip_rate_type2:.2%}")
    print(f"Reject-persona share in Type-1:   {reject_persona_share:.2%}")
    print(f"Unique flip step texts:           {len(unique_flip_steps)}")
    print(f"")
    print("Flip matrix (top 10):")
    for (cur, flip), cnt in sorted(flip_matrix.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cur:>12} ⇄ {flip:>12} : {cnt}")
    print(f"")
    print(f"→ Saved to {args.output}")
    print()
    if n_type2 > 0:
        print("[Proposition 3] Label flip observed → (A7) empirically justified ✓")
    else:
        print("[WARNING] No label flips found! (A7) cannot be empirically justified.")
        print("Consider: stronger persona definitions, or more cross-belief check coverage.")


if __name__ == "__main__":
    main()
