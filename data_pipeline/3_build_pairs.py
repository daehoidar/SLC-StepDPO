"""
data/build_pairs.py

Stage 3: SFT 모델(π_ref)로 on-policy 풀이 K개 샘플링, GPT-4o judge로
belief-conditional win/lose 라벨링, Type-1/Type-2 preference pair 구축.

두 종류의 pair:
  Type-1 (step_pair): 같은 belief 내, 같은 prefix 위의
                       acceptable step (win) vs unacceptable step (lose)
  Type-2 (belief_flip_pair): 같은 prefix 위의 같은 텍스트 step이
                              한 belief에선 acceptable, 다른 belief에선 unacceptable

핵심 출력 (JSONL):
{
  "problem_id": "...",
  "problem": "...",
  "persona_id": "elem_low",
  "persona_tag": "<초등-하위권>",
  "prefix_steps": ["Step 1: ..."],
  "step_win": "Step 2: ...",
  "step_lose": "Step 2: ...",
  "pair_type": "step_pair" | "belief_flip_pair",
  "reject_type": "reject_math" | "reject_persona" | "n/a",
  "flip_persona_id": "high_high" or null   # Type-2일 때만, 반대 belief
}
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from judge_prompts import (  # noqa: E402
    STEP_JUDGE_SYSTEM, STEP_JUDGE_USER_TEMPLATE,
    CROSS_BELIEF_CHECK_SYSTEM, CROSS_BELIEF_CHECK_USER_TEMPLATE,
    build_step_judge_kwargs, build_cross_belief_kwargs,
)
from utils import load_personas, parse_steps  # noqa: E402


def format_steps_numbered(steps: list[str]) -> str:
    return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(steps))


def call_step_judge(client: OpenAI, problem: dict, steps: list[str], persona: dict) -> dict:
    """각 step의 belief-conditional 라벨링."""
    sys_prompt = STEP_JUDGE_SYSTEM.format(**build_step_judge_kwargs(persona))
    user_prompt = STEP_JUDGE_USER_TEMPLATE.format(
        problem=problem["problem"],
        ground_truth=problem["ground_truth"],
        persona_tag=persona["tag"],
        solution_with_steps=format_steps_numbered(steps),
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"steps": [], "first_reject_step": None, "first_reject_type": None}


def call_cross_belief_check(
    client: OpenAI, step_text: str, prefix_text: str, problem: dict,
    persona_a: dict, persona_b: dict,
) -> dict:
    """동일 step이 두 페르소나에서 다른 라벨을 받는지 확인."""
    sys_prompt = CROSS_BELIEF_CHECK_SYSTEM.format(
        **build_cross_belief_kwargs(
            step_text=step_text,
            prefix_text=prefix_text,
            problem=problem["problem"],
            persona_a=persona_a,
            persona_b=persona_b,
        )
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": CROSS_BELIEF_CHECK_USER_TEMPLATE},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {"persona_a_acceptable": False, "persona_b_acceptable": False, "flip": False}


def sample_solutions(
    llm: LLM, problem: dict, persona: dict, k: int = 8, temperature: float = 0.9,
) -> list[list[str]]:
    prompt = (
        f"{persona['tag']}\n"
        f"Problem: {problem['problem']}\n"
        f"Solution:\n"
    )
    sp = SamplingParams(temperature=temperature, max_tokens=800, n=k, stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    return [parse_steps(o.text) for o in outputs[0].outputs]


def build_type1_pairs(judged_samples: list[dict], problem: dict, persona: dict) -> list[dict]:
    """Type-1: 같은 belief 내, 같은 prefix 위의 acceptable vs reject step.

    수학 오류든 페르소나 drift든 모두 reject로 통합. reject_type만 보조 정보로 저장.
    """
    # prefix → [{step, label, reject_type, step_idx}]
    prefix_groups = defaultdict(list)
    for j in judged_samples:
        steps = j["steps"]
        step_labels = {s["index"]: (s["label"], s.get("reason", "")) for s in j["judge"].get("steps", [])}
        for i in range(len(steps)):
            label_info = step_labels.get(i + 1)
            if label_info is None:
                continue
            label, _ = label_info
            prefix_key = "||".join(steps[:i])
            prefix_groups[prefix_key].append({
                "step": steps[i],
                "label": label,  # acceptable / reject_math / reject_persona
                "step_idx": i + 1,
            })

    pairs = []
    for prefix_key, candidates in prefix_groups.items():
        wins = [c for c in candidates if c["label"] == "acceptable"]
        loses = [c for c in candidates if c["label"].startswith("reject_")]
        if not wins or not loses:
            continue
        # 페어 수 제한: prefix당 최대 2개
        for w in wins[:2]:
            for l in loses[:2]:
                pairs.append({
                    "problem_id": problem["problem_id"],
                    "problem": problem["problem"],
                    "persona_id": persona["id"],
                    "persona_tag": persona["tag"],
                    "prefix_steps": prefix_key.split("||") if prefix_key else [],
                    "step_win": w["step"],
                    "step_lose": l["step"],
                    "pair_type": "step_pair",
                    "reject_type": l["label"],  # reject_math or reject_persona
                    "flip_persona_id": None,
                })
    return pairs


def build_type2_pairs(
    client: OpenAI, type1_pairs: list[dict], personas: list[dict],
    max_per_problem: int = 3,
) -> list[dict]:
    """Type-2: 동일 step이 다른 belief에선 다른 라벨을 받는 경우.

    Strategy: type1_pairs 중 reject_persona인 lose step을 추출 → 그 step이
    "다른 페르소나(예: 고학년)"에서는 acceptable한지 cross-belief check.
    그렇다면 그 step을 "다른 페르소나의 win step"이자 "현재 페르소나의 lose step"으로
    사용하여 새 Type-2 pair 생성.
    """
    persona_by_id = {p["id"]: p for p in personas}
    type2_pairs = []
    problem_count = defaultdict(int)

    for p in type1_pairs:
        if p["reject_type"] != "reject_persona":
            continue  # 수학 오류는 belief flip 후보가 아님
        if problem_count[p["problem_id"]] >= max_per_problem:
            continue

        # 현재 페르소나
        cur_persona = persona_by_id[p["persona_id"]]
        # 후보 페르소나 (학년대 또는 난이도가 다른 것)
        candidate_personas = [
            other for other in personas
            if other["id"] != cur_persona["id"]
            and (other["grade_band"] != cur_persona["grade_band"]
                 or other["level"] != cur_persona["level"])
        ]

        for other_persona in candidate_personas[:2]:  # 후보 2개만 시도
            check = call_cross_belief_check(
                client=client,
                step_text=p["step_lose"],
                prefix_text="\n".join(p["prefix_steps"]),
                problem={"problem": p["problem"]},
                persona_a=cur_persona,
                persona_b=other_persona,
            )
            if check.get("flip") and not check.get("persona_a_acceptable") and check.get("persona_b_acceptable"):
                # cur_persona에선 lose, other_persona에선 win
                # → Type-2 pair: cur_persona의 win step과 other_persona용 step을 lose로
                type2_pairs.append({
                    "problem_id": p["problem_id"],
                    "problem": p["problem"],
                    "persona_id": p["persona_id"],
                    "persona_tag": p["persona_tag"],
                    "prefix_steps": p["prefix_steps"],
                    "step_win": p["step_win"],  # 현 페르소나에 적합한 step
                    "step_lose": p["step_lose"],  # 다른 페르소나엔 OK지만 현 페르소나엔 부적합
                    "pair_type": "belief_flip_pair",
                    "reject_type": "reject_persona",
                    "flip_persona_id": other_persona["id"],
                })
                problem_count[p["problem_id"]] += 1
                break  # 한 (problem, persona)당 한 flip이면 충분

    return type2_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-model", required=True)
    parser.add_argument("--seed-problems", required=True)
    parser.add_argument("--personas-path", default="data/personas.json")
    parser.add_argument("--k-samples", type=int, default=8)
    parser.add_argument("--output", default="data/preference_pairs.jsonl")
    args = parser.parse_args()

    client = OpenAI()
    llm = LLM(model=args.ref_model, dtype="bfloat16", gpu_memory_utilization=0.85)
    personas = load_personas(args.personas_path)

    problems = []
    with open(args.seed_problems, encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))

    all_type1, all_type2 = [], []

    for i, problem in enumerate(problems):
        for persona in personas:
            # On-policy 샘플링
            sample_steps_list = sample_solutions(llm, problem, persona, k=args.k_samples)
            # 각 샘플을 belief-conditional judge로 라벨링
            judged = []
            for steps in sample_steps_list:
                if len(steps) < 2:
                    continue
                judge_out = call_step_judge(client, problem, steps, persona)
                judged.append({"steps": steps, "judge": judge_out})
            # Type-1 pair 구축
            t1 = build_type1_pairs(judged, problem, persona)
            all_type1.extend(t1)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(problems)}] type1: {len(all_type1)}")

    # Type-2 pair 구축 (Type-1 결과에서 cross-belief check)
    print("Building Type-2 belief-flip pairs...")
    all_type2 = build_type2_pairs(client, all_type1, personas)

    # 통합 저장
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in all_type1 + all_type2:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Type-1 pairs: {len(all_type1)}")
    print(f"Type-2 pairs: {len(all_type2)}")
    print(f"Total: {len(all_type1) + len(all_type2)} → {args.output}")


if __name__ == "__main__":
    main()
