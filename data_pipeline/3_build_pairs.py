"""
data/build_pairs.py

Stage 3 (Full-Step-DPO via GPT-4o judge, persona cascade 적용):
SFT 모델(π_ref)로 K개 풀이 샘플 → step마다 (a) PersonaVerifier로 페르소나 라벨,
(b) GPT-4o로 수학 라벨 산출 → Type-1/Type-2 preference pair 구축.

페르소나 검증은 더 이상 step judge prompt에 통째로 들어가지 않는다. cascade
A→B→C가 별도 모듈에서 처리하며, GPT-4o는 (i) 수학 단일-step judge, (ii) Type-2
belief-flip cross-check에만 호출된다. → GPT-4o 호출량 ~80% 감소.

두 종류의 pair:
  Type-1 (step_pair): 같은 belief 내, 같은 prefix 위의
                       acceptable step (win) vs unacceptable step (lose).
  Type-2 (belief_flip_pair): 같은 prefix 위의 같은 텍스트 step이
                              한 belief에선 acceptable, 다른 belief에선 unacceptable.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import openai  # noqa: E402
from openai import OpenAI  # noqa: E402

try:
    from vllm import LLM, SamplingParams  # type: ignore  # noqa: E402
    _VLLM_AVAILABLE = True
except ImportError:
    from inference_backend import (  # noqa: E402
        TransformersLLM as LLM,
        TransformersSamplingParams as SamplingParams,
    )
    _VLLM_AVAILABLE = False

from judge_prompts import (  # noqa: E402
    CROSS_BELIEF_CHECK_SYSTEM, CROSS_BELIEF_CHECK_USER_TEMPLATE,
    build_cross_belief_kwargs,
)
from utils import load_personas, parse_steps  # noqa: E402
from persona_verifier import PersonaVerifier  # noqa: E402


# ── 수학 single-step judge prompt (페르소나 정보 제외, 짧게 유지) ──────────

MATH_STEP_JUDGE_SYSTEM = """You judge whether a SINGLE math step is mathematically valid.
Persona/vocabulary is judged separately — focus ONLY on math correctness.

Output JSON only:
{
  "is_wrong": true | false,
  "confidence": 0.0-1.0,
  "reason": "<one short sentence>"
}
"""

MATH_STEP_JUDGE_USER = """Problem: {problem}
Ground-truth final answer: {ground_truth}
Previous steps (assumed valid):
{prefix_text}

Step to judge:
"{step}"

Output JSON only."""


def judge_math_step(client: OpenAI, model: str, problem: dict,
                    prefix: list[str], step: str) -> dict:
    user = MATH_STEP_JUDGE_USER.format(
        problem=problem["problem"],
        ground_truth=problem["ground_truth"],
        prefix_text="\n".join(prefix) if prefix else "(none)",
        step=step,
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": MATH_STEP_JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except openai.AuthenticationError:
            raise
        except Exception as e:
            print(f"[math-judge retry {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return {"is_wrong": False, "confidence": 0.0, "reason": "judge failed"}


def call_cross_belief_check(
    client: OpenAI, model: str, step_text: str, prefix_text: str, problem: dict,
    persona_a: dict, persona_b: dict,
) -> dict:
    sys_prompt = CROSS_BELIEF_CHECK_SYSTEM.format(
        **build_cross_belief_kwargs(
            step_text=step_text,
            prefix_text=prefix_text,
            problem=problem["problem"],
            persona_a=persona_a,
            persona_b=persona_b,
        )
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": CROSS_BELIEF_CHECK_USER_TEMPLATE},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"[cross-belief error] {e}")
        return {"persona_a_acceptable": False, "persona_b_acceptable": False,
                "flip": False}


def sample_solutions(
    llm: LLM, problem: dict, persona: dict, k: int = 8, temperature: float = 0.9,
) -> list[list[str]]:
    prompt = (
        f"{persona['tag']}\n"
        f"Problem: {problem['problem']}\n"
        f"Solution:\n"
    )
    sp = SamplingParams(temperature=temperature, max_tokens=800, n=k,
                        stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    return [parse_steps(o.text) for o in outputs[0].outputs]


def label_steps(
    verifier: PersonaVerifier, math_judge_client: OpenAI, math_judge_model: str,
    problem: dict, persona: dict, steps: list[str],
) -> list[dict]:
    """각 step에 대해 (persona, math) 두 축으로 라벨링.

    반환: [{ index, label, persona_stage, evidence_code, trigger_term,
             math_reason }] — label ∈ {acceptable, reject_persona, reject_math}.
    'reject_persona'가 'reject_math'보다 우선 (페르소나 위반이 있으면 그 step에서
    멈추는 게 자연스러움 — Step-DPO 정의와 일관).
    """
    out = []
    for i, step in enumerate(steps):
        p = verifier.verify_step(step, persona, prefix=steps[:i])
        if p.verdict == "reject_persona":
            out.append({
                "index": i + 1,
                "label": "reject_persona",
                "persona_stage": p.stage,
                "evidence_code": p.evidence_code,
                "trigger_term": p.trigger_term,
                "math_reason": None,
                "reason": p.reasoning,
            })
            continue
        # 페르소나 OK → 수학 검증
        m = judge_math_step(math_judge_client, math_judge_model, problem,
                            steps[:i], step)
        if m.get("is_wrong"):
            out.append({
                "index": i + 1,
                "label": "reject_math",
                "persona_stage": p.stage,
                "evidence_code": None,
                "trigger_term": None,
                "math_reason": m.get("reason", ""),
                "reason": m.get("reason", ""),
            })
        else:
            out.append({
                "index": i + 1,
                "label": "acceptable",
                "persona_stage": p.stage,
                "evidence_code": None,
                "trigger_term": None,
                "math_reason": None,
                "reason": "",
            })
    return out


def build_type1_pairs(judged_samples: list[dict], problem: dict,
                      persona: dict) -> list[dict]:
    """Type-1: 같은 belief 내, 같은 prefix 위의 acceptable vs reject step."""
    prefix_groups = defaultdict(list)
    for j in judged_samples:
        steps = j["steps"]
        labels_by_idx = {s["index"]: s for s in j["labels"]}
        for i in range(len(steps)):
            info = labels_by_idx.get(i + 1)
            if info is None:
                continue
            prefix_key = "||".join(steps[:i])
            prefix_groups[prefix_key].append({
                "step": steps[i],
                "label": info["label"],
                "evidence_code": info.get("evidence_code"),
                "trigger_term": info.get("trigger_term"),
                "persona_stage": info.get("persona_stage"),
                "step_idx": i + 1,
            })

    pairs = []
    for prefix_key, candidates in prefix_groups.items():
        wins = [c for c in candidates if c["label"] == "acceptable"]
        loses = [c for c in candidates if c["label"].startswith("reject_")]
        if not wins or not loses:
            continue
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
                    "pair_subtype": ("persona_first_error" if l["label"] == "reject_persona"
                                     else "math_first_error"),
                    "reject_type": l["label"],
                    "evidence_code": l.get("evidence_code"),
                    "trigger_term": l.get("trigger_term"),
                    "verifier_stage": l.get("persona_stage"),
                    "flip_persona_id": None,
                })
    return pairs


def build_type2_pairs(
    client: OpenAI, verifier: PersonaVerifier,
    type1_pairs: list[dict], personas: list[dict],
    cross_belief_model: str = "gpt-4o",
    max_per_problem: int = 3,
) -> list[dict]:
    """Type-2: 동일 step이 다른 belief에선 acceptable.

    후보: type1의 reject_persona lose step. 다른 페르소나에서 acceptable인지
    (i) 다른 페르소나 입장의 cascade verifier 결과로 한 번 sanity-check,
    (ii) GPT-4o cross-belief check로 최종 확정.
    """
    persona_by_id = {p["id"]: p for p in personas}
    type2_pairs = []
    problem_count = defaultdict(int)

    for p in type1_pairs:
        if p["reject_type"] != "reject_persona":
            continue
        if problem_count[p["problem_id"]] >= max_per_problem:
            continue

        cur_persona = persona_by_id[p["persona_id"]]
        candidates = [
            other for other in personas
            if other["id"] != cur_persona["id"]
            and (other["grade_band"] != cur_persona["grade_band"]
                 or other["level"] != cur_persona["level"])
        ]

        for other_persona in candidates[:2]:
            # (i) verifier로 다른 페르소나 입장에서 sanity check
            other_res = verifier.verify_step(
                p["step_lose"], other_persona, prefix=p["prefix_steps"],
            )
            if other_res.verdict == "reject_persona":
                continue  # other에서도 위반이면 flip 후보 아님

            # (ii) GPT-4o cross-belief check로 최종 확정
            check = call_cross_belief_check(
                client=client, model=cross_belief_model,
                step_text=p["step_lose"],
                prefix_text="\n".join(p["prefix_steps"]),
                problem={"problem": p["problem"]},
                persona_a=cur_persona,
                persona_b=other_persona,
            )
            if (check.get("flip")
                    and not check.get("persona_a_acceptable")
                    and check.get("persona_b_acceptable")):
                type2_pairs.append({
                    "problem_id": p["problem_id"],
                    "problem": p["problem"],
                    "persona_id": p["persona_id"],
                    "persona_tag": p["persona_tag"],
                    "prefix_steps": p["prefix_steps"],
                    "step_win": p["step_win"],
                    "step_lose": p["step_lose"],
                    "pair_type": "belief_flip_pair",
                    "pair_subtype": "persona_first_error",
                    "reject_type": "reject_persona",
                    "evidence_code": (p.get("evidence_code")
                                      or check.get("curriculum_basis")),
                    "trigger_term": (p.get("trigger_term")
                                     or check.get("trigger_term")),
                    "verifier_stage": p.get("verifier_stage"),
                    "flip_persona_id": other_persona["id"],
                })
                problem_count[p["problem_id"]] += 1
                break

    return type2_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-model", required=True)
    parser.add_argument("--seed-problems", required=True)
    parser.add_argument("--personas-path", default="personas.json")
    parser.add_argument("--k-samples", type=int, default=8)
    parser.add_argument("--gpt-model", default="gpt-4o",
                        help="Stage C + 수학 step judge + cross-belief 모델")
    parser.add_argument("--math-judge-model", default=None,
                        help="비우면 --gpt-model 사용. e.g., gpt-4o-mini로 비용 추가 절감.")
    parser.add_argument("--output", default="data/preference_pairs.jsonl")

    # cascade verifier
    parser.add_argument("--verifier-base-url", default="http://localhost:8001/v1")
    parser.add_argument("--verifier-model",
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--verifier-api-key", default="EMPTY")
    parser.add_argument("--stage-b-threshold", type=float, default=0.85)
    parser.add_argument("--disable-stage-b", action="store_true")
    parser.add_argument("--disable-stage-c", action="store_true")
    parser.add_argument("--stage-log-path",
                        default="data/output/stage_log.jsonl")
    args = parser.parse_args()

    gpt_client = OpenAI()
    stage_b_client = None
    if not args.disable_stage_b:
        stage_b_client = OpenAI(base_url=args.verifier_base_url,
                                api_key=args.verifier_api_key)
    verifier = PersonaVerifier(
        stage_b_client=stage_b_client,
        stage_b_model=args.verifier_model,
        stage_c_client=gpt_client,
        stage_c_model=args.gpt_model,
        stage_b_conf_threshold=args.stage_b_threshold,
        enable_stage_b=not args.disable_stage_b,
        enable_stage_c=not args.disable_stage_c,
        stage_log_path=args.stage_log_path or None,
    )

    math_judge_model = args.math_judge_model or args.gpt_model

    llm = LLM(model=args.ref_model, dtype="bfloat16",
              gpu_memory_utilization=0.85)
    personas = load_personas(args.personas_path)

    problems = []
    with open(args.seed_problems, encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))

    if args.stage_log_path:
        Path(args.stage_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stage_log_path).write_text("")

    all_type1: list[dict] = []

    for i, problem in enumerate(problems):
        for persona in personas:
            verifier.problem_context = f"{problem.get('problem_id','?')}::{persona['id']}"
            sample_steps_list = sample_solutions(llm, problem, persona,
                                                 k=args.k_samples)
            judged = []
            for steps in sample_steps_list:
                if len(steps) < 2:
                    continue
                labels = label_steps(verifier, gpt_client, math_judge_model,
                                     problem, persona, steps)
                judged.append({"steps": steps, "labels": labels})
            t1 = build_type1_pairs(judged, problem, persona)
            all_type1.extend(t1)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(problems)}] type1: {len(all_type1)} "
                  f"cascade: {verifier.dump_counters()}")

    print("Building Type-2 belief-flip pairs...")
    all_type2 = build_type2_pairs(gpt_client, verifier, all_type1, personas,
                                  cross_belief_model=args.gpt_model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in all_type1 + all_type2:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Type-1 pairs: {len(all_type1)}")
    print(f"Type-2 pairs: {len(all_type2)}")
    print(f"Total: {len(all_type1) + len(all_type2)} → {args.output}")
    print(f"Final cascade counters: {verifier.dump_counters()}")


if __name__ == "__main__":
    main()
