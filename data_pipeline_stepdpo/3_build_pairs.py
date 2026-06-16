"""
data/build_pairs.py

Stage 3 (Full-Step-DPO via GPT-4o judge, persona cascade 적용):

두 가지 운용 모드:
  (1) Shared 모드 (권장):
       --samples-path data_pipeline/output/samples_with_persona_labels.jsonl
       → π_ref 샘플링 + persona cascade SKIP
       → step별 수학 라벨만 GPT-4o single-step judge로 채움
       → Type-1 / Type-2 페어 구축

  (2) Standalone 모드 (기존):
       --ref-model 만 주면 π_ref로 자체 샘플링 + cascade 직접 호출.

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

from judge_prompts import (  # noqa: E402
    CROSS_BELIEF_CHECK_SYSTEM, CROSS_BELIEF_CHECK_USER_TEMPLATE,
    build_cross_belief_kwargs,
)
from utils import load_personas, parse_steps  # noqa: E402
from openai_client import make_openai_client  # noqa: E402
from persona_verifier import PersonaVerifier  # noqa: E402


# ── 수학 single-step judge (페르소나 정보 제외, 짧게) ────────────────────

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


# ──────────────────────────── 페어 구축 (모드 무관) ─────────────────────

def build_type1_pairs(judged_samples: list[dict], problem: dict,
                      persona: dict) -> list[dict]:
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
    client: OpenAI, verifier: PersonaVerifier | None,
    type1_pairs: list[dict], personas: list[dict],
    cross_belief_model: str = "gpt-4o",
    max_per_problem: int = 3,
) -> list[dict]:
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
            if verifier is not None:
                other_res = verifier.verify_step(
                    p["step_lose"], other_persona, prefix=p["prefix_steps"],
                )
                if other_res.verdict == "reject_persona":
                    continue

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


# ──────────────────────────── Shared 모드 ────────────────────────────────

def load_shared_samples(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def label_steps_from_cache(
    persona_labels: list[dict], steps: list[str],
    math_judge_client: OpenAI, math_judge_model: str, problem: dict,
) -> list[dict]:
    """캐시된 persona 라벨 + GPT-4o 수학 single-step judge → step 통합 라벨."""
    out = []
    for i, step in enumerate(steps):
        p_lab = persona_labels[i] if i < len(persona_labels) else {}
        if p_lab.get("verdict") == "reject_persona":
            out.append({
                "index": i + 1,
                "label": "reject_persona",
                "persona_stage": p_lab.get("stage"),
                "evidence_code": p_lab.get("evidence_code"),
                "trigger_term": p_lab.get("trigger_term"),
                "reason": p_lab.get("reasoning", ""),
            })
            continue
        # 페르소나 OK → 수학 검증
        m = judge_math_step(math_judge_client, math_judge_model, problem,
                            steps[:i], step)
        if m.get("is_wrong"):
            out.append({
                "index": i + 1,
                "label": "reject_math",
                "persona_stage": p_lab.get("stage"),
                "evidence_code": None,
                "trigger_term": None,
                "reason": m.get("reason", ""),
            })
        else:
            out.append({
                "index": i + 1,
                "label": "acceptable",
                "persona_stage": p_lab.get("stage"),
                "evidence_code": None,
                "trigger_term": None,
                "reason": "",
            })
    return out


def label_steps_standalone(
    verifier: PersonaVerifier, math_judge_client: OpenAI, math_judge_model: str,
    problem: dict, persona: dict, steps: list[str],
) -> list[dict]:
    out = []
    for i, step in enumerate(steps):
        p = verifier.verify_step(step, persona, prefix=steps[:i])
        if p.verdict == "reject_persona":
            out.append({
                "index": i + 1, "label": "reject_persona",
                "persona_stage": p.stage,
                "evidence_code": p.evidence_code,
                "trigger_term": p.trigger_term,
                "reason": p.reasoning,
            })
            continue
        m = judge_math_step(math_judge_client, math_judge_model, problem,
                            steps[:i], step)
        if m.get("is_wrong"):
            out.append({
                "index": i + 1, "label": "reject_math",
                "persona_stage": p.stage,
                "evidence_code": None, "trigger_term": None,
                "reason": m.get("reason", ""),
            })
        else:
            out.append({
                "index": i + 1, "label": "acceptable",
                "persona_stage": p.stage,
                "evidence_code": None, "trigger_term": None,
                "reason": "",
            })
    return out


def shared_mode(args, gpt_client, persona_by_id, personas, fout_path):
    shared = load_shared_samples(Path(args.samples_path))
    print(f"[shared] {len(shared)} samples loaded")
    math_judge_model = args.math_judge_model or args.gpt_model

    # (problem_id, persona_id) 단위로 그룹화
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in shared:
        grouped[(s["problem_id"], s["persona_id"])].append(s)

    all_type1: list[dict] = []
    for key, samples in grouped.items():
        problem_id, persona_id = key
        persona = persona_by_id.get(persona_id)
        if persona is None:
            continue
        problem = {"problem_id": problem_id,
                   "problem": samples[0]["problem"],
                   "ground_truth": samples[0]["ground_truth"]}
        judged = []
        for s in samples:
            steps = s.get("steps", [])
            labels = s.get("step_persona_labels", [])
            if len(steps) < 2 or len(labels) != len(steps):
                continue
            step_labels = label_steps_from_cache(
                labels, steps, gpt_client, math_judge_model, problem,
            )
            judged.append({"steps": steps, "labels": step_labels})

        t1 = build_type1_pairs(judged, problem, persona)
        all_type1.extend(t1)

    print(f"[shared] Type-1 pairs: {len(all_type1)}")
    print("Building Type-2 belief-flip pairs...")
    # Shared 모드에서는 cross-belief verifier sanity check를 위해 verifier 필요할 수 있음.
    # 비용 절감 위해 verifier=None으로 호출 (GPT-4o cross-belief check만 사용).
    all_type2 = build_type2_pairs(
        gpt_client, verifier=None, type1_pairs=all_type1, personas=personas,
        cross_belief_model=args.gpt_model,
    )
    print(f"[shared] Type-2 pairs: {len(all_type2)}")

    with open(fout_path, "w", encoding="utf-8") as f:
        for p in all_type1 + all_type2:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Done. → {fout_path}")


# ──────────────────────────── Standalone 모드 ────────────────────────────

def standalone_mode(args, gpt_client, personas, fout_path):
    try:
        from vllm import LLM, SamplingParams  # type: ignore  # noqa: E402
    except ImportError:
        from inference_backend import (  # noqa: E402
            TransformersLLM as LLM,
            TransformersSamplingParams as SamplingParams,
        )

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
    if args.stage_log_path:
        Path(args.stage_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stage_log_path).write_text("")

    math_judge_model = args.math_judge_model or args.gpt_model
    llm = LLM(model=args.ref_model, dtype="bfloat16",
              gpu_memory_utilization=0.85)

    problems = []
    with open(args.seed_problems, encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))

    all_type1: list[dict] = []
    for i, problem in enumerate(problems):
        for persona in personas:
            verifier.problem_context = f"{problem.get('problem_id','?')}::{persona['id']}"
            persona_tag = persona.get("tag", "")
            persona_prefix = f"{persona_tag}\n" if persona_tag else ""
            prompt = f"{persona_prefix}Problem: {problem['problem']}\nSolution:\n"
            sp = SamplingParams(temperature=0.9, max_tokens=800,
                                n=args.k_samples,
                                stop=["Problem:", "\n\n\n"])
            outputs = llm.generate([prompt], sp)
            sample_steps_list = [parse_steps(o.text) for o in outputs[0].outputs]

            judged = []
            for steps in sample_steps_list:
                if len(steps) < 2:
                    continue
                labels = label_steps_standalone(
                    verifier, gpt_client, math_judge_model,
                    problem, persona, steps,
                )
                judged.append({"steps": steps, "labels": labels})

            t1 = build_type1_pairs(judged, problem, persona)
            all_type1.extend(t1)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(problems)}] type1: {len(all_type1)} "
                  f"cascade: {verifier.dump_counters()}")

    print("Building Type-2 belief-flip pairs...")
    all_type2 = build_type2_pairs(gpt_client, verifier, all_type1, personas,
                                  cross_belief_model=args.gpt_model)

    with open(fout_path, "w", encoding="utf-8") as f:
        for p in all_type1 + all_type2:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Type-1: {len(all_type1)}, Type-2: {len(all_type2)} → {fout_path}")
    print(f"Final cascade counters: {verifier.dump_counters()}")


def main():
    parser = argparse.ArgumentParser()
    # Standalone 모드용 (samples-path가 없을 때만)
    parser.add_argument("--ref-model", default=None)
    parser.add_argument("--seed-problems", default=None)
    parser.add_argument("--k-samples", type=int, default=8)

    parser.add_argument("--personas-path", default="personas.json")
    parser.add_argument("--gpt-model", default="gpt-4o")
    parser.add_argument("--math-judge-model", default=None,
                        help="비우면 --gpt-model 사용")
    parser.add_argument("--output", default="data/preference_pairs.jsonl")

    # Shared 모드
    parser.add_argument("--samples-path", default=None,
                        help="shared_sampling.py 산출 jsonl 경로")

    # cascade (standalone 모드 전용)
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

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    gpt_client = make_openai_client()
    personas = load_personas(args.personas_path)
    persona_by_id = {p["id"]: p for p in personas}

    if args.samples_path:
        print("[mode] SHARED — reusing samples + persona labels")
        shared_mode(args, gpt_client, persona_by_id, personas, args.output)
    else:
        print("[mode] STANDALONE — sampling π_ref + cascade verify in-process")
        if not args.ref_model or not args.seed_problems:
            raise SystemExit(
                "Standalone 모드는 --ref-model + --seed-problems 필요."
                " 또는 --samples-path 로 shared 산출물을 넘기세요."
            )
        standalone_mode(args, gpt_client, personas, args.output)


if __name__ == "__main__":
    main()
