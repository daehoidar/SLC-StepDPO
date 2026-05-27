"""data_pipeline_stepdpo/3_locate_first_error.py

Step-DPO Stage 3: 최초 오류 스텝 검출 + 페르소나 cascade 검증.

흐름 (수정판):
  1) π_ref(SFT 모델)로 각 problem×persona에 대해 K개의 CoT 풀이 샘플링.
  2) 각 샘플에 대해:
     (a) PersonaVerifier(cascade A→B→C)로 step 1번부터 훑음.
         첫 reject_persona step이 있으면:
              first_error_idx = 그 step, error_type = "persona".
              → GPT-4o math-locate 호출 SKIP.
     (b) 페르소나 검증 모두 통과 + 최종 답이 ground_truth와 다름:
              기존 GPT-4o `locate_error`로 수학 최초 오류 인덱스 검출,
              error_type = "math".
     (c) 둘 다 아니면 → skip (완전 정답 궤적).
  3) located_errors.jsonl 한 행에 error_type, evidence_code, verifier_stage,
     trigger_term을 함께 저장 → 4_build_pairs.py 및 논문 ablation에 사용.

논문 산출물:
  - stage_log.jsonl  (cascade funnel; --stage-log-path)
  - error_type 분포  (persona vs math)
  - evidence_code per pair  (특허 청구항 핵심 지표)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import openai  # noqa: E402
from openai import OpenAI  # noqa: E402

try:
    from vllm import LLM, SamplingParams  # type: ignore
    _VLLM = True
except ImportError:
    from inference_backend import (  # type: ignore
        TransformersLLM as LLM,
        TransformersSamplingParams as SamplingParams,
    )
    _VLLM = False

from utils import parse_steps, load_personas  # noqa: E402
from persona_verifier import PersonaVerifier, first_persona_violation_index  # noqa: E402


LOCATE_PROMPT = """### Problem:
{problem}

### Ground-truth answer:
{ground_truth}

### Persona (belief b):
{persona_tag} — {persona_brief}

### Incorrect step-by-step answer (written under persona b):
{numbered_steps}

---

The step-by-step answer above is MATHEMATICALLY incorrect (persona compliance
has already been verified separately). Identify the first step where the
mathematics first goes wrong.

Output:
1. A short analysis (<=150 words) of the first math error.
2. The index of the first incorrect step, in the exact format:
   `First incorrect step: <N>`

If every step is mathematically correct (only the final boxed answer is wrong),
output `First incorrect step: -1`."""


def persona_brief(persona: dict) -> str:
    parts = []
    if persona.get("grade_band"):
        parts.append(f"grade {persona['grade_band']}")
    if persona.get("level"):
        parts.append(f"level {persona['level']}")
    if persona.get("vocabulary_guide"):
        vg = persona["vocabulary_guide"]
        if isinstance(vg, str):
            parts.append(f"vocab: {vg[:120]}")
    return "; ".join(parts) if parts else "(no brief available)"


def numbered(steps: list[str]) -> str:
    return "\n".join(f"[{i+1}] {s}" for i, s in enumerate(steps))


def parse_first_error_idx(text: str) -> tuple[int, str]:
    import re
    m = re.search(r"First incorrect step:\s*(-?\d+)", text)
    idx = int(m.group(1)) if m else -1
    return idx, text.strip()


def final_answer_correct(steps: list[str], gt: str) -> bool:
    if not steps:
        return False
    tail = steps[-1].lower()
    return gt.strip().lower() in tail


def sample_solutions(
    llm: LLM, problem: str, k: int, persona_tag: str = "", temperature: float = 0.9,
) -> list[list[str]]:
    persona_prefix = f"{persona_tag}\n" if persona_tag else ""
    prompt = f"{persona_prefix}Problem: {problem}\nSolution:\n"
    sp = SamplingParams(temperature=temperature, max_tokens=800, n=k,
                        stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    return [parse_steps(o.text) for o in outputs[0].outputs]


def locate_math_error(
    client: OpenAI, problem: dict, steps: list[str], persona: dict,
    model: str = "gpt-4o",
) -> dict | None:
    """기존 GPT-4o math-locate. 페르소나 검증은 이미 통과한 상태로 호출."""
    user_prompt = LOCATE_PROMPT.format(
        problem=problem["question"],
        ground_truth=problem["gt_answer"],
        persona_tag=persona.get("tag", ""),
        persona_brief=persona_brief(persona),
        numbered_steps=numbered(steps),
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            text = resp.choices[0].message.content
            idx, reason = parse_first_error_idx(text)
            return {"first_error_idx": idx, "error_reason": reason}
        except openai.AuthenticationError as e:
            print(f"[Fatal Error] API Key / Auth failed: {e}")
            raise
        except Exception as e:
            print(f"[locate retry {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-model", required=True, help="SFT-trained π_ref 경로")
    ap.add_argument("--seed-problems", required=True)
    ap.add_argument("--personas-path", default=str(REPO_ROOT / "personas.json"))
    ap.add_argument("--k-samples", type=int, default=8)
    ap.add_argument("--gpt-model", default="gpt-4o",
                    help="Stage C(외부 judge) + 수학 locate에 사용할 GPT 모델")
    ap.add_argument("--output", default="data_pipeline_stepdpo/output/located_errors.jsonl")
    ap.add_argument("--max-rows", type=int, default=0)

    # ── Persona cascade verifier (Stage B / C) ───────────────────────
    ap.add_argument("--verifier-base-url", default="http://localhost:8001/v1",
                    help="Stage B용 OpenAI-호환 endpoint "
                         "(예: vllm serve Llama-3.1-8B-Instruct --port 8001)")
    ap.add_argument("--verifier-model",
                    default="meta-llama/Llama-3.1-8B-Instruct",
                    help="Stage B 모델 ID. policy(π_ref)와 *다른 family* 강력 권장.")
    ap.add_argument("--verifier-api-key", default="EMPTY")
    ap.add_argument("--stage-b-threshold", type=float, default=0.85)
    ap.add_argument("--disable-stage-b", action="store_true",
                    help="Stage B를 끄고 A→C로만 직행 (ablation용)")
    ap.add_argument("--disable-stage-c", action="store_true",
                    help="Stage C를 끄고 borderline은 Stage B 결과 사용 (ablation용)")
    ap.add_argument("--stage-log-path",
                    default="data_pipeline_stepdpo/output/stage_log.jsonl",
                    help="cascade funnel 분석용 jsonl. 빈 문자열이면 비활성.")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 외부 GPT-4o (Stage C + math locate)
    gpt_client = OpenAI()

    # Stage B (local Llama-3.1 등) — OpenAI 호환 endpoint
    stage_b_client = None
    if not args.disable_stage_b:
        stage_b_client = OpenAI(
            base_url=args.verifier_base_url,
            api_key=args.verifier_api_key,
        )

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

    # Policy 모델 (sampling 전용)
    llm = LLM(model=args.ref_model, dtype="bfloat16", gpu_memory_utilization=0.85)

    persona_by_id = {p["id"]: p for p in load_personas(args.personas_path)}
    print(f"[load] {len(persona_by_id)} personas")

    rows: list[dict] = []
    with open(args.seed_problems, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[load] {len(rows)} (problem, persona) rows")

    # stage log 파일은 cascade 호출마다 append되므로 헤더 없이 fresh로 시작
    if args.stage_log_path:
        Path(args.stage_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stage_log_path).write_text("")

    n_persona = n_math = n_skip = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, prob in enumerate(rows):
            persona = persona_by_id.get(prob.get("persona", ""))
            if persona is None:
                continue
            verifier.problem_context = f"{prob.get('problem_id','?')}::{persona['id']}"
            sampled = sample_solutions(
                llm, prob["question"], k=args.k_samples,
                persona_tag=persona.get("tag", ""),
            )
            for sample_idx, steps in enumerate(sampled):
                if len(steps) < 2:
                    continue

                # (a) Persona cascade — 첫 reject_persona step 찾기
                p_idx, p_res = first_persona_violation_index(verifier, steps, persona)

                if p_idx is not None:
                    located = {
                        "first_error_idx": p_idx,
                        "error_reason": p_res.reasoning,
                        "error_type": "persona",
                        "evidence_code": p_res.evidence_code,
                        "trigger_term": p_res.trigger_term,
                        "verifier_stage": p_res.stage,
                        "verifier_confidence": p_res.confidence,
                    }
                    n_persona += 1
                # (b) Persona OK + 수학 오답
                elif not final_answer_correct(steps, prob["gt_answer"]):
                    math_loc = locate_math_error(gpt_client, prob, steps, persona,
                                                 model=args.gpt_model)
                    if math_loc is None or math_loc["first_error_idx"] < 1:
                        n_skip += 1
                        continue
                    located = {
                        "first_error_idx": math_loc["first_error_idx"],
                        "error_reason": math_loc["error_reason"],
                        "error_type": "math",
                        "evidence_code": None,
                        "trigger_term": None,
                        "verifier_stage": None,
                        "verifier_confidence": None,
                    }
                    n_math += 1
                else:
                    n_skip += 1
                    continue

                fout.write(json.dumps({
                    "problem_id": prob["problem_id"],
                    "problem": prob["question"],
                    "ground_truth": prob["gt_answer"],
                    "ground_truth_solution": prob.get("gt_answer_raw", ""),
                    "persona_id": persona["id"],
                    "persona_tag": persona.get("tag", ""),
                    "sampled_steps": steps,
                    "sample_idx": sample_idx,
                    **located,
                }, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                c = verifier.dump_counters()
                print(f"[{i+1}/{len(rows)}] persona={n_persona} math={n_math} "
                      f"skip={n_skip} | cascade {c}")

    print(f"Done. persona={n_persona} math={n_math} skip={n_skip} → {out_path}")
    print(f"Cascade counters: {verifier.dump_counters()}")


if __name__ == "__main__":
    main()
