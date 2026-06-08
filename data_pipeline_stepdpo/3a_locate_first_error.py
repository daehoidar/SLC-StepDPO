"""data_pipeline_stepdpo/3a_locate_first_error.py

Step-DPO Stage 3a: 최초 오류 스텝 검출 + 페르소나 cascade 검증.

두 가지 운용 모드:
  (1) Shared 모드 (권장):
       --samples-path data_pipeline/output/samples_with_persona_labels.jsonl
       → π_ref 샘플링 + cascade 검증을 SKIP (shared_sampling.py 결과 재사용)
       → 첫 reject_persona step은 캐시된 라벨에서 즉시 찾음
       → 페르소나 OK + 수학 오답인 경우만 GPT-4o math-locate

  (2) Standalone 모드 (기존):
       --ref-model 만 주면 π_ref로 자체 K-샘플링 + cascade 직접 호출.

출력 한 행 → 3b_build_pairs.py로:
  {
    "problem_id", "problem", "ground_truth",
    "persona_id", "persona_tag",
    "sampled_steps": [...],
    "sample_idx",
    "first_error_idx", "error_reason",
    "error_type": "persona" | "math",
    "evidence_code", "trigger_term", "verifier_stage", "verifier_confidence"
  }
"""
from __future__ import annotations
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


def locate_math_error(
    client: OpenAI, problem_text: str, ground_truth: str, steps: list[str],
    persona: dict, model: str = "gpt-4o",
) -> dict | None:
    user_prompt = LOCATE_PROMPT.format(
        problem=problem_text,
        ground_truth=ground_truth,
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


def first_persona_violation_from_cache(step_labels: list[dict]) -> tuple[int | None, dict | None]:
    """캐시된 라벨에서 첫 reject_persona step (1-based) 반환."""
    for i, lab in enumerate(step_labels):
        if lab.get("verdict") == "reject_persona":
            return i + 1, lab
    return None, None


# ──────────────────────────── Standalone 모드 (fallback) ────────────────

def standalone_main(args, gpt_client, verifier, persona_by_id, rows, fout):
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except ImportError:
        from inference_backend import (  # type: ignore
            TransformersLLM as LLM, TransformersSamplingParams as SamplingParams,
        )
    llm = LLM(model=args.ref_model, dtype="bfloat16",
              gpu_memory_utilization=0.85)

    n_persona = n_math = n_skip = 0
    for i, prob in enumerate(rows):
        persona = persona_by_id.get(prob.get("persona", ""))
        if persona is None:
            continue
        verifier.problem_context = f"{prob.get('problem_id','?')}::{persona['id']}"

        persona_tag = persona.get("tag", "")
        persona_prefix = f"{persona_tag}\n" if persona_tag else ""
        prompt = f"{persona_prefix}Problem: {prob['question']}\nSolution:\n"
        sp = SamplingParams(temperature=0.9, max_tokens=800, n=args.k_samples,
                            stop=["Problem:", "\n\n\n"])
        outputs = llm.generate([prompt], sp)
        sampled = [parse_steps(o.text) for o in outputs[0].outputs]

        for sample_idx, steps in enumerate(sampled):
            if len(steps) < 2:
                continue
            n_persona, n_math, n_skip = _process_sample(
                prob, persona, steps, sample_idx, verifier, gpt_client,
                args.gpt_model, fout, n_persona, n_math, n_skip,
            )
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(rows)}] persona={n_persona} math={n_math} skip={n_skip}")

    print(f"Done. persona={n_persona} math={n_math} skip={n_skip}")
    print(f"Cascade counters: {verifier.dump_counters()}")


def _process_sample(prob, persona, steps, sample_idx, verifier, gpt_client,
                     gpt_model, fout, n_persona, n_math, n_skip):
    """Standalone 모드 헬퍼: cascade로 검증 후 결과 기록."""
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
    elif not final_answer_correct(steps, prob["gt_answer"]):
        math_loc = locate_math_error(gpt_client, prob["question"],
                                      prob["gt_answer"], steps, persona,
                                      model=gpt_model)
        if math_loc is None or math_loc["first_error_idx"] < 1:
            return n_persona, n_math, n_skip + 1
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
        return n_persona, n_math, n_skip + 1

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
    return n_persona, n_math, n_skip


# ──────────────────────────── Shared 모드 메인 ───────────────────────────

def shared_main(args, gpt_client, persona_by_id, fout):
    shared = load_shared_samples(Path(args.samples_path))
    print(f"[shared] {len(shared)} samples loaded from {args.samples_path}")

    # (problem_id, persona_id) → seed 정보를 위해 seed_problems도 로드
    seed_lookup: dict[tuple[str, str], dict] = {}
    if args.seed_problems:
        with open(args.seed_problems, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                seed_lookup[(r["problem_id"], r["persona"])] = r

    n_persona = n_math = n_skip = 0
    for i, s in enumerate(shared):
        persona = persona_by_id.get(s.get("persona_id", ""))
        if persona is None:
            continue
        steps = s.get("steps", [])
        labels = s.get("step_persona_labels", [])
        if len(steps) < 2 or len(labels) != len(steps):
            n_skip += 1
            continue

        # 1) 캐시 라벨에서 첫 reject_persona 검색
        p_idx, p_lab = first_persona_violation_from_cache(labels)

        if p_idx is not None:
            located = {
                "first_error_idx": p_idx,
                "error_reason": p_lab.get("reasoning", ""),
                "error_type": "persona",
                "evidence_code": p_lab.get("evidence_code"),
                "trigger_term": p_lab.get("trigger_term"),
                "verifier_stage": p_lab.get("stage"),
                "verifier_confidence": p_lab.get("confidence"),
            }
            n_persona += 1
        elif not final_answer_correct(steps, s["ground_truth"]):
            # 2) 페르소나 OK + 수학 오답 → GPT-4o math-locate
            math_loc = locate_math_error(
                gpt_client, s["problem"], s["ground_truth"], steps, persona,
                model=args.gpt_model,
            )
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

        seed = seed_lookup.get((s["problem_id"], s["persona_id"]), {})
        fout.write(json.dumps({
            "problem_id": s["problem_id"],
            "problem": s["problem"],
            "ground_truth": s["ground_truth"],
            "ground_truth_solution": seed.get("gt_answer_raw", ""),
            "persona_id": s["persona_id"],
            "persona_tag": s["persona_tag"],
            "sampled_steps": steps,
            "sample_idx": s.get("sample_idx", -1),
            **located,
        }, ensure_ascii=False) + "\n")

        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(shared)}] persona={n_persona} math={n_math} skip={n_skip}")

    print(f"Done. persona={n_persona} math={n_math} skip={n_skip}")


def main():
    ap = argparse.ArgumentParser()
    # Standalone 모드용 (samples-path가 없을 때만 사용)
    ap.add_argument("--ref-model", default=None,
                    help="Standalone 모드용 SFT π_ref 경로 "
                         "(--samples-path 사용 시 불필요)")
    ap.add_argument("--k-samples", type=int, default=8,
                    help="Standalone 모드 K (shared 모드에선 무시)")

    ap.add_argument("--seed-problems", default=None,
                    help="standalone 모드에 필요; shared 모드는 ground_truth_raw 채울 때만 사용")
    ap.add_argument("--personas-path",
                    default=str(REPO_ROOT / "personas.json"))
    ap.add_argument("--output",
                    default="data_pipeline_stepdpo/output/located_errors.jsonl")
    ap.add_argument("--max-rows", type=int, default=0)

    # Shared 모드 (권장)
    ap.add_argument("--samples-path", default=None,
                    help="shared_sampling.py 산출 jsonl. 주어지면 sampling/verify SKIP.")

    # GPT-4o / cascade
    ap.add_argument("--gpt-model", default="gpt-4o",
                    help="Stage C + 수학 locate에 사용")
    ap.add_argument("--verifier-base-url", default="http://localhost:8001/v1")
    ap.add_argument("--verifier-model",
                    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--verifier-api-key", default="EMPTY")
    ap.add_argument("--stage-b-threshold", type=float, default=0.85)
    ap.add_argument("--disable-stage-b", action="store_true")
    ap.add_argument("--disable-stage-c", action="store_true")
    ap.add_argument("--stage-log-path",
                    default="data_pipeline_stepdpo/output/stage_log.jsonl")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gpt_client = OpenAI()
    persona_by_id = {p["id"]: p for p in load_personas(args.personas_path)}

    with open(out_path, "w", encoding="utf-8") as fout:
        if args.samples_path:
            print("[mode] SHARED — reusing samples + persona labels")
            shared_main(args, gpt_client, persona_by_id, fout)
        else:
            print("[mode] STANDALONE — sampling π_ref + cascade verify in-process")
            if not args.ref_model or not args.seed_problems:
                raise SystemExit(
                    "Standalone 모드는 --ref-model + --seed-problems 필요."
                    " 또는 --samples-path 로 shared 산출물을 넘기세요."
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

            rows = []
            with open(args.seed_problems, encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
            if args.max_rows:
                rows = rows[: args.max_rows]
            print(f"[load] {len(rows)} (problem, persona) rows")

            standalone_main(args, gpt_client, verifier, persona_by_id, rows, fout)


if __name__ == "__main__":
    main()
