"""data_pipeline_stepdpo/4_build_pairs.py

Step-DPO Stage 4: 페어 빌드.

3_locate_first_error.py 산출의 each row에 대해 한 개의 (chosen, rejected)
페어를 만든다. `error_type ∈ {persona, math}`에 따라 rectify prompt를 분기:

  - error_type == "persona": 페르소나 친화 prompt — forbidden term/근거 코드를
    명시하고, 페르소나가 허용하는 어휘로만 다음 step을 작성하도록 요청.
  - error_type == "math":    수학 교정 prompt (기존 동작).

출력 한 행:
  {
    "problem_id", "problem", "ground_truth",
    "persona_id", "persona_tag",
    "prefix_steps": [...],
    "step_win":  "Step k: ... (rectified)",
    "step_lose": "Step k: ... (sampled wrong step)",
    "pair_type": "step_pair",
    "pair_subtype": "math_first_error" | "persona_first_error",
    "reject_type": "reject_math" | "reject_persona",
    "evidence_code": "[6수01-08]" | null,
    "trigger_term": str | null,
    "verifier_stage": "A"|"B"|"C" | null,
    "flip_persona_id": null,
    "sample_idx", "error_reason"
  }
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

from utils import load_personas  # noqa: E402


RECTIFY_PROMPT_MATH = """You are a careful math tutor who matches the persona b below.

Persona (belief b): {persona_tag}
The corrected step MUST be appropriate for this persona's vocabulary and level.

Problem:
{problem}

Ground-truth final answer: {ground_truth}

A student wrote the following correct steps so far:
{prefix}

The student then wrote this INCORRECT next step:
{wrong_step}

Why it is wrong (mathematically): {reason}

Write the SINGLE next step the student should have written instead.
Format: start with "Step {next_idx}: " and write exactly one step.
Do not write the rest of the solution — only this one corrected step."""


RECTIFY_PROMPT_PERSONA = """You are a careful math tutor writing for the persona b below.

Persona (belief b): {persona_tag}
Grade band: {grade_band}
Level: {level}
Vocabulary guide: {vocabulary_guide}

FORBIDDEN terms for this persona (introduced AFTER this grade per 2022 Korean
math curriculum):
{forbidden_block}

PREFERRED terms / phrasing:
{preferred_block}

Problem:
{problem}

Ground-truth final answer: {ground_truth}

The student wrote the following steps so far (mathematically valid):
{prefix}

The student then wrote this PERSONA-INAPPROPRIATE next step:
{wrong_step}

Why it is inappropriate: {reason}
Trigger term (must NOT appear): "{trigger_term}"
Curriculum evidence: {evidence_code}

Rewrite the SAME mathematical operation as a SINGLE next step, but using ONLY
vocabulary appropriate to this persona's grade (and avoiding the trigger term).
Format: start with "Step {next_idx}: " and write exactly one step.
Do not write the rest of the solution — only this one corrected step."""


def _format_forbidden(persona: dict) -> str:
    ev = persona.get("term_evidence", {})
    lines = []
    for t in persona.get("forbidden_terms", []):
        x = ev.get(t, {}) or {}
        lines.append(f"  - {t} ({x.get('first_introduced','?')} {x.get('source_code','?')})")
    return "\n".join(lines) if lines else "  (none)"


def _format_preferred(persona: dict) -> str:
    pts = persona.get("preferred_terms", [])
    return ", ".join(pts) if pts else "(none specified)"


def make_chosen_step_math(
    client: OpenAI, problem: str, ground_truth: str, prefix_steps: list[str],
    wrong_step: str, reason: str, next_idx: int, persona_tag: str = "",
    model: str = "gpt-4o",
) -> str | None:
    prefix_text = "\n".join(prefix_steps) if prefix_steps else "(no prior steps)"
    prompt = RECTIFY_PROMPT_MATH.format(
        persona_tag=persona_tag or "(none)",
        problem=problem, ground_truth=ground_truth,
        prefix=prefix_text, wrong_step=wrong_step, reason=reason,
        next_idx=next_idx,
    )
    return _call_rectify(client, prompt, model)


def make_chosen_step_persona(
    client: OpenAI, problem: str, ground_truth: str, prefix_steps: list[str],
    wrong_step: str, reason: str, next_idx: int, persona: dict,
    trigger_term: str | None, evidence_code: str | None,
    model: str = "gpt-4o",
) -> str | None:
    prefix_text = "\n".join(prefix_steps) if prefix_steps else "(no prior steps)"
    prompt = RECTIFY_PROMPT_PERSONA.format(
        persona_tag=persona.get("tag", ""),
        grade_band=persona.get("grade_band", ""),
        level=persona.get("level", ""),
        vocabulary_guide=persona.get("vocabulary_guide", ""),
        forbidden_block=_format_forbidden(persona),
        preferred_block=_format_preferred(persona),
        problem=problem, ground_truth=ground_truth,
        prefix=prefix_text, wrong_step=wrong_step, reason=reason,
        next_idx=next_idx,
        trigger_term=trigger_term or "(unspecified)",
        evidence_code=evidence_code or "(unspecified)",
    )
    return _call_rectify(client, prompt, model)


def _call_rectify(client: OpenAI, prompt: str, model: str) -> str | None:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200,
            )
            text = resp.choices[0].message.content.strip()
            if not text:
                continue
            return text.splitlines()[0].strip()
        except openai.AuthenticationError as e:
            print(f"[Fatal Error] API Key / Auth failed: {e}")
            raise
        except Exception as e:
            print(f"[rectify retry {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--located", required=True)
    ap.add_argument("--personas-path", default=str(REPO_ROOT / "personas.json"))
    ap.add_argument("--gpt-model", default="gpt-4o")
    ap.add_argument("--output", default="data_pipeline_stepdpo/output/pairs_stepdpo.jsonl")
    args = ap.parse_args()

    client = OpenAI()
    persona_by_id = {p["id"]: p for p in load_personas(args.personas_path)}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_persona = n_math = n_skip = 0
    with open(args.located, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            steps = row["sampled_steps"]
            err_idx = row["first_error_idx"]
            if err_idx < 1 or err_idx > len(steps):
                n_skip += 1
                continue
            prefix = steps[: err_idx - 1]
            wrong = steps[err_idx - 1]
            error_type = row.get("error_type", "math")
            persona = persona_by_id.get(row.get("persona_id", "")) or {}

            if error_type == "persona":
                chosen = make_chosen_step_persona(
                    client,
                    problem=row["problem"],
                    ground_truth=row["ground_truth"],
                    prefix_steps=prefix,
                    wrong_step=wrong,
                    reason=row.get("error_reason", "")[:500],
                    next_idx=err_idx,
                    persona=persona,
                    trigger_term=row.get("trigger_term"),
                    evidence_code=row.get("evidence_code"),
                    model=args.gpt_model,
                )
                reject_type = "reject_persona"
                pair_subtype = "persona_first_error"
            else:
                chosen = make_chosen_step_math(
                    client,
                    problem=row["problem"],
                    ground_truth=row["ground_truth"],
                    prefix_steps=prefix,
                    wrong_step=wrong,
                    reason=row.get("error_reason", "")[:500],
                    next_idx=err_idx,
                    persona_tag=row.get("persona_tag", ""),
                    model=args.gpt_model,
                )
                reject_type = "reject_math"
                pair_subtype = "math_first_error"

            if chosen is None or chosen == wrong:
                n_skip += 1
                continue

            fout.write(json.dumps({
                "problem_id": row["problem_id"],
                "problem": row["problem"],
                "ground_truth": row["ground_truth"],
                "persona_id": row.get("persona_id", ""),
                "persona_tag": row.get("persona_tag", ""),
                "prefix_steps": prefix,
                "step_win": chosen,
                "step_lose": wrong,
                "pair_type": "step_pair",
                "pair_subtype": pair_subtype,
                "reject_type": reject_type,
                "evidence_code": row.get("evidence_code"),
                "trigger_term": row.get("trigger_term"),
                "verifier_stage": row.get("verifier_stage"),
                "flip_persona_id": None,
                "sample_idx": row.get("sample_idx", -1),
                "error_reason": row.get("error_reason", ""),
            }, ensure_ascii=False) + "\n")

            if error_type == "persona":
                n_persona += 1
            else:
                n_math += 1
            if (n_persona + n_math) % 100 == 0:
                print(f"[built persona={n_persona} math={n_math}] (skipped {n_skip})")

    print(f"Done. persona-pairs={n_persona} math-pairs={n_math} "
          f"(skipped {n_skip}) → {out_path}")


if __name__ == "__main__":
    main()
