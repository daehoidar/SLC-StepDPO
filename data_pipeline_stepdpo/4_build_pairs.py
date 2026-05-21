"""data_pipeline_stepdpo/4_build_pairs.py

Step-DPO Stage 4 (pair 빌드 단계):

3_locate_first_error.py가 산출한 located_errors.jsonl을 읽어,
각 실패 궤적마다 **단 하나의 (chosen, rejected) 페어**를 구성한다.

  - prefix_steps  := sampled_steps[: first_error_idx - 1]
  - rejected_step := sampled_steps[first_error_idx - 1]
  - chosen_step   := GPT-4o가 prefix 위에서 생성한 *올바른 다음 스텝*

Lai et al. (2024) Step-DPO와 정확히 같은 구조. 한 (problem, sample) → 한 pair.

출력 한 행:
  {
    "problem_id": "...", "problem": "...", "ground_truth": "...",
    "prefix_steps": [...],
    "chosen_step": "...",
    "rejected_step": "...",
    "sample_idx": 3,
    "error_reason": "..."
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

import openai  # noqa: E402  (예외 타입용)
from openai import OpenAI  # noqa: E402


RECTIFY_PROMPT = """You are a careful math tutor who matches the persona b below.

Persona (belief b): {persona_tag}
The corrected step MUST be appropriate for this persona's vocabulary and level.

Problem:
{problem}

Ground-truth final answer: {ground_truth}

A student wrote the following correct steps so far:
{prefix}

The student then wrote this INCORRECT next step:
{wrong_step}

Why it is wrong: {reason}

Write the SINGLE next step the student should have written instead.
Format: start with "Step {next_idx}: " and write exactly one step.
Do not write the rest of the solution — only this one corrected step."""


def make_chosen_step(
    client: OpenAI, problem: str, ground_truth: str, prefix_steps: list[str],
    wrong_step: str, reason: str, next_idx: int, persona_tag: str = "",
    model: str = "gpt-4o",
) -> str | None:
    prefix_text = "\n".join(prefix_steps) if prefix_steps else "(no prior steps)"
    prompt = RECTIFY_PROMPT.format(
        persona_tag=persona_tag or "(none)",
        problem=problem,
        ground_truth=ground_truth,
        prefix=prefix_text,
        wrong_step=wrong_step,
        reason=reason,
        next_idx=next_idx,
    )
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
            return text.splitlines()[0].strip()  # 첫 줄(한 스텝)만 보존
        except openai.AuthenticationError as e:
            print(f"[Fatal Error] API Key / Auth failed: {e}")
            raise  # 인증 에러는 재시도해도 무의미 → 즉시 중단
        except Exception as e:
            print(f"[rectify retry {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--located", required=True,
                    help="3_locate_first_error.py 산출 located_errors.jsonl")
    ap.add_argument("--gpt-model", default="gpt-4o")
    ap.add_argument("--output", default="data_pipeline_stepdpo/output/pairs_stepdpo.jsonl")
    args = ap.parse_args()

    client = OpenAI()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_pairs = 0
    n_skip = 0
    with open(args.located, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            steps = row["sampled_steps"]
            err_idx = row["first_error_idx"]  # 1-based
            if err_idx < 1 or err_idx > len(steps):
                n_skip += 1
                continue
            prefix = steps[: err_idx - 1]
            wrong = steps[err_idx - 1]
            chosen = make_chosen_step(
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
            if chosen is None or chosen == wrong:
                n_skip += 1
                continue
            # BC-StepDPO 학습용 스키마 (Proposition 2):
            #   (x=problem, b=persona_id/tag, s_{1:k-1}=prefix_steps, s_w=step_win, s_l=step_lose)
            # 필드명을 기존 3_build_pairs.py의 step_pair 스키마와 정렬한다.
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
                "reject_type": "n/a",
                "flip_persona_id": None,
                "sample_idx": row.get("sample_idx", -1),
                "error_reason": row.get("error_reason", ""),
            }, ensure_ascii=False) + "\n")
            n_pairs += 1
            if n_pairs % 100 == 0:
                print(f"[built {n_pairs} pairs] (skipped {n_skip})")

    print(f"Done. {n_pairs} Step-DPO pairs (skipped {n_skip}) → {out_path}")


if __name__ == "__main__":
    main()
