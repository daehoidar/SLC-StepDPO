"""data_pipeline_stepdpo/3_locate_first_error.py

Step-DPO Stage 3: 최초 오류 스텝 검출.

흐름:
  1) π_ref(SFT 모델)로 각 problem에 대해 K개의 CoT 풀이 샘플링
  2) 최종 답이 ground_truth와 *다른* 궤적만 보존 (실패 궤적 수집)
  3) 각 실패 궤적에 대해 GPT-4o로 *최초로 잘못된 스텝의 인덱스*만 식별
     (Lai et al., 2024 — 모든 스텝을 라벨링하지 않고 "first incorrect" 한 점만)
  4) 결과를 located_errors.jsonl로 저장 (페어 구축은 4_build_pairs.py에서 진행)

출력 한 행:
  {
    "problem_id": "...", "problem": "...", "ground_truth": "...",
    "sampled_steps": ["Step 1: ...", ...],
    "first_error_idx": 3,
    "error_reason": "..."         # GPT-4o 200-word 이내 분석
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


LOCATE_PROMPT = """### Problem:
{problem}

### Ground-truth answer:
{ground_truth}

### Persona (belief b):
{persona_tag} — {persona_brief}

### Incorrect step-by-step answer (written under persona b):
{numbered_steps}

---

A math problem, its correct answer, and a persona condition are given above.
The step-by-step answer is incorrect either because (a) a step is
mathematically wrong, or (b) a step is mathematically OK but inappropriate
for the persona's vocabulary/level (persona drift). Output:

1. A short analysis (<=150 words) of where the reasoning first goes wrong.
2. The index of the first incorrect step, in the exact format:
   `First incorrect step: <N>`

If every step is correct (e.g., only the final boxed answer is wrong),
output `First incorrect step: -1`."""


def persona_brief(persona: dict) -> str:
    """LOCATE_PROMPT의 'persona_brief' 슬롯에 넣을 한두 줄짜리 요약."""
    parts = []
    if persona.get("grade_band"):
        parts.append(f"학년대 {persona['grade_band']}")
    if persona.get("level"):
        parts.append(f"수준 {persona['level']}")
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
    """매우 단순한 정답 매칭: 마지막 스텝 텍스트에 ground_truth substring이 있는지."""
    if not steps:
        return False
    tail = steps[-1].lower()
    return gt.strip().lower() in tail


def sample_solutions(
    llm: LLM, problem: str, k: int, persona_tag: str = "", temperature: float = 0.9,
) -> list[list[str]]:
    """persona_tag가 주어지면 prompt prefix로 포함 → SFT 모델이 b로 conditioning."""
    persona_prefix = f"{persona_tag}\n" if persona_tag else ""
    prompt = f"{persona_prefix}Problem: {problem}\nSolution:\n"
    sp = SamplingParams(temperature=temperature, max_tokens=800, n=k,
                        stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    return [parse_steps(o.text) for o in outputs[0].outputs]


def locate_error(
    client: OpenAI, problem: dict, steps: list[str], persona: dict,
    model: str = "gpt-4o",
) -> dict | None:
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
            raise  # 인증 에러는 재시도해도 무의미 → 즉시 중단
        except Exception as e:
            print(f"[locate retry {attempt+1}] {e}")
            time.sleep(2 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-model", required=True, help="SFT-trained π_ref 경로")
    ap.add_argument("--seed-problems", required=True)
    ap.add_argument("--personas-path", default=str(REPO_ROOT / "personas.json"),
                    help="persona_id → tag/level/grade_band 매핑")
    ap.add_argument("--k-samples", type=int, default=8)
    ap.add_argument("--gpt-model", default="gpt-4o")
    ap.add_argument("--output", default="data_pipeline_stepdpo/output/located_errors.jsonl")
    ap.add_argument("--max-rows", type=int, default=0,
                    help="0이면 전체. 디버그용 (problem, persona) 행 수 제한.")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    llm = LLM(model=args.ref_model, dtype="bfloat16", gpu_memory_utilization=0.85)

    persona_by_id = {p["id"]: p for p in load_personas(args.personas_path)}
    print(f"[load] {len(persona_by_id)} personas")

    # seed_problems.jsonl은 (problem × persona) 단위 행. dedup하지 말고 그대로 사용.
    rows: list[dict] = []
    with open(args.seed_problems, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[load] {len(rows)} (problem, persona) rows")

    n_failed_traj = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, prob in enumerate(rows):
            persona = persona_by_id.get(prob.get("persona", ""))
            if persona is None:
                continue  # personas.json에 없는 id면 skip
            sampled = sample_solutions(
                llm, prob["question"], k=args.k_samples,
                persona_tag=persona.get("tag", ""),
            )
            for sample_idx, steps in enumerate(sampled):
                if len(steps) < 2:
                    continue
                if final_answer_correct(steps, prob["gt_answer"]):
                    continue  # 성공 궤적은 Step-DPO 신호 없음
                located = locate_error(client, prob, steps, persona, model=args.gpt_model)
                if located is None or located["first_error_idx"] < 1:
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
                    "first_error_idx": located["first_error_idx"],
                    "error_reason": located["error_reason"],
                }, ensure_ascii=False) + "\n")
                n_failed_traj += 1
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(rows)}] located failed trajectories: {n_failed_traj}")

    print(f"Done. failed-trajectory rows: {n_failed_traj} → {out_path}")


if __name__ == "__main__":
    main()
