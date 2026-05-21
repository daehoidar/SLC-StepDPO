"""data_pipeline_fullstepdpo/3a_mc_rollout_label.py

Full-Step DPO Stage 3a: Monte Carlo rollout으로 **각 스텝의 가치**를 자가 지도
라벨링한다 (Math-Shepherd / OmegaPRM 계열의 접근).

핵심 아이디어:
  체인 c = [s_1, s_2, ..., s_T]가 주어졌을 때, 스텝 s_i의 "value"는
  s_1...s_i를 prefix로 두고 π_ref로 M회 rollout했을 때 *최종 정답에 도달하는
  비율*로 정의한다. 외부 모델(GPT-4) 없이도 ground_truth만 있으면 자동 라벨이
  생성된다.

  step_value_i := (1/M) * sum_{m=1..M} 1[ final_answer(rollout_m) == gt ]

출력 한 행 = 한 (problem, sample, step_i):
  {
    "problem_id": "...", "problem": "...", "ground_truth": "18",
    "prefix_until_step":  ["Step 1: ...", "Step 2: ..."],
    "step_idx": 2,
    "step_value": 0.625,    # 8회 rollout 중 5회 정답
    "sample_idx": 3
  }

참고: 본 스크립트는 GPU 친화적 (vLLM의 batched generation 활용). API 비용 = 0.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from vllm import LLM, SamplingParams  # type: ignore
    _VLLM = True
except ImportError:
    from inference_backend import (  # type: ignore
        TransformersLLM as LLM,
        TransformersSamplingParams as SamplingParams,
    )
    _VLLM = False

from utils import parse_steps  # noqa: E402


# 간단 정답 매칭. 도메인별로 정교화 권장 (sympy.simplify 등).
ANS_RE = re.compile(r"(?:answer|Answer)\s*(?:is)?\s*[:=]?\s*([\-\d\./]+)")


def extract_answer(text: str) -> str:
    m = ANS_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    # fallback: 마지막 비공백 라인
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def answer_matches(pred: str, gt: str) -> bool:
    p, g = pred.strip().lower(), gt.strip().lower()
    if p == g:
        return True
    # 숫자 단순 비교
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return g in p or p in g


def build_prompt(problem: str, prefix_steps: list[str]) -> str:
    prefix_text = "\n".join(prefix_steps)
    if prefix_text:
        prefix_text += "\n"
    return f"Problem: {problem}\nSolution:\n{prefix_text}"


def mc_step_value(
    llm: LLM, problem: str, prefix_steps: list[str], gt: str, m_rollouts: int,
    temperature: float = 1.0,
) -> float:
    """prefix까지 fixed → M개의 continuation rollout → 정답 도달 비율 반환."""
    prompt = build_prompt(problem, prefix_steps)
    sp = SamplingParams(temperature=temperature, max_tokens=400,
                        n=m_rollouts, stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    n_hit = 0
    for o in outputs[0].outputs:
        if answer_matches(extract_answer(o.text), gt):
            n_hit += 1
    return n_hit / max(1, m_rollouts)


def sample_chain(
    llm: LLM, problem: str, k: int, temperature: float = 0.9,
) -> list[list[str]]:
    prompt = f"Problem: {problem}\nSolution:\n"
    sp = SamplingParams(temperature=temperature, max_tokens=800, n=k,
                        stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    return [parse_steps(o.text) for o in outputs[0].outputs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-model", required=True)
    ap.add_argument("--seed-problems", required=True)
    ap.add_argument("--k-samples", type=int, default=6,
                    help="문제당 샘플링할 CoT 체인 수")
    ap.add_argument("--m-rollouts", type=int, default=8,
                    help="각 스텝 value 추정에 사용할 rollout 수")
    ap.add_argument("--max-steps-per-chain", type=int, default=10,
                    help="체인당 라벨링할 최대 스텝 수 (긴 체인 cost 제어)")
    ap.add_argument("--output", default="data_pipeline_fullstepdpo/output/step_values.jsonl")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(model=args.ref_model, dtype="bfloat16", gpu_memory_utilization=0.85)

    # unique problem만
    seen = set()
    problems = []
    with open(args.seed_problems, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["problem_id"] in seen:
                continue
            seen.add(row["problem_id"])
            problems.append(row)
    print(f"[load] {len(problems)} unique problems")

    n_rows = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, prob in enumerate(problems):
            chains = sample_chain(llm, prob["question"], k=args.k_samples)
            for sample_idx, steps in enumerate(chains):
                if len(steps) < 2:
                    continue
                limit = min(len(steps), args.max_steps_per_chain)
                for t in range(1, limit + 1):
                    prefix = steps[:t]
                    v = mc_step_value(
                        llm, prob["question"], prefix, prob["gt_answer"],
                        m_rollouts=args.m_rollouts,
                    )
                    fout.write(json.dumps({
                        "problem_id": prob["problem_id"],
                        "problem": prob["question"],
                        "ground_truth": prob["gt_answer"],
                        "prefix_until_step": prefix,
                        "step_idx": t,
                        "step_value": v,
                        "sample_idx": sample_idx,
                    }, ensure_ascii=False) + "\n")
                    n_rows += 1
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(problems)}] step-value rows: {n_rows}")

    print(f"Done. {n_rows} step-value rows → {out_path}")


if __name__ == "__main__":
    main()
