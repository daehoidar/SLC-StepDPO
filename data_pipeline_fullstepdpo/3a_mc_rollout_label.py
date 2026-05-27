"""data_pipeline_fullstepdpo/3a_mc_rollout_label.py

Full-Step DPO Stage 3a: 각 스텝의 두 채널 라벨 산출.

  step_value         := MC rollout 정답 도달률 (기존)
  persona_validity   := PersonaVerifier 결과 → reject_persona이면 0, 아니면 1 (신규)

→ PRM(3b)이 (r_math, r_persona) 2-head로 학습된다. 현재까지 페르소나 신호가
0이었던 Full-Step-DPO에 belief-conditional 신호를 주입한다.

출력 한 행 = (problem, persona, sample, step):
  {
    "problem_id", "problem", "ground_truth", "persona_id", "persona_tag",
    "prefix_until_step":  ["Step 1: ...", "Step 2: ..."],
    "step_idx": 2,
    "step_value": 0.625,            # 정답 rollout 비율
    "persona_validity": 1.0,        # 1.0 = persona_ok, 0.0 = reject_persona
    "verifier_stage": "A"|"B"|"C",
    "evidence_code": "[6수01-08]" | null,
    "trigger_term": str | null,
    "sample_idx": 3
  }

입력 (--seed-problems)은 (problem × persona) 행 jsonl로 가정.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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
from persona_verifier import PersonaVerifier  # noqa: E402


ANS_RE = re.compile(r"(?:answer|Answer)\s*(?:is)?\s*[:=]?\s*([\-\d\./]+)")


def extract_answer(text: str) -> str:
    m = ANS_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def answer_matches(pred: str, gt: str) -> bool:
    p, g = pred.strip().lower(), gt.strip().lower()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return g in p or p in g


def build_prompt(persona_tag: str, problem: str, prefix_steps: list[str]) -> str:
    prefix_text = "\n".join(prefix_steps)
    if prefix_text:
        prefix_text += "\n"
    persona_prefix = f"{persona_tag}\n" if persona_tag else ""
    return f"{persona_prefix}Problem: {problem}\nSolution:\n{prefix_text}"


def mc_step_value(
    llm: LLM, persona_tag: str, problem: str, prefix_steps: list[str], gt: str,
    m_rollouts: int, temperature: float = 1.0,
) -> float:
    prompt = build_prompt(persona_tag, problem, prefix_steps)
    sp = SamplingParams(temperature=temperature, max_tokens=400,
                        n=m_rollouts, stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    n_hit = 0
    for o in outputs[0].outputs:
        if answer_matches(extract_answer(o.text), gt):
            n_hit += 1
    return n_hit / max(1, m_rollouts)


def sample_chain(
    llm: LLM, persona_tag: str, problem: str, k: int, temperature: float = 0.9,
) -> list[list[str]]:
    persona_prefix = f"{persona_tag}\n" if persona_tag else ""
    prompt = f"{persona_prefix}Problem: {problem}\nSolution:\n"
    sp = SamplingParams(temperature=temperature, max_tokens=800, n=k,
                        stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    return [parse_steps(o.text) for o in outputs[0].outputs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-model", required=True)
    ap.add_argument("--seed-problems", required=True,
                    help="(problem × persona) 행 jsonl")
    ap.add_argument("--personas-path",
                    default=str(REPO_ROOT / "personas.json"))
    ap.add_argument("--k-samples", type=int, default=6)
    ap.add_argument("--m-rollouts", type=int, default=8)
    ap.add_argument("--max-steps-per-chain", type=int, default=10)
    ap.add_argument("--output",
                    default="data_pipeline_fullstepdpo/output/step_values.jsonl")

    # cascade verifier
    ap.add_argument("--verifier-base-url", default="http://localhost:8001/v1")
    ap.add_argument("--verifier-model",
                    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--verifier-api-key", default="EMPTY")
    ap.add_argument("--stage-b-threshold", type=float, default=0.85)
    ap.add_argument("--disable-stage-b", action="store_true")
    ap.add_argument("--disable-stage-c", action="store_true")
    ap.add_argument("--gpt-model", default="gpt-4o",
                    help="Stage C용 외부 모델")
    ap.add_argument("--stage-log-path",
                    default="data_pipeline_fullstepdpo/output/stage_log.jsonl")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(model=args.ref_model, dtype="bfloat16",
              gpu_memory_utilization=0.85)

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

    persona_by_id = {p["id"]: p for p in load_personas(args.personas_path)}

    rows = []
    with open(args.seed_problems, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"[load] {len(rows)} (problem, persona) rows")

    if args.stage_log_path:
        Path(args.stage_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stage_log_path).write_text("")

    n_rows = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, prob in enumerate(rows):
            persona = persona_by_id.get(prob.get("persona", ""))
            if persona is None:
                continue
            persona_tag = persona.get("tag", "")
            verifier.problem_context = f"{prob.get('problem_id','?')}::{persona['id']}"

            chains = sample_chain(llm, persona_tag, prob["question"],
                                  k=args.k_samples)
            for sample_idx, steps in enumerate(chains):
                if len(steps) < 2:
                    continue
                limit = min(len(steps), args.max_steps_per_chain)
                for t in range(1, limit + 1):
                    prefix = steps[:t]
                    # 수학 채널
                    v_math = mc_step_value(
                        llm, persona_tag, prob["question"], prefix,
                        prob["gt_answer"], m_rollouts=args.m_rollouts,
                    )
                    # 페르소나 채널 — 현재 step(=prefix[-1])이 위반인지
                    p_res = verifier.verify_step(
                        steps[t - 1], persona, prefix=steps[: t - 1],
                    )
                    v_persona = 0.0 if p_res.verdict == "reject_persona" else 1.0

                    fout.write(json.dumps({
                        "problem_id": prob["problem_id"],
                        "problem": prob["question"],
                        "ground_truth": prob["gt_answer"],
                        "persona_id": persona["id"],
                        "persona_tag": persona_tag,
                        "prefix_until_step": prefix,
                        "step_idx": t,
                        "step_value": v_math,
                        "persona_validity": v_persona,
                        "verifier_stage": p_res.stage,
                        "evidence_code": p_res.evidence_code,
                        "trigger_term": p_res.trigger_term,
                        "sample_idx": sample_idx,
                    }, ensure_ascii=False) + "\n")
                    n_rows += 1
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(rows)}] step-rows: {n_rows} "
                      f"cascade: {verifier.dump_counters()}")

    print(f"Done. {n_rows} step rows → {out_path}")
    print(f"Final cascade counters: {verifier.dump_counters()}")


if __name__ == "__main__":
    main()
