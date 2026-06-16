"""data_pipeline/shared_sampling.py

Stage 3-pre: π_ref 1회 샘플링 + cascade verifier 1회 호출.
세 모드(Step-DPO / Full-Step GPT-4o / Full-Step PRM)가 *공유* 사용한다.

이 스크립트가 만드는 산출물:
  data_pipeline/output/samples_with_persona_labels.jsonl
    한 행 = 한 (problem × persona × sample_idx) chain.
    {
      "problem_id", "problem", "ground_truth",
      "persona_id", "persona_tag",
      "sample_idx": 3,
      "steps": ["Step 1: ...", "Step 2: ..."],
      "step_persona_labels": [
        {"verdict": "persona_ok"|"reject_persona",
         "confidence": float, "stage": "A"|"B"|"C",
         "trigger_term": str|null, "evidence_code": str|null,
         "reasoning": str},
         ...                    # len == len(steps)
      ]
    }

각 모드의 Stage 3 스크립트는 이 jsonl을 --samples-path로 받아 sampling/verify
단계를 모두 skip한다. → GPU 시간 + verifier API 비용 약 67% 절감.

Resume: 같은 (problem_id, persona_id, sample_idx)가 이미 있으면 skip하고
부족분만 새로 생성. 디스크 append 모드.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI  # noqa: E402

try:
    from vllm import LLM, SamplingParams  # type: ignore
except ImportError:
    from inference_backend import (  # type: ignore
        TransformersLLM as LLM,
        TransformersSamplingParams as SamplingParams,
    )

from utils import parse_steps, load_personas  # noqa: E402
from openai_client import make_openai_client  # noqa: E402
from persona_verifier import PersonaVerifier  # noqa: E402


def sample_chains(
    llm, persona_tag: str, problem: str, k: int, temperature: float = 0.9,
) -> list[list[str]]:
    persona_prefix = f"{persona_tag}\n" if persona_tag else ""
    prompt = f"{persona_prefix}Problem: {problem}\nSolution:\n"
    sp = SamplingParams(temperature=temperature, max_tokens=800, n=k,
                        stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    return [parse_steps(o.text) for o in outputs[0].outputs]


def load_done_keys(path: Path) -> dict[tuple[str, str], int]:
    """이미 저장된 (problem_id, persona_id) → max sample_idx + 1 카운트."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    if not path.exists():
        return counts
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r.get("problem_id", ""), r.get("persona_id", ""))
            counts[key] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-model", required=True, help="SFT-trained π_ref 경로")
    ap.add_argument("--seed-problems", required=True,
                    help="(problem × persona) jsonl")
    ap.add_argument("--personas-path",
                    default=str(REPO_ROOT / "personas.json"))
    ap.add_argument("--k-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-rows", type=int, default=0,
                    help="디버그용 (problem, persona) 행 수 제한.")
    ap.add_argument("--output",
                    default="data_pipeline/output/samples_with_persona_labels.jsonl")

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
                    default="data_pipeline/output/stage_log_shared.jsonl")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # cascade verifier
    gpt_client = make_openai_client()
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
    if args.stage_log_path and not Path(args.stage_log_path).exists():
        Path(args.stage_log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stage_log_path).write_text("")

    # π_ref
    llm = LLM(model=args.ref_model, dtype="bfloat16",
              gpu_memory_utilization=0.85)
    persona_by_id = {p["id"]: p for p in load_personas(args.personas_path)}

    # seed problems
    rows = []
    with open(args.seed_problems, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.max_rows:
        rows = rows[: args.max_rows]
    print(f"[load] {len(rows)} (problem, persona) rows; "
          f"{len(persona_by_id)} personas; K={args.k_samples}")

    # resume
    done_counts = load_done_keys(out_path)
    if done_counts:
        print(f"[resume] {sum(done_counts.values())} samples already exist "
              f"across {len(done_counts)} (problem, persona) keys")

    n_new = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, prob in enumerate(rows):
            persona = persona_by_id.get(prob.get("persona", ""))
            if persona is None:
                continue
            key = (prob["problem_id"], persona["id"])
            already = done_counts.get(key, 0)
            need = args.k_samples - already
            if need <= 0:
                continue

            verifier.problem_context = f"{prob['problem_id']}::{persona['id']}"
            chains = sample_chains(
                llm, persona.get("tag", ""), prob["question"],
                k=need, temperature=args.temperature,
            )

            for offset, steps in enumerate(chains):
                sample_idx = already + offset
                if len(steps) < 2:
                    # 짧은 chain은 학습 신호 약함. 저장은 하되 labels는 빈 리스트.
                    fout.write(json.dumps({
                        "problem_id": prob["problem_id"],
                        "problem": prob["question"],
                        "ground_truth": prob["gt_answer"],
                        "persona_id": persona["id"],
                        "persona_tag": persona.get("tag", ""),
                        "sample_idx": sample_idx,
                        "steps": steps,
                        "step_persona_labels": [],
                    }, ensure_ascii=False) + "\n")
                    n_new += 1
                    continue

                step_labels = []
                for j, step in enumerate(steps):
                    res = verifier.verify_step(step, persona, prefix=steps[:j])
                    step_labels.append({
                        "verdict": res.verdict,
                        "confidence": res.confidence,
                        "stage": res.stage,
                        "trigger_term": res.trigger_term,
                        "evidence_code": res.evidence_code,
                        "first_introduced": res.first_introduced,
                        "reasoning": res.reasoning,
                    })

                fout.write(json.dumps({
                    "problem_id": prob["problem_id"],
                    "problem": prob["question"],
                    "ground_truth": prob["gt_answer"],
                    "persona_id": persona["id"],
                    "persona_tag": persona.get("tag", ""),
                    "sample_idx": sample_idx,
                    "steps": steps,
                    "step_persona_labels": step_labels,
                }, ensure_ascii=False) + "\n")
                n_new += 1

            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(rows)}] new samples: {n_new} "
                      f"cascade: {verifier.dump_counters()}")

    print(f"Done. new samples: {n_new} → {out_path}")
    print(f"Final cascade counters: {verifier.dump_counters()}")


if __name__ == "__main__":
    main()
