"""data_pipeline_fullstepdpo/3c_score_and_pack.py

Full-Step DPO Stage 3c: 학습 데이터 패킹 (2-channel).

  1) π_ref로 (problem × persona) 단위 K개 CoT 체인 샘플.
  2) 학습된 2-head PRM으로 각 step에 (r_math, r_persona) 동시 부여.
  3) 추가 안전망: PersonaVerifier도 같이 호출해 cascade verdict를 함께 저장
     (PRM이 놓치는 hard violation을 catch — 학습 loss에서 가중치로 활용 가능).

출력 한 행:
  {
    "problem_id", "problem", "ground_truth", "persona_id", "persona_tag",
    "chain": [
      {"step": "Step 1: ...",
       "r_math": 0.91, "r_persona": 0.98,
       "verifier_verdict": "persona_ok",
       "verifier_stage": "A"|"B"|"C",
       "evidence_code": null,
       "trigger_term": null},
      ...
    ],
    "final_correct": false,
    "sample_idx": 3
  }

학습 측 Full-Step DPO loss는 per-step gradient weight를:
  w_t = α * r_math_t + β * r_persona_t   (또는 multiplicative)
로 산정한다 (loss 모듈에서 alpha/beta config로 노출).
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from importlib import import_module
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

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

# 3b의 PRM 클래스를 그대로 import
sys.path.insert(0, str(Path(__file__).resolve().parent))
_prm_mod = import_module("3b_train_prm")
PRM = _prm_mod.PRM


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


def load_prm(path: str, base_model: str, device: str) -> tuple[PRM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = PRM(base_model).to(device)
    state = torch.load(Path(path) / "prm_state.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, tok


@torch.no_grad()
def score_step(prm: PRM, tok, persona_tag: str, problem: str,
               prefix_steps: list[str], device: str,
               max_len: int = 1024) -> dict:
    prefix_text = "\n".join(prefix_steps)
    text = (f"{persona_tag}\n" if persona_tag else "") \
           + f"Problem: {problem}\nSolution:\n{prefix_text}"
    enc = tok(text, truncation=True, max_length=max_len,
              return_tensors="pt").to(device)
    out = prm(enc.input_ids, enc.attention_mask)
    return {"r_math": float(out["r_math"].item()),
            "r_persona": float(out["r_persona"].item())}


def sample_chains(llm: LLM, persona_tag: str, problem: str, k: int,
                  temperature: float = 0.9) -> list[list[str]]:
    persona_prefix = f"{persona_tag}\n" if persona_tag else ""
    prompt = f"{persona_prefix}Problem: {problem}\nSolution:\n"
    sp = SamplingParams(temperature=temperature, max_tokens=800, n=k,
                        stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    return [parse_steps(o.text) for o in outputs[0].outputs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-model", required=True)
    ap.add_argument("--prm-model", required=True,
                    help="3b 산출 디렉토리 (prm_state.pt + prm_config.json)")
    ap.add_argument("--prm-base-model", default=None)
    ap.add_argument("--seed-problems", required=True,
                    help="(problem × persona) 행 jsonl")
    ap.add_argument("--personas-path",
                    default=str(REPO_ROOT / "personas.json"))
    ap.add_argument("--k-samples", type=int, default=8)
    ap.add_argument("--output",
                    default="data_pipeline_fullstepdpo/output/chains_fullstepdpo.jsonl")

    # cascade verifier (추가 안전망)
    ap.add_argument("--verifier-base-url", default="http://localhost:8001/v1")
    ap.add_argument("--verifier-model",
                    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--verifier-api-key", default="EMPTY")
    ap.add_argument("--stage-b-threshold", type=float, default=0.85)
    ap.add_argument("--disable-stage-b", action="store_true")
    ap.add_argument("--disable-stage-c", action="store_true")
    ap.add_argument("--gpt-model", default="gpt-4o")
    ap.add_argument("--disable-verifier", action="store_true",
                    help="cascade verifier 호출을 완전히 끔 (PRM만 사용)")
    ap.add_argument("--stage-log-path",
                    default="data_pipeline_fullstepdpo/output/stage_log_3c.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(model=args.ref_model, dtype="bfloat16",
              gpu_memory_utilization=0.65)
    prm, tok = load_prm(args.prm_model,
                        base_model=args.prm_base_model or args.ref_model,
                        device=device)

    # cascade verifier (PRM 위 추가 안전망)
    verifier = None
    if not args.disable_verifier:
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
        if args.stage_log_path:
            Path(args.stage_log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.stage_log_path).write_text("")

    persona_by_id = {p["id"]: p for p in load_personas(args.personas_path)}

    rows = []
    with open(args.seed_problems, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"[load] {len(rows)} (problem, persona) rows")

    n_chains = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, prob in enumerate(rows):
            persona = persona_by_id.get(prob.get("persona", ""))
            if persona is None:
                continue
            persona_tag = persona.get("tag", "")
            if verifier is not None:
                verifier.problem_context = (
                    f"{prob.get('problem_id','?')}::{persona['id']}"
                )

            chains = sample_chains(llm, persona_tag, prob["question"],
                                   k=args.k_samples)
            for sample_idx, steps in enumerate(chains):
                if len(steps) < 2:
                    continue
                scored = []
                for t in range(1, len(steps) + 1):
                    r = score_step(prm, tok, persona_tag, prob["question"],
                                   steps[:t], device)
                    entry = {
                        "step": steps[t - 1],
                        "r_math": r["r_math"],
                        "r_persona": r["r_persona"],
                    }
                    if verifier is not None:
                        p_res = verifier.verify_step(
                            steps[t - 1], persona, prefix=steps[: t - 1],
                        )
                        entry.update({
                            "verifier_verdict": p_res.verdict,
                            "verifier_stage": p_res.stage,
                            "verifier_confidence": p_res.confidence,
                            "evidence_code": p_res.evidence_code,
                            "trigger_term": p_res.trigger_term,
                        })
                    scored.append(entry)

                final_pred = extract_answer("\n".join(steps))
                fout.write(json.dumps({
                    "problem_id": prob["problem_id"],
                    "problem": prob["question"],
                    "ground_truth": prob["gt_answer"],
                    "persona_id": persona["id"],
                    "persona_tag": persona_tag,
                    "chain": scored,
                    "final_correct": answer_matches(final_pred, prob["gt_answer"]),
                    "sample_idx": sample_idx,
                }, ensure_ascii=False) + "\n")
                n_chains += 1
            if (i + 1) % 50 == 0:
                msg = f"[{i+1}/{len(rows)}] chains: {n_chains}"
                if verifier is not None:
                    msg += f" cascade: {verifier.dump_counters()}"
                print(msg)

    print(f"Done. {n_chains} scored chains → {out_path}")
    if verifier is not None:
        print(f"Final cascade counters: {verifier.dump_counters()}")


if __name__ == "__main__":
    main()
