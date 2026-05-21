"""data_pipeline_fullstepdpo/3c_score_and_pack.py

Full-Step DPO Stage 3c: 학습 데이터 패킹.

  1) π_ref(SFT 모델)로 새 K개의 CoT 체인 샘플
  2) 학습된 PRM(3b 산출)으로 체인 내 *모든 스텝*에 reward r_i ∈ [0,1] 부여
  3) (chain, [r_1,...,r_T])를 한 행으로 저장 — 페어로 분해 안 함.

출력 한 행:
  {
    "problem_id": "...", "problem": "...", "ground_truth": "18",
    "chain": [
      {"step": "Step 1: ...", "reward": 0.91},
      {"step": "Step 2: ...", "reward": 0.83},
      ...
    ],
    "final_correct": false,
    "sample_idx": 3
  }

학습 측 Full-Step DPO 손실은 이 reward 시퀀스를 사용해 per-step gradient
weight를 동적으로 산정한다 (별도 손실 모듈에서 구현).
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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

from utils import parse_steps  # noqa: E402

# 3b에서 정의한 PRM 클래스를 그대로 import해서 가중치 로드
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
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


def load_prm(path: str, base_model: str, device: str) -> tuple[PRM, "AutoTokenizer"]:
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = PRM(base_model).to(device)
    state = torch.load(Path(path) / "prm_state.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, tok


@torch.no_grad()
def score_step(prm: PRM, tok, problem: str, prefix_steps: list[str],
               device: str, max_len: int = 1024) -> float:
    prefix_text = "\n".join(prefix_steps)
    text = f"Problem: {problem}\nSolution:\n{prefix_text}"
    enc = tok(text, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    r = prm(enc.input_ids, enc.attention_mask).item()
    return float(r)


def sample_chains(llm: LLM, problem: str, k: int, temperature: float = 0.9) -> list[list[str]]:
    prompt = f"Problem: {problem}\nSolution:\n"
    sp = SamplingParams(temperature=temperature, max_tokens=800, n=k,
                        stop=["Problem:", "\n\n\n"])
    outputs = llm.generate([prompt], sp)
    return [parse_steps(o.text) for o in outputs[0].outputs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-model", required=True)
    ap.add_argument("--prm-model", required=True, help="3b_train_prm.py 산출 디렉토리")
    ap.add_argument("--prm-base-model", default=None,
                    help="PRM의 backbone (default: --ref-model과 동일)")
    ap.add_argument("--seed-problems", required=True)
    ap.add_argument("--k-samples", type=int, default=8)
    ap.add_argument("--output", default="data_pipeline_fullstepdpo/output/chains_fullstepdpo.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(model=args.ref_model, dtype="bfloat16", gpu_memory_utilization=0.85)
    prm, tok = load_prm(args.prm_model,
                        base_model=args.prm_base_model or args.ref_model,
                        device=device)

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

    n_chains = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for i, prob in enumerate(problems):
            chains = sample_chains(llm, prob["question"], k=args.k_samples)
            for sample_idx, steps in enumerate(chains):
                if len(steps) < 2:
                    continue
                scored = []
                for t in range(1, len(steps) + 1):
                    r = score_step(prm, tok, prob["question"], steps[:t], device)
                    scored.append({"step": steps[t - 1], "reward": r})
                final_pred = extract_answer("\n".join(steps))
                fout.write(json.dumps({
                    "problem_id": prob["problem_id"],
                    "problem": prob["question"],
                    "ground_truth": prob["gt_answer"],
                    "chain": scored,
                    "final_correct": answer_matches(final_pred, prob["gt_answer"]),
                    "sample_idx": sample_idx,
                }, ensure_ascii=False) + "\n")
                n_chains += 1
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(problems)}] chains: {n_chains}")

    print(f"Done. {n_chains} scored chains → {out_path}")


if __name__ == "__main__":
    main()
