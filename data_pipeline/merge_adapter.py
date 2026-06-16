"""data_pipeline/merge_adapter.py

LoRA 어댑터를 base 모델에 머지해 *단일 standalone 모델*로 저장한다.

이유: 샘플링 백엔드(inference_backend.TransformersLLM / vLLM)는 LoRA 어댑터를
직접 로드하지 못한다. 머지된 모델을 만들어두면 모든 백엔드가 일반 모델처럼
바로 로드할 수 있다 (Stage 3 logprob 분석에서도 그대로 사용 가능).

Usage:
    python data_pipeline/merge_adapter.py \
        --base-model Qwen/Qwen3-1.7B \
        --adapter checkpoints/sft_qwen3_1.7b_eos \
        --output  checkpoints/sft_qwen3_1.7b_eos_merged
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = Path(args.output)
    if (out / "config.json").exists():
        print(f"[skip] 이미 머지된 모델 존재: {out}")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[merge] base={args.base_model} + adapter={args.adapter}")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()  # LoRA 가중치를 base에 흡수

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    # 토크나이저는 어댑터 디렉토리에 있으면 그걸, 없으면 base에서
    try:
        tok = AutoTokenizer.from_pretrained(args.adapter)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.base_model)
    tok.save_pretrained(str(out))
    print(f"[done] merged model → {out}")


if __name__ == "__main__":
    main()
